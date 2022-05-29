import os, sys, pdb
import numpy as np
import random
import deepspeed

from torch import nn, no_grad
from tqdm import tqdm as progress_bar
from components.logger import ExperienceLogger
from components.detector import ExemplarDetective
from components.trainer import DSTrainer
from tqdm import tqdm

from utils.help import *
from utils.process import process_data, get_dataloader
from utils.arguments import solicit_params
from utils.load import load_tokenizer, load_model, load_data, load_best_model, load_support
from utils.evaluate import eval_quantify, eval_qualify, test_quantify, parse_output
from assets.static_vars import device, debug_break, STOP_TOKENS
from transformers import HfArgumentParser, TrainingArguments
import megatron.mpu as mpu
# import mpu

# print(mpu.get_data_parallel_world_size())
# pdb.set_trace()
def run_train(args, model, datasets, exp_logger, detective):
  dataset, dev_dataset = datasets['train'], datasets['dev']
  train_dataloader = get_dataloader(args, dataset)
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs

  optimizer, scheduler = setup_optimization(args, model, total_steps)


  if args.deepspeed:

    deepspeed.init_distributed()
    mpu.initialize_model_parallel(tensor_model_parallel_size_=args.world_size)
    # args.batch_size = args.world_size
    args.per_device_train_batch_size = args.batch_size * mpu.get_data_parallel_world_size() / args.world_size
    args.hf_deepspeed_config = HfTrainerDeepSpeedConfig(args.deepspeed)
    args.hf_deepspeed_config.trainer_config_process(args)
    hf_deepspeed_config = args.hf_deepspeed_config
    hf_deepspeed_config.trainer_config_finalize(args, model, total_steps)
    config = hf_deepspeed_config.config
    # pdb.set_trace()
    config['train_micro_batch_size_per_gpu'] = args.batch_size
    # print(mpu.get_data_parallel_world_size())

    # pdb.set_trace()
    deepspeed_engine, optimizer, _, scheduler = deepspeed.initialize(
              model=model,
              model_parameters=list(filter(lambda p: p.requires_grad, model.parameters())),
              config_params=config,
              optimizer=optimizer,
              lr_scheduler=scheduler,
              mpu=mpu)
    model = deepspeed_engine
    optimizer = optimizer
    scheduler = scheduler

  exp_logger.update_optimization(optimizer, scheduler)
  
  if args.task == 'meta_learn':
    dataset.add_detective(detective)
    dev_dataset.add_detective(detective)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader, args.percent)
    model.train()

    for step, batch in (enumerate(train_dataloader)):
      inputs, targets = dataset.collate(args, batch)
      review_inputs(args, inputs, targets, datasets['train'].tokenizer)

      if args.deepspeed:
        kwargs = dict(device=args.device)
        # pdb.set_trace()
        # kwargs.update(dict(dtype=args.hf_deepspeed_config.dtype()))
        inputs['input_ids'] =  inputs['input_ids'].to(**kwargs)
        inputs['attention_mask'] =  inputs['attention_mask'].to(**kwargs)
        targets = targets.to(**kwargs)

      # pdb.set_trace()
      outputs = model(**inputs, labels=targets)
      exp_logger.tr_loss += outputs.loss.item()
      loss = outputs.loss / args.grad_accum_steps
      if args.deepspeed:
        loss = model.backward(loss)
      else:
        loss.backward()

      if step % 24 == 0:
        print('loss', loss)

      if (step + 1) % args.grad_accum_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if args.deepspeed:
          model.step()
        else:
          exp_logger.optimizer.step()  # backprop to update the weights
          exp_logger.scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        exp_logger.log_train(step)
      if exp_logger.train_stop(args, step, debug_break): break

    if args.task == 'meta_learn' and args.do_leave:
      run_leftout(args, model, dev_dataset, exp_logger)
    eval_res = run_eval(args, model, dev_dataset, exp_logger)
    if eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
      exp_logger.best_score = eval_res
      exp_logger.save_best_model(model, tokenizer, args.prune_keep)
    early_stop = exp_logger.end_epoch()
    if early_stop: break

  return model

def run_test(args, dataset, exp_logger, detective):
  ontology, tokenizer = exp_logger.ontology, dataset.tokenizer
  dataset.add_detective(detective)
  exp_logger.start_eval(len(dataset), args.eval_interval)
  
  if args.task in ['meta_learn', 'fine_tune']:
    model = load_best_model(args, exp_logger, tokenizer)
  else:
    model = load_model(args, ontology, tokenizer, exp_logger.save_path)

  all_targets = defaultdict(list)
  prior_pred_state = defaultdict(dict)
  for conversation in progress_bar(dataset.data, total=len(dataset)):
    for global_id, turn in conversation.items():
      # turn is a list of examples

      batches = batchify(args, turn, global_id, prior_pred_state)
      for batch in batches:
        inputs, target_dict = dataset.collate(args, batch)
        review_inputs(args, inputs, inputs['input_ids'], tokenizer)
        all_targets[global_id].extend(target_dict) #  all the target labels for this turn 

        if args.task == 'in_context':
          maxl = 2048 if args.size == 'large' else 1024
        else:
          maxl = inputs['input_ids'].shape[1] + 12

        with no_grad():
          outputs = model.generate(**inputs, max_length=maxl, repetition_penalty=args.threshold,
                                              early_stopping=True, temperature=args.temperature, 
                                              forced_eos_token_id=tokenizer.eos_token_id)
        output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)
       
        for target, output_str in zip(target_dict, output_strings):
          state_key = f"{target['domain']}-{target['slot']}"
          pred_value = parse_output(args, output_str)
          prior_pred_state[global_id][state_key] = pred_value
    if exp_logger.log_eval(args.qualify, output_strings, target_dict):
      results = test_quantify(args, prior_pred_state, all_targets, exp_logger, tokenizer)
      dataset.detective.report(args.verbose, args.task)
  
  if args.do_save:
    output_name = f'{args.prompt_style}_lr{args.learning_rate}_clen{args.context_length}.json'
    json.dump(results, open(os.path.join(save_path, output_name), 'w'), indent=2)

def run_leftout(args, model, dataset, exp_logger):
  tokenizer = dataset.tokenizer
  bs, num_exp = args.batch_size, len(dataset.leftout)
  description = f"Evaluating {args.left_out}"
  all_outputs, all_targets = [], []

  for idx in progress_bar(range(0, num_exp, bs), total=num_exp//bs, desc=description):
    if random.random() < 0.7: continue  # sample from the data to speed things up
    batch = dataset.leftout[idx:idx+bs]
    inputs, target_dict = dataset.collate(args, batch)
    all_targets.extend(target_dict)   # notice this is "extend", not "append"
    
    maxl = inputs['input_ids'].shape[1] + 12
    with no_grad():
      outputs = model.generate(**inputs, max_length=maxl, early_stopping=True,
                          repetition_penalty=args.threshold, temperature=args.temperature)
    output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)
    all_outputs.extend(output_strings)
  
  eval_quantify(args, all_outputs, all_targets, exp_logger, tokenizer)
  eval_qualify(args, all_outputs, all_targets)


def run_eval(args, model, dataset, exp_logger):
  tokenizer = dataset.tokenizer
  dataloader = get_dataloader(args, dataset, 'dev')
  num_batches = debug_break if args.debug else len(dataloader)
  exp_logger.start_eval(num_batches, args.eval_interval)
  all_outputs, all_targets = [], []
  
  ''' goes through model generation without backprop, rather than classification '''
  for batch in progress_bar(dataloader, total=len(dataloader)):
    inputs, target_dict = dataset.collate(args, batch)
    all_targets.extend(target_dict)   # notice this is "extend", not "append"

    maxl = inputs['input_ids'].shape[1] + 12
    with no_grad():
      # defaults to greedy sampling, for param details see https://huggingface.co/docs/transformers/
      #        v4.15.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate 
      if args.deepspeed:
        outputs = model.module.generate(**inputs, max_length=maxl, early_stopping=True,
                            repetition_penalty=args.threshold, temperature=args.temperature)
      else:
        outputs = model.generate(**inputs, max_length=maxl, early_stopping=True,
                            repetition_penalty=args.threshold, temperature=args.temperature)
    output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)
    all_outputs.extend(output_strings)

    if exp_logger.log_eval(args.qualify, output_strings, target_dict):
      results = eval_quantify(args, all_outputs, all_targets, exp_logger, tokenizer)
    if args.debug and exp_logger.eval_step >= debug_break: break

  return results

def check_support(args, datasets):
  if args.task == 'meta_learn':
    supports = load_support(args)
    datasets['train'].add_support(supports, args.left_out)
    datasets['dev'].add_support(supports, args.left_out)
  return datasets

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  args, save_path = check_directories(args)
  set_seed(args)
  # pdb.set_trace()

  if args.deepspeed:

    from transformers.deepspeed import HfTrainerDeepSpeedConfig
    training_args = TrainingArguments(output_dir=args.output_dir, fp16=args.fp16, fp16_backend=args.fp16_backend, 
                                      learning_rate=args.learning_rate, do_train=args.do_train, do_eval=args.do_eval,
                                      save_strategy="epoch", seed=args.seed,)
    for arg in vars(args):
      try:
        setattr(training_args, arg, getattr(args, arg))
      except:
        pdb.set_trace()
    args = training_args

  reformat_data(args)
  raw_data = load_data(args)
  tokenizer = load_tokenizer(args)
  datasets, ontology = process_data(args, raw_data, tokenizer)
  exp_logger = ExperienceLogger(args, ontology, save_path)
  detective = ExemplarDetective(args, datasets['train'])

  if args.do_train:
    model = load_model(args, ontology, tokenizer, save_path)
    datasets = check_support(args, datasets)
    run_train(args, model, datasets, exp_logger, detective)
  elif args.do_eval:
    run_test(args, datasets['test'], exp_logger, detective)



  # # parser = HfArgumentParser(ourarugments, TrainingArguments)
  # # training_args = parser.parse_args_into_dataclasses()

  # # training_args = TrainingArguments(output_dir=args.output_dir, deepspeed=args.deepspeed, fp16=args.fp16, 
  # #           do_train=args.do_train, do_eval=args.do_eval, do_predict=args.do_eval, learning_rate=args.learning_rate, 
  # #           num_train_epochs=args.n_epochs, logging_steps=args.log_interval, save_strategy="epoch", seed=args.seed, 
  # #           eval_steps=args.eval_interval,)

  # training_args = TrainingArguments(output_dir=args.output_dir)
  # for arg in vars(args):
  #   try:
  #     setattr(training_args, arg, getattr(args, arg))
  #   except:
  #     pdb.set_trace()
  # # print("*"*20)
  # # pdb.set_trace()
  # datasets = check_support(training_args, datasets)
  # model = load_model(training_args, ontology, tokenizer, save_path)

  # # pdb.set_trace()
  # # training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
  # # Initialize our Trainer
  # trainer = DSTrainer(
  #     model=model,
  #     args=training_args,
  #     train_dataset=datasets["train"],
  #     eval_dataset=datasets["dev"],
  #     tokenizer=tokenizer,
  # )


  # if args.do_train:
  #   trainer.train(exp_logger, detective)
  # elif args.do_eval:
  #   trainer.predict(exp_logger, detective)















