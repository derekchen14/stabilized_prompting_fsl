import os, sys, pdb
import numpy as np
import random

from torch import nn, no_grad
from tqdm import tqdm as progress_bar
from components.logger import ExperienceLogger
from components.detector import ExemplarDetective

from utils.help import *
from utils.process import process_data, get_dataloader
from utils.arguments import solicit_params
from utils.load import load_tokenizer, load_model, load_data, load_best_model, load_support
from utils.evaluate import eval_quantify, eval_qualify, test_quantify, parse_output
from assets.static_vars import device, debug_break, STOP_TOKENS
from torch.cuda.amp import autocast, GradScaler

def run_train(args, model, datasets, exp_logger, detective):
  dataset, dev_dataset = datasets['train'], datasets['dev']
  train_dataloader = get_dataloader(args, dataset)
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs
  optimizer, scheduler = setup_optimization(args, model, total_steps)
  scaler = GradScaler()
  exp_logger.checkpoint_interval = int((len(train_dataloader)/args.chunk_per_epoch) // 1000 * 1000)

  if args.task == 'meta_learn':
    dataset.add_detective(detective)
    dev_dataset.add_detective(detective)
    if exp_logger.use_chunks:
      dev_dataset.data = random.sample(dev_dataset.data, len(dev_dataset)//6)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader, args.percent)
    model.train()

    for step, batch in enumerate(train_dataloader):
      exp_logger.start_chunk(step)    # start chunk here
      inputs, targets = dataset.collate(args, batch)
      review_inputs(args, inputs, targets, datasets['train'].tokenizer)
      with autocast(dtype=torch.bfloat16):
        outputs = model(**inputs, labels=targets)
        exp_logger.tr_loss += outputs.loss.item()
        loss = outputs.loss / args.grad_accum_steps
      scaler.scale(loss).backward()

      if (step + 1) % args.grad_accum_steps == 0:
        scaler.step(optimizer)  
        scaler.update()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
      
      exp_logger.log_train(step, scheduler)
      if exp_logger.train_stop(args, step, debug_break): break

      if exp_logger.use_chunks and step > 0 and step % exp_logger.checkpoint_interval == 0:
        if exp_logger.chunk_num > 2:
          if args.task == 'meta_learn' and args.do_leave:
            run_eval(args, model, dev_dataset, exp_logger)
            eval_res = run_leftout(args, model, dev_dataset, exp_logger)
          else:
            eval_res = run_eval(args, model, dev_dataset, exp_logger)

          if eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
            exp_logger.best_score = eval_res
            exp_logger.save_best_model(model, tokenizer, args.prune_keep)
          
        early_stop = exp_logger.end_chunk()
        if early_stop: break

    if not exp_logger.use_chunks:
      # only do epoch validation for non-meta training
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
          maxl = 512 if args.model == 't5' else 1024
        else:
          maxl = inputs['input_ids'].shape[1] + 16
        with no_grad():
          outputs = model.generate(**inputs, max_length=maxl, repetition_penalty=args.threshold,
                                              early_stopping=True, temperature=args.temperature, 
                                              forced_eos_token_id=tokenizer.eos_token_id)
        output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)
        output_strings = desemble(args, output_strings)
        for target, output_str in zip(target_dict, output_strings):
          state_key = f"{target['domain']}-{target['slot']}"
          if len(output_str) < 50:
            prior_pred_state[global_id][state_key] = output_str
          else:
            output_str = output_str[:50]
            prior_pred_state[global_id][state_key] = output_str
    if exp_logger.log_eval(args.qualify, output_strings, target_dict):
      results = test_quantify(args, prior_pred_state, all_targets, exp_logger, tokenizer)
      dataset.detective[0].report(args.verbose, args.task)
  
  if args.do_save:
    output_name = f'{args.prompt_style}_{args.num_shots}.json'
    output_results = {'preds': prior_pred_state, 'targets': all_targets }
    json.dump(output_results, open(os.path.join(save_path, output_name), 'w'), indent=2)

def run_leftout(args, model, dataset, exp_logger):
  tokenizer = dataset.tokenizer
  bs, num_exp, leftout_ratio = args.batch_size, len(dataset.leftout), 0.5
  description = f"Evaluating {args.left_out} with ratio {leftout_ratio}"
  all_outputs, all_targets = [], []

  for idx in progress_bar(range(0, num_exp, bs), total=num_exp//bs, desc=description):
    if random.random() < leftout_ratio: continue  # sample from the data to speed things up
    batch = dataset.leftout[idx:idx+bs]
    inputs, target_dict = dataset.collate(args, batch)
    all_targets.extend(target_dict)   # notice this is "extend", not "append"
    
    maxl = inputs['input_ids'].shape[1] + 16
    with no_grad():
      outputs = model.generate(**inputs, max_length=maxl, early_stopping=True,
                          repetition_penalty=args.threshold, temperature=args.temperature)
    output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)
    all_outputs.extend(output_strings)
  
  eval_qualify(args, all_outputs, all_targets)
  return eval_quantify(args, all_outputs, all_targets, exp_logger, tokenizer)

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

    maxl = inputs['input_ids'].shape[1] + 16
    with no_grad():
      # defaults to greedy sampling, for param details see https://huggingface.co/docs/transformers/
      #        v4.15.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate 
      outputs = model.generate(**inputs, max_length=maxl, early_stopping=True,
                          repetition_penalty=args.threshold, temperature=args.temperature)
    output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)
    all_outputs.extend(output_strings)

    if exp_logger.log_eval(args.qualify, output_strings, target_dict):
      results = eval_quantify(args, all_outputs, all_targets, exp_logger, tokenizer)
    if args.debug and exp_logger.eval_step >= debug_break: break

  return results

def check_support(args, datasets, tokenizer):
  if args.task in ['meta_learn', 'pre_train']:
    supports = load_support(args, tokenizer)
    datasets['train'].add_support(supports, args.left_out)
    datasets['dev'].add_support(supports, args.left_out)
  return datasets

def ensemble_of_detectives(args, datasets):
  import copy
  # cosine
  args_cos = copy.deepcopy(args)
  args_cos.ignore_cache = True
  args_cos.search_method = "cosine"
  detective1 = ExemplarDetective(args_cos, datasets['train'])
  # cosine with pretrained embedding
  args_sbert = copy.deepcopy(args)
  args_sbert.ignore_cache = False
  args_sbert.search_method = "cosine"
  detective3 = ExemplarDetective(args_sbert, datasets['train'])
  # euclidean
  args_euc = copy.deepcopy(args)
  args_euc.ignore_cache = True
  args_euc.search_method = "euclidean"
  detective2 = ExemplarDetective(args_euc, datasets['train'])

  detective = [detective1, detective2, detective3]
  if args.ensemble <= 3:
    detective = detective[:args.ensemble]
  else:
    # random
    for _ in range(args.ensemble - 3):
      args_rad = copy.deepcopy(args)
      args_rad.ignore_cache = True
      args_rad.search_method = "random"
      detective.append(ExemplarDetective(args_rad, datasets['train']))
  return detective


if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  args, save_path = check_directories(args)
  set_seed(args)

  reformat_data(args)
  raw_data = load_data(args)
  tokenizer = load_tokenizer(args)
  datasets, ontology = process_data(args, raw_data, tokenizer)
  exp_logger = ExperienceLogger(args, ontology, save_path)
  if args.ensemble > 1:
    detective = ensemble_of_detectives(args, datasets)
  else:
    detective = ExemplarDetective(args, datasets['train']) 

  if args.do_train:
    model = load_model(args, ontology, tokenizer, save_path)
    datasets = check_support(args, datasets, tokenizer)
    run_train(args, model, datasets, exp_logger, detective)
  elif args.do_eval:
    run_test(args, datasets['test'], exp_logger, detective)
