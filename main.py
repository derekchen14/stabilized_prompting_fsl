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
from utils.load import load_tokenizer, load_model, load_data, load_best_model
from utils.load import load_support
from utils.evaluate import eval_quantify, eval_qualify, parse_output
from assets.static_vars import device, debug_break, STOP_TOKENS

def run_train(args, model, datasets, exp_logger, detective):
  dataset = datasets['train']
  train_dataloader = get_dataloader(args, dataset)
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs
  optimizer, scheduler = setup_optimization(args, model, total_steps)
  exp_logger.update_optimization(optimizer, scheduler)
  dataset.add_detective(detective)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    model.train()
    for step, batch in enumerate(train_dataloader):
      inputs, targets = dataset.collate(args, batch)
      review_inputs(args, targets, datasets['train'].tokenizer)
      outputs = model(**inputs, labels=targets)
      exp_logger.tr_loss += outputs.loss.item()
      loss = outputs.loss / args.grad_accum_steps
      loss.backward()

      if (step + 1) % args.grad_accum_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        exp_logger.optimizer.step()  # backprop to update the weights
        exp_logger.scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        exp_logger.log_train(step)
      if args.debug and step >= debug_break*args.log_interval:
        break

    eval_res = run_eval(args, model, datasets['dev'], exp_logger, detective)
    if eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
      exp_logger.best_score = eval_res
      exp_logger.save_best_model(model, tokenizer, args.prune_keep)
    early_stop = exp_logger.end_epoch()
    if early_stop: break

  return model

def run_test(args, dataset, exp_logger, detective):
  ontology, tokenizer = exp_logger.ontology, dataset.tokenizer
  dataset.add_detective(detective)
  exp_logger.eval_step = 0

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
        all_targets[global_id].extend(target_dict) #  all the target labels for this turn 

        if args.task == 'in_context':
          maxl = 2048 if args.size == 'large' else 1024
        else:
          maxl = inputs['input_ids'].shape[1] + 12

        with no_grad():
          outputs = model.generate(**inputs, max_length=maxl, early_stopping=True)
        output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)
        exp_logger.eval_step += 1

        for target, output_str in zip(target_dict, output_strings):
          state_key = f"{target['domain']}-{target['slot']}"
          pred_value = parse_output(args, output_str)
          prior_pred_state[global_id][state_key] = pred_value
        if args.debug and exp_logger.eval_step >= (debug_break * 200): break

  if args.quantify:
    results = test_quantify(args, prior_pred_state, all_targets, exp_logger, tokenizer)
  # elif args.qualify:
  # results = eval_qualify(args, prior_pred_state, all_targets, exp_logger)
  if args.do_save:
    output_name = f'{args.prompt_style}_lr{args.learning_rate}_clen{args.context_length}.json'
    json.dump(outputs, open(os.path.join(save_path, output_name), 'w'), indent=2)

def run_eval(args, model, dataset, exp_logger, detective):
  tokenizer = dataset.tokenizer
  dataset.add_detective(detective)
  all_outputs, all_targets = [], []
  exp_logger.eval_step = 0

  dataloader = get_dataloader(args, dataset, 'dev')
  ''' goes through model generation without backprop, rather than classification '''
  for batch in progress_bar(dataloader, total=len(dataloader)):
    inputs, target_dict = dataset.collate(args, batch, prior_pred_state)
    all_targets.extend(target_dict)   # notice this is "extend", not "append"

    maxl = inputs['input_ids'].shape[1] + 12
    with no_grad():
      # defaults to greedy sampling, for param details see https://huggingface.co/docs/transformers/
      #        v4.15.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate 
      outputs = model.generate(**inputs, max_length=maxl, early_stopping=True)
    output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)
    # output_strings = [output_strings[idx].replace("<pad>","")+" "+target_dict[idx]['value'] for idx in range(len(output_strings))]
    all_outputs.extend(output_strings)

    exp_logger.eval_step += 1
    exp_logger.eval_loss = 0  # no loss, since inference only
    if args.debug and exp_logger.eval_step >= debug_break: break

  results = eval_quantify(args, all_outputs, all_targets, exp_logger, tokenizer)
  return results

def check_support(args, datasets):
  if args.task == 'meta_learn':
    supports = load_support(args)
    datasets['train'].add_support(supports, args.left_out)
    datasets['dev'].add_support(supports, args.left_out)
  return datasets

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
  detective = ExemplarDetective(args, datasets['train'])

  if args.do_train:
    model = load_model(args, ontology, tokenizer, save_path)
    datasets = check_support(args, datasets)
    run_train(args, model, datasets, exp_logger, detective)
  elif args.do_eval:
    run_test(args, datasets['test'], exp_logger, detective)
