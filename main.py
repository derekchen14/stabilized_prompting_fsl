import os, sys, pdb
import numpy as np
import random

from torch import nn, no_grad
from tqdm import tqdm as progress_bar
from components.logger import ExperienceLogger

from utils.help import *
from utils.process import process_data, get_dataloader
from utils.arguments import solicit_params
from utils.load import load_tokenizer, load_model, load_data, load_best_model
from utils.load import load_support
from utils.evaluate import eval_quantify, eval_qualify
from assets.static_vars import device, debug_break, STOP_TOKENS

def run_train(args, model, datasets, exp_logger):
  train_dataloader = get_dataloader(args, datasets['train'])
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs
  optimizer, scheduler = setup_optimization(args, model, total_steps)
  exp_logger.update_optimization(optimizer, scheduler)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    model.train()
    for step, batch in enumerate(train_dataloader):
      inputs, targets = batch
      review_inputs(args, targets, datasets['train'].tokenizer)
      outputs = model(**inputs, labels=targets)
      # pdb.set_trace()
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

    eval_res = run_eval(args, model, datasets, exp_logger)
    if eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
      exp_logger.best_score = eval_res
      exp_logger.save_best_model(model, tokenizer, args.prune_keep)
    early_stop = exp_logger.end_epoch()
    if early_stop: break

  return model

def run_inference(args, model, dataloader, exp_logger, tokenizer, split):
  ''' goes through model generation without backprop, rather than classification '''
  all_outputs, all_targets = [], []
  exp_logger.eval_step = 0

  for inputs, target_dict in progress_bar(dataloader, total=len(dataloader)):
    all_targets.extend(target_dict)   # notice this is "extend", not "append"

    with no_grad():
      # defaults to greedy sampling, for param details see https://huggingface.co/docs/transformers/
      #        v4.15.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate 
      outputs = model.generate(**inputs, max_length=args.maximum_length, early_stopping=True)
      output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)
      all_outputs.extend(output_strings)
    # print(target_dict)
    # print(output_strings)
    # pdb.set_trace()

    if split == 'dev':
      exp_logger.eval_loss = 0  # no loss, since inference only
      exp_logger.eval_step += 1
      if args.debug and exp_logger.eval_step >= debug_break: break

  return all_outputs, all_targets

def run_eval(args, model, datasets, exp_logger, split='dev'):
  dataloader = get_dataloader(args, datasets[split], split)
  tokenizer = datasets[split].tokenizer

  if split == 'test' and args.task in ['meta_learn', 'fine_tune']:
    model = load_best_model(args, exp_logger, tokenizer)
  model.eval()

  outputs = run_inference(args, model, dataloader, exp_logger, tokenizer, split)
  if args.quantify or split == 'dev':
    results = eval_quantify(args, *outputs, exp_logger, tokenizer)
  elif args.qualify:
    results = eval_qualify(args, *outputs, exp_logger)
  # pdb.set_trace()
  with open(os.path.join(args.output_dir, 'output.json'), 'w') as tf:
    json.dump(results, tf, indent=2)
  return results


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

  if args.do_train:
    model = load_model(args, ontology, tokenizer, save_path)
    if args.task == 'meta_learn':
      supports = load_support(args, datasets)
      datasets.add_support(supports, args.left_out)
    run_train(args, model, datasets, exp_logger)
  elif args.do_eval:
    model = load_model(args, ontology, tokenizer, save_path) if args.task == 'in_context' else {}
    run_eval(args, model, datasets, exp_logger, split='test')
    run_eval(args, {}, datasets, exp_logger, split='test')
