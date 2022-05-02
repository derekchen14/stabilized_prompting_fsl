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
from utils.evaluate import eval_quantify, eval_qualify
from assets.static_vars import device, debug_break, STOP_TOKENS
from torch.utils.data import DataLoader, SequentialSampler

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
      if step > 20000:
        break

    _, eval_res = run_eval(args, model, datasets, exp_logger, detective)
    if eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
      exp_logger.best_score = eval_res
      exp_logger.save_best_model(model, tokenizer, args.prune_keep)
    early_stop = exp_logger.end_epoch()
    if early_stop: break

  return model

def run_inference(args, model, dataset, exp_logger, tokenizer, split):
  ''' goes through model generation without backprop, rather than classification '''
  dataloader = get_dataloader(args, dataset, split)
  all_outputs, all_targets = [], []
  exp_logger.eval_step = 0

  for batch in progress_bar(dataloader, total=len(dataloader)):
    inputs, target_dict = dataset.collate(args, batch)
    all_targets.extend(target_dict)   # notice this is "extend", not "append"

    if args.task == 'in_context':
      maxl = 2048 if args.size == 'large' else 1024
    else:
      maxl = inputs['input_ids'].shape[1] + 12

    with no_grad():
      # defaults to greedy sampling, for param details see https://huggingface.co/docs/transformers/
      #        v4.15.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate 
      outputs = model.generate(**inputs, max_length=maxl, early_stopping=True)
    output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)
    all_outputs.extend(output_strings)

    if split == 'dev':
      exp_logger.eval_loss = 0  # no loss, since inference only
      exp_logger.eval_step += 1
      if exp_logger.eval_step >= 5000:
        break
      # if args.debug and exp_logger.eval_step >= debug_break: break
    if split == 'test' and args.debug:
      exp_logger.eval_step += 1
      if exp_logger.eval_step >= (debug_break * 200): break
  return all_outputs, all_targets

def leftout_inference(args, model, dataset, tokenizer):
  leftout_data = DataLoader(dataset.leftout, batch_size=8, collate_fn=lambda x:x,
                              sampler=SequentialSampler(dataset.leftout) )
  all_outputs, all_targets = [], []

  temp_steps = 0
  for batch in progress_bar(leftout_data, total=len(leftout_data), desc='Leftout'):
    inputs, target_dict = dataset.collate(args, batch)
    maxl = inputs['input_ids'].shape[1] + 12
    with no_grad():
      outputs = model.generate(**inputs, max_length=maxl, early_stopping=True)
    output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)

    all_targets.extend(target_dict)
    all_outputs.extend(output_strings)

    temp_steps += 1
    if temp_steps >= 1000: break
  return all_outputs, all_targets

def run_eval(args, model, datasets, exp_logger, detective, split='dev'):
  dataset = datasets[split]
  tokenizer = dataset.tokenizer
  dataset.add_detective(detective)
  if split == 'test' and args.task in ['meta_learn', 'fine_tune']:
    model = load_best_model(args, exp_logger, tokenizer)
  model.eval()

  if args.task == 'meta_learn' and args.verbose:
    leftout_exp = datasets['dev'].leftout
    left_outputs = leftout_inference(args, model, dataset, tokenizer)
    eval_quantify(args, *left_outputs, exp_logger, tokenizer, args.left_out)

  outputs = run_inference(args, model, dataset, exp_logger, tokenizer, split)
  if args.quantify or split == 'dev':
    results = eval_quantify(args, *outputs, exp_logger, tokenizer)
  elif args.qualify:
    results = eval_qualify(args, *outputs, exp_logger)
  else:
    results = None
  return outputs, results

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
    model = load_model(args, ontology, tokenizer, save_path) if args.task == 'in_context' else {}
    outputs, _ = run_eval(args, model, datasets, exp_logger, detective, split='test')

