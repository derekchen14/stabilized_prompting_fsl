import os, sys, pdb
import numpy as np
import random
import torch

from tqdm import tqdm as progress_bar
from components.logger import ExperienceLogger

from utils.help import *
from utils.process import process_data, get_dataloader
from utils.arguments import solicit_params
from utils.load import *
from utils.evaluate import eval_quantify, eval_qualify
from assets.static_vars import device, debug_break

def run_train(args, model, datasets, tokenizer, exp_logger):
  train_dataloader = get_dataloader(args, datasets['train'], tokenizer)
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs
  optimizer, scheduler = setup_optimization(args, model, total_steps)
  exp_logger.update_optimization(optimizer, scheduler)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    model.train()
      
    for step, batch in enumerate(train_dataloader):
      inputs, targets = batch
      outputs = model(**inputs, labels=targets)
      exp_logger.tr_loss += outputs.loss.item()
      loss = outputs.loss / args.grad_accum_steps
      loss.backward()

      if (step + 1) % args.grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        exp_logger.optimizer.step()  # backprop to update the weights
        exp_logger.scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        exp_logger.log_train(step)
      if args.debug and step >= debug_break*args.log_interval:
        break

    eval_res = run_eval(args, model, datasets, exp_logger)
    if args.do_save and eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
      exp_logger.best_score = eval_res
      exp_logger.save_best_model(model, tokenizer, args.prune_keep)
    early_stop = exp_logger.end_epoch()
    if early_stop: break

  return model

def run_inference(args, model, dataloader, exp_logger, split, extract_text=False):
  # runs a single epoch without gradients, optionally collects meta-data along the way
  if args.task == 'track':
    return dst_inference(args, model, dataloader, exp_logger, split)
  elif args.task == 'classify':
    return intent_inference(args, model, dataloader, exp_logger, split, extract_text)
  elif args.task == 'generate':
    return gen_inference(args, model, dataloader, exp_logger, split)

def intent_inference(args, model, dataloader, exp_logger, split, extract_text):
  all_preds, all_targets, all_contexts = [], [], []
  exp_logger.eval_step = 0

  for inputs, labels in progress_bar(dataloader, total=len(dataloader)):
    with torch.no_grad():
      outputs = model(**inputs, labels=labels)
    batch_loss = outputs.loss

    pred = outputs.logits.softmax(dim=-1) # detach().cpu().flatten().numpy().tolist()
    all_preds.append(pred)
    all_targets.append(labels)
    all_contexts.append(inputs['input_ids'])

    if split == 'dev':
      exp_logger.eval_loss = batch_loss.mean().item()
      exp_logger.eval_step += 1
      if args.debug and exp_logger.eval_step >= debug_break: break

  if args.task == 'rg':
    return all_preds, all_targets
  predictions = torch.cat(all_preds, axis=0).detach().cpu()
  targets = torch.cat(all_targets).detach().cpu()
  
  if extract_text:
    contexts = torch.cat(all_contexts).detach().cpu()
    return predictions, targets, contexts
  else:
    return predictions, targets

def gen_inference(args, model, dataloader, exp_logger, split):
  all_inputs, all_outputs, all_labels = [], [], []
  exp_logger.eval_step = 0

  for inputs, input_ids, label_dicts in progress_bar(dataloader, total=len(dataloader)):

    with torch.no_grad():
      outputs = model.generate(**inputs)
    batch_loss = outputs.loss

    logits = outputs.logits.softmax(dim=-1)       # batch_size, seq_len, vocab_size
    output_ids = torch.argmax(logits, axis=-1)    # batch_size, seq_len

    for single_input, single_output in zip(input_ids, output_ids):
      input_string = tokenizer.decode(single_input, skip_special_tokens=True)
      output_string = tokenizer.decode(single_output, skip_special_tokens=False)
      all_inputs.append(input_string)
      all_outputs.append(output_string)
    all_labels.extend(label_dicts)   # notice this is "extend", not "append"

    if split == 'dev':
      exp_logger.eval_loss = batch_loss.mean().item()
      exp_logger.eval_step += 1
      if args.debug and exp_logger.eval_step >= debug_break: break

  assert(len(all_labels) == len(all_inputs))
  assert(len(all_labels) == len(all_outputs))
  pairing = [all_inputs, all_outputs]
  return pairing, all_labels

def run_eval(args, model, datasets, exp_logger, split='dev'):
  dataloader = get_dataloader(args, datasets[split], split)
  tokenizer = datasets[split].tokenizer

  if split == 'test':        
    if args.qualify:
      outputs = run_inference(args, model, dataloader, exp_logger, split, True)
      results = eval_qualify(args, *outputs, exp_logger, tokenizer)
    elif args.quantify:
      outputs = run_inference(args, model, dataloader, exp_logger, split)
      results = eval_quantify(args, *outputs, exp_logger, tokenizer, split)
  else:
    model.eval()
    outputs = run_inference(args, model, dataloader, exp_logger, split)
    results = eval_quantify(args, *outputs, exp_logger, tokenizer, split)

  return results


if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  args, save_path = check_directories(args)
  set_seed(args)

  raw_data = load_data(args)
  tokenizer = load_tokenizer(args)
  dataset, ontology = process_data(args, raw_data, tokenizer)
  exp_logger = ExperienceLogger(args, ontology, save_path)

  if args.do_train:
    model = load_model(args, ontology, tokenizer)
    run_train(args, model, dataset, tokenizer, exp_logger)
  if args.do_eval:
    model = load_best_model(args, ontology, save_path)
    run_eval(args, model, dataset, exp_logger, split='test')
