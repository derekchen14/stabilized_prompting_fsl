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

def gen_inference(args, model, dataloader, exp_logger, split):
  all_inputs, all_outputs, all_labels = [], [], []
  exp_logger.eval_step = 0

  for inputs, input_ids, label_dicts in progress_bar(dataloader, total=len(dataloader)):
    input_strings = tokenizer.batch_decode(input_ids.detach(), skip_special_tokens=True)
    all_inputs.extend(input_strings)
    all_labels.extend(label_dicts)   # notice this is "extend", not "append"
    
    with torch.no_grad():
      # defaults to greedy sampling, for param details see https://huggingface.co/docs/transformers/
      #        v4.15.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate 
      output_ids = model.generate(**inputs, max_length=512, min_length=60, early_stopping=True)
      output_strings = tokenizer.batch_decode(output_ids.detach(), skip_special_tokens=False)
      all_outputs.extend(output_strings)

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

def run_interaction(args, model, dataset, exp_logger):
  dataset = datasets['dev']
  for _ in range(args.batch_size):
    sample_id = random.randrange(dataset.size)
    sample = dataset[sample_id]
    for turn in sample['dialogue']:
      print(turn)
      pdb.set_trace()

    prompt = input("<customer> ")
    input_text = sample['dialogue'] + " <customer> " + prompt
    print("-- <start debug> --")
    print(input_text)
    print("-- <end debug> --")
    inputs = dataset.tokenizer(input_text, return_tensors='pt').to(device)

    # https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate 
    with torch.no_grad():
      if args.kappa > 1:
        output_embeds = model.generate(**inputs, max_length=512, early_stopping=True, do_sample=True, 
                                          num_beams=args.kappa, temperature=args.temperature, top_p=0.95)
        output_texts = tokenizer.batch_decode(output_embeds.detach(), skip_special_tokens=False)
        for i, out_text in enumerate(output_texts):
          print(f"<agent {i+1}> {out_text}")
      else: 
        output_embed = model.generate(**inputs, max_length=512, early_stopping=True)
        output_text = tokenizer.decode(output_embed[0].detach(), skip_special_tokens=False)
        print(f"<agent> {output_text}")


if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  args, save_path = check_directories(args)
  set_seed(args)

  raw_data = load_data(args)
  tokenizer = load_tokenizer(args)
  datasets, ontology = process_data(args, raw_data, tokenizer)
  exp_logger = ExperienceLogger(args, ontology, save_path)

  model = load_model(args, ontology, tokenizer, save_path)
  if args.do_train:
    run_train(args, model, datasets, tokenizer, exp_logger)
  elif args.do_interact:
    run_interaction(args, model, datasets, tokenizer, exp_logger)
  elif args.do_eval:
    run_eval(args, model, datasets, exp_logger, split='test')
