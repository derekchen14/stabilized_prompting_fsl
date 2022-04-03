import os, pdb, sys
import numpy as np
import pickle as pkl
import torch
import random
import json
import re
import shutil

from collections import defaultdict
from tqdm import tqdm as progress_bar
from assets.static_vars import device
from copy import deepcopy
from transformers import get_scheduler
from utils.reformat import *

def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

def setup_gpus(args):
  n_gpu = 0  # set the default to 0
  if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
  args.n_gpu = n_gpu
  if n_gpu > 0:   # this is not an 'else' statement and cannot be combined
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
  return args

def check_directories(args):
  dataset_path = os.path.join(args.output_dir, args.dataset)
  save_path = os.path.join(dataset_path, args.task)
  if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    print(f"Created {dataset_path} for {args.dataset} results")
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Created {save_path} directory")
  
  cache_path = os.path.join(args.input_dir, 'cache', args.dataset)
  if not os.path.exists(cache_path):
    os.makedirs(cache_path)
    print(f"Created {cache_path} directory")

  if args.debug:
    args.log_interval /= 10
  if args.num_shots in ['zero', 'few', 'percent']:
    assert(len(args.left_out) > 0)
    if args.style == 'dataset':
      assert(args.dataset == args.left_out)
  return args, save_path

def trade_loss(predictions, targets):
  pass

def prepare_inputs(batch):
  # change as needed
  pass
  # return inputs

def setup_optimization(args, model, total_steps):
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
      {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
      },
      {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0},
  ]
  warmup = int(total_steps * 0.2)

  optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
  scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)
  return optimizer, scheduler

def review_inputs(args, targets, tokenizer):
  if args.debug and args.verbose:
    tbd = tokenizer.batch_decode(targets)
    print(f"Batch with {len(tbd)} items")
    for batch_item in tbd:
      print(batch_item.replace('<pad>', ''))
      pdb.set_trace()

def get_all_checkpoints(args, load_dir):
  print('Loading all finetuned models ...')
  filenames = [f for f in os.listdir(load_dir) if f.endswith('.pt')]
  if len(filenames) == 0:
    raise RuntimeError(f'No models were found in {load_dir}')

  checkpoints = []
  for fname in filenames:
    ckpt_path = os.path.join(load_dir, fname)
    print(f'Found {ckpt_path} in directory')
    checkpoints.append(ckpt_path)
  return checkpoints

def memstat(message):
  malloc = torch.cuda.memory_allocated()
  human_malloc = str(round( (malloc / 1000000), 2)) + "MB"
  maxmem = torch.cuda.max_memory_allocated()
  human_maxmem = str(round( (maxmem / 1000000), 2)) + "MB"
  print(f"{message} -- Current memory: {human_malloc}, Max: {human_maxmem}")


def reformat_data(args):
  """
  TODO:
  add more dataset (GSIM, ABCD, TaskMaster)
  """

  if not os.path.exists(os.path.join(args.input_dir, args.dataset)) or args.ignore_cache:
    os.makedirs(os.path.join(args.input_dir, args.dataset), exist_ok=True)
    if args.dataset == 'mwoz20':  # MultiWoz 2.0
      reformatter = ReformatMultiWOZ20(args.input_dir)
    elif args.dataset == 'mwoz21':  # MultiWoz 2.1
      from utils.trade_proc import trade_process
      trade_process(args)
      reformatter = ReformatMultiWOZ21(args.input_dir)
      shutil.copyfile(os.path.join(args.input_dir, "multiwoz_dst/MULTIWOZ2.1/ontology.json"), 
                      os.path.join(args.input_dir, args.dataset, "ontology.json"))
    elif args.dataset == 'mwoz22':  # MultiWoz 2.2
      reformatter = ReformatMultiWOZ22(args.input_dir)
      shutil.copyfile(os.path.join(args.input_dir, "multiwoz_dst/MULTIWOZ2.2/otgy.json"), 
                      os.path.join(args.input_dir, args.dataset, "ontology.json"))
    elif args.dataset == 'sgd':   # Schema Guided Dialogue
      reformatter = ReformatSGD()
    # loads, reformats and saves the data in the background
    reformatter.reformat()
