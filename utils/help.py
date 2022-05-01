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
  save_path = os.path.join(dataset_path, args.task, f'{args.model}_{args.size}')
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
  assert(args.context_length != 0)
  return args, save_path

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

def model_match(fname, args):
  """
  check if the ckpt with path 'fname' fits the current args

  follow the format:
  f'results/{dataset}/{task}/{model}_{size}/{prompt_style}_lr{}_clen{context_length}_epoch{}_acc{}.pt'
  """
  model_type, model_size = fname.split('/')[-2].split("_")
  prompt_style, lr, clen, _, _ = fname.split('/')[-1].split("_")
  if model_type == args.model and \
     model_size == args.size  and \
     prompt_style == args.prompt_style and \
     lr == f'lr{args.learning_rate}' and \
     clen == f'clen{args.context_length}':
      return True
  return False


def reformat_data(args):
  if not os.path.exists(os.path.join(args.input_dir, args.dataset)) and args.ignore_cache:
    os.makedirs(os.path.join(args.input_dir, args.dataset), exist_ok=True)
    if args.dataset == 'abcd':  # MultiWoz 2.0
      reformatter = ReformatABCD(args.input_dir)
    elif args.dataset == 'mwoz' or args.dataset == 'mwoz22':  # MultiWoz 2.2
      reformatter = ReformatMultiWOZ22(args.input_dir)
      shutil.copyfile(os.path.join(args.input_dir, "multiwoz_dst/MULTIWOZ2.2/otgy.json"), 
                      os.path.join(args.input_dir, args.dataset, "ontology.json"))
    elif args.dataset == 'sgd':   # Schema Guided Dialogue
      reformatter = ReformatSGD()
      shutil.copyfile(os.path.join(args.input_dir, "google_sgd/ontology.json"),
                      os.path.join(args.input_dir, args.dataset, "ontology.json"))
    else:
      reformatter = ReformatBase()
    # loads, reformats and saves the data in the background
    reformatter.reformat()

def determine_dataset(global_id):
  dialog_id, turn_count = global_id.split('_')
  if dialog_id.endswith('json'):
    return 'mwoz'
  elif dialog_id.endswith('_00000'):
    return 'sgd'
  elif dialog_id.startswith('voip'):
    return 'dstc'
  elif dialog_id.startswith('movies_') or dialog_id.startswith('restaurant_'):
    return 'gsim'
  elif dialog_id.startswith('dlg-'):
    return 'tt'
  elif re.match("^\d{4}", dialog_id):  # starts with four digits
    return 'abcd'
  else:
    raise KeyError(f"{global_id} could not be identified")