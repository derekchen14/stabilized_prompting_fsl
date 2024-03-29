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
from assets.static_vars import device, SLOT_MAPPING
from utils.evaluate import parse_output
from copy import deepcopy
from transformers import get_scheduler
from utils.reformat import *
import torch_optimizer as ada_optim

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
  if args.train_percent > 0:
    save_path = os.path.join(dataset_path, args.task, f'{args.model}_{args.size}_{args.train_percent}')
  else:
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
  if args.percent < 0.1 or args.percent > 1.0:
    raise IndexError("Data percentage must be between 10% and 100% \
      If you want to run even faster, consider using debug mode instead.")
  return args, save_path

def setup_optimization(args, model, total_steps):
  no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
      {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
      },
      {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0},
  ]
  warmup = int(total_steps * 0.2)

  optimizer = ada_optim.Adafactor(optimizer_grouped_parameters, lr=args.learning_rate,
                                    scale_parameter=False, relative_step=False)
  scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)
  return optimizer, scheduler

def review_inputs(args, inputs, targets, tokenizer):
  if args.debug and args.verbose:
    if args.model == 'gpt':
      tbd = tokenizer.batch_decode(targets)
      print(f"Batch with {len(tbd)} items")
      for batch_item in tbd:
        print(batch_item.replace('<pad>', '|'))
    else:
      tbdi = tokenizer.batch_decode(inputs['input_ids'])
      targets[targets==-100] = 0
      tbdt = tokenizer.batch_decode(targets)
      print(f"Batch with {len(tbdi)} items")
      for batch_input, batch_target in zip(tbdi, tbdt):
        print(batch_input.replace('<pad> ', '|'), batch_target.replace('<pad>', ''))
    pdb.set_trace()

def batchify(args, turn, global_id, prior_pred_state):
  """ returns a list of batches where the ground_truth prev_state has been 
  replaced with the predicted prior_state from the previous turn """
  batches = []

  convo_id, turn_str = global_id.split('_')
  turn_count = int(turn_str)
  if turn_count == 1:
    prev_state = {}
  else:
    prev_gid = f"{convo_id}_{turn_count - 1}"
    prev_state = prior_pred_state[prev_gid]

  batch = []
  for example in turn:
    example['prev_state'] = prev_state
    batch.append(example)

    if len(batch) == args.batch_size:
      batches.append(batch)
      batch = []
  
  if len(batch) > 0:
    batches.append(batch)
  return batches

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
  f'results/{dataset}/{task}/{model}_{size}/{prompt_style}_lr{}_{saliency}_epoch{}_acc{}.pt'
  """
  model_type, model_size = fname.split('/')[-2].split("_")[0], fname.split('/')[-2].split("_")[1]
  if len(fname.split('/')[-1].split("_")) != 5:
    return False
  prompt_style, lr, saliency, epoch, _ = fname.split('/')[-1].split("_")

  type_match = model_type == args.model
  size_match = model_size == args.size
  train_data_match = True
  if args.train_percent > 0:
    if len(fname.split('/')[-2].split("_")) < 3:
      train_data_match = False
    else:
      train_data_match = args.train_percent == float(fname.split('/')[-2].split("_")[2])
  prompt_match = prompt_style == args.prompt_style
  lr_match = lr == f'lr{args.learning_rate}'

  if type_match and size_match and prompt_match and lr_match:
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

def standardize_format(raw_domain, raw_slot):
  domain = raw_domain.lower()       # services_2
  domain = domain.split('_')[0]     # services
  if domain.endswith('ses'):        # service (movies)
    domain = domain[:-2]
  if domain.endswith('s') and domain != 'bus':          # service
    domain = domain[:-1]

  slot = raw_slot.lower()
  slot = slot.replace('_', ' ').replace('.', ' ')
  if slot in SLOT_MAPPING:
    slot = SLOT_MAPPING[slot]
  return domain, slot

def desemble(args, output_strings):
  pred_values = [parse_output(args, output_str) for output_str in output_strings]
  if args.ensemble <= 1:
    return pred_values

  output_desemble = []
  group_num = len(pred_values) // args.ensemble
  group_size = args.ensemble
  for group_id in range(group_num):
    output_dict = defaultdict(int)
    for pred_value in pred_values[group_size*group_id:group_size*(group_id+1)]:
      output_dict[pred_value] += 1
    output_best = max(output_dict, key=output_dict.get)
    output_desemble.append(output_best)
  return output_desemble



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
