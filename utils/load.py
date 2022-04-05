import os, pdb, sys
import json
import re
import random
import glob
import csv 
import pickle as pkl
import numpy as np
import pandas as pd

from torch import load as torch_load
from torch import float16 as torch_float16
from tqdm import tqdm as progress_bar
from transformers import GPT2LMHeadModel,GPT2ForSequenceClassification, GPT2Config, GPT2Tokenizer, \
                          BartForConditionalGeneration, BartConfig, BartTokenizer, \
                          T5ForConditionalGeneration, T5Config, T5Tokenizer
from transformers import logging, GPTJForCausalLM, AutoTokenizer
from assets.static_vars import device, DATASETS, CHECKPOINTS
from components.embed import Embedder
from components.trade import TradeModel
from utils.trade_utils import prepare_data_seq

logging.set_verbosity_error()

def load_data(args):
  data = {}
  for split in ['train', 'dev', 'test', 'ontology']:
    split_path = os.path.join(args.input_dir, args.dataset, f"{split}.json")
    split_data = json.load(open(split_path, 'r'))
    if split == 'ontology':
      data[split] = split_data
      example_type = 'domains'
    else:
      data[split] = split_data
      example_type = 'conversations'
    if args.verbose:
      print(f"Loaded {split} data with {len(data[split])} {example_type}")
  return data

def load_support(args):
  support_data = {}
  if args.num_shots == 'full' or args.task != 'meta_learn':
    return support_data

  for dataset, full_name in DATASETS.items():
    if dataset != args.left_out:
      support_path = os.path.join(args.input_dir, dataset, "train.json")
      sdata = json.load(open(support_path, 'r'))  # consider loading dev data too
      if args.debug:
        support_data[dataset]['data'] = sdata[:500]
      else:
        support_data[dataset]['data'] = sdata

      sont = json.load(open(os.path.join(args.input_dir, dataset, "ontology.json"), 'r'))
      support_data[dataset]['ont'] = sont
  return support_data

def load_tokenizer(args):
  special = { 'additional_special_tokens': ['<customer>', '<agent>', '<label>']  }
  token_ckpt = CHECKPOINTS[args.model][args.size]

  if args.model == 't5':
    tokenizer = T5Tokenizer.from_pretrained(token_ckpt)
    special['sep_token'] = '<sep>'
  elif args.model == 'gpt':
    tokenizer = AutoTokenizer.from_pretrained(token_ckpt)
    special['sep_token'] = '<sep>'
    special['pad_token'] = '<pad>'
  elif args.model == 'bart':
    tokenizer = BartTokenizer.from_pretrained(token_ckpt)
  elif args.model == 'trade':
    tokenizer = prepare_data_seq(args, tokenizer=True)
    return tokenizer
  else:
    print(f'{args.model} not supported at this time')
    sys.exit()

  if args.do_train or args.num_shots == 'percent':
    print(f"Adding special tokens {special}")
    tokenizer.add_special_tokens(special)
  tokenizer.padding_side = 'left'
  return tokenizer

def load_model(args, ontology, tokenizer, load_dir):
  print(f"Setting up {args.size} {args.model} model for {args.num_shots} shot learning")
  if args.num_shots == 'percent':
    return load_best_model(args, ontology, load_dir)
  
  ckpt_name = CHECKPOINTS[args.model][args.size]
  if args.model == 'gpt':
    if args.size == 'large':
      model = GPTJForCausalLM.from_pretrained(ckpt_name,
               revision="float16", torch_dtype=torch_float16, low_cpu_mem_usage=True)
    else:
      model = GPT2LMHeadModel.from_pretrained(ckpt_name)
    # use GPTJForCausalLM: https://huggingface.co/docs/transformers/model_doc/gptj
  elif args.model == 'bart':
    model = BartForConditionalGeneration.from_pretrained(ckpt_name)
  elif args.model == 't5':
    model = T5ForConditionalGeneration.from_pretrained(ckpt_name)
  elif args.model == 'trade':
    model = TradeModel(args, tokenizer, ontology)
    ckpt_path = os.path.join(args.input_dir, 'cache', f"{ckpt_name}.pt")
    if os.path.exists(ckpt_path):
      model.load_state_dict(torch_load(ckpt_path))
    return model.to(device)

  if args.do_train or args.num_shots == 'percent': 
    model.config.pad_token = tokenizer.pad_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))  # transformer_check
  
  return model.to(device)

def load_glove(size=300):
  if size > 0:
    root_path = "/persist"
    path_name = ".embeddings/glove/"
    file_name = f"glove.6B.{size}d.txt"
    full_path = os.path.join(root_path, path_name, file_name)
    print(f'Loading embeddings from {full_path} ...')
    return Embedder(full_path, size)
  else:
    return None  # embedder is not needed for this task

def load_best_model(args, ontology, load_dir):
  print(f'Loading best finetuned model from {load_dir} ...')
  if len(args.checkpoint) > 0:
    top_filename = args.checkpoint
  else:
    folders = glob.glob(os.path.join(load_dir, "*pt"))
    top_acc, top_folder = 0, ''

    for fname in folders:
      re_str = r'acc([0-9]{3})\.pt$'
      current_score = re.findall(re_str, fname)
      score = int(current_score[0]) if len(current_score) > 0 else 0
      if args.do_eval:
        fname = fname.split('/')[-1]  # convert path to single folder
        parts = fname.split('_')
        model_type = parts[0]
        model_match = args.model == model_type
      else:
        model_match = True

      if score > top_acc and model_match:
        top_acc = score
        top_folder = fname

  if len(top_folder) == 0:
    raise RuntimeError(f'No models were found in {load_dir}')
  else: 
    ckpt_path = os.path.join(load_dir, top_folder)
    print(f'Attempting to load {ckpt_path} as best model')
  
  # checkpoint = torch_load(ckpt_path, map_location='cpu')
  # model.load_state_dict(checkpoint)
  model = load_model(args, ontology, {}, ckpt_path)
  model.eval()
  return model
