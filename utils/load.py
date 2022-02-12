import os, pdb, sys
import json
import re
import random
import glob
import csv 
import pickle as pkl
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm as progress_bar
from transformers import GPT2LMHeadModel,GPT2ForSequenceClassification, GPT2Config, GPT2Tokenizer, \
                          BartForConditionalGeneration, BartConfig, BartTokenizer \
                          T5ForConditionalGeneration, T5Config, T5Tokenizer
from transformers import logging, GPTJForCausalLM, AutoTokenizer
from assets.static_vars import device, CHECKPOINTS, TASKS
from components.embed import Embedder

logging.set_verbosity_error()

def load_data(args):  
  data = {}
  for split in ['train', 'dev', 'test', 'ontology']:
    split_path = os.path.join(args.input_dir, args.dataset, f"{split}.json")
    split_data = json.load(open(split_path, 'r'))

    if split == 'ontology':
      data[split] = split_data[args.style]
      example_type = 'labels'
    else:
      data[split] = split_data
      example_type = 'conversations'
    if args.verbose:
      print(f"Loaded {split} data with {len(data[split])} {example_type}")

  return data

def load_tokenizer(args):
  special = { 'additional_special_tokens': 
          ['<customer>', '<agent>', '<label>', '<kb>']  }
  token_ckpt = CHECKPOINTS[args.model][args.size]

  if args.model == 't5':
    tokenizer = T5Tokenizer.from_pretrained(token_ckpt)
  elif args.model == 'gpt':
    tokenizer = AutoTokenizer.from_pretrained(token_ckpt)
    special['sep_token'] = '<sep>'
    special['pad_token'] = '<pad>'
  elif args.model == 'bart':
    tokenizer = BartTokenizer.from_pretrained(token_ckpt)
  elif args.model == 'bart':
    tokenizer = BartTokenizer.from_pretrained(token_ckpt)
  else:
    print(f'{args.model} not supported at this time')
    sys.exit()

  if args.do_train or args.num_shots == 'percent':
    print(f"Adding special tokens {special}")
    tokenizer.add_special_tokens(special)
  tokenizer.padding_side = 'left'
  return tokenizer

def load_model(args, ontology, tokenizer, load_dir):
  print(f"Setting up {args.size} {args.model} model for {TASKS[args.task]} task")
  if args.num_shots == 'percent':
    return load_best_model(args, ontology, load_dir)
  
  ckpt_name = CHECKPOINTS[args.model][args.size]
  if args.model == 'gpt':
    if args.size == 'large':
      model = GPTJForCausalLM.from_pretrained(ckpt_name,
               revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
      model = GPT2LMHeadModel.from_pretrained(ckpt_name)
    # use GPTJForCausalLM: https://huggingface.co/docs/transformers/model_doc/gptj
  elif args.model == 'bart':
    model = BartForConditionalGeneration.from_pretrained(ckpt_name)
  elif args.model == 't5':
    model = BartForConditionalGeneration.from_pretrained(ckpt_name)

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
  
  # checkpoint = torch.load(ckpt_path, map_location='cpu')
  # model.load_state_dict(checkpoint)
  model = load_model(args, ontology, {}, ckpt_path)
  model.eval()
  return model
