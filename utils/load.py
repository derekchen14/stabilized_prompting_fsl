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
                    GPTJForCausalLM, BartForConditionalGeneration, BartConfig, BartTokenizer
from transformers import logging, pipeline
from assets.static_vars import device
from components.embed import Embedder

logging.set_verbosity_error()

def load_data(args):  
  data = {}
  for split in ['train', 'dev', 'test', 'ontology']:
    split_path = os.path.join(args.input_dir, args.dataset, f"{split}.json")
    split_data = json.load(open(split_path, 'r'))

    if split == 'ontology':
      if args.task == 'gsim':
        data[split] = split_data['user_acts']
      if args.task == 'abcd':
        data[split] = split_data['subflows']
      if args.task == 'mwoz':
        data[split] = split_data
      example_type = 'labels'
    else:
      data[split] = split_data
      example_type = 'conversations'
    if args.verbose:
      print(f"Loaded {split} data with {len(data[split])} {example_type}")

  return data

def load_tokenizer(args):
  special = { 'additional_special_tokens': 
          ['<customer>', '<agent>', '<sep>', '<service>']  }
  tkn_name = CHECKPOINTS[args.model]

  if args.model == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained(tkn_name)
  elif args.model == 'gpt':
    tokenizer = GPT2Tokenizer.from_pretrained(tkn_name)
    special['sep_token'] = '<sep>'
    special['pad_token'] = '<pad>'
  elif args.model == 'bart':
    tokenizer = BartTokenizer.from_pretrained(tkn_name)
  else:
    print(f'{args.model} not supported at this time')
    sys.exit()

  tokenizer.add_special_tokens(special)
  return tokenizer

def load_model(args, ontology, tokenizer, ckpt_path=None):
  print(f"Setting up {args.model} model for {TASKS[args.task]} task")

  if args.size == 'small':
    ckpt_name = 'gpt2'
  elif args.size == 'medium':
    ckpt_name = 'gpt2-medium'
  elif args.size == 'large':
    ckpt_name = 'EleutherAI/gpt-j-6B'
    # use GPTJForCausalLM: https://huggingface.co/docs/transformers/model_doc/gptj


  if args.model == 'gpt':
    if args.task in ['classify', 'track']:
      model = GPT2ForSequenceClassification.from_pretrained(ckpt_name)
    elif args.task == 'generate':
      model = GPT2LMHeadModel.from_pretrained(ckpt_name)
  elif args.model == 'bart':
    if args.task == 'classify':
      model = BartForSequenceClassification.from_pretrained(ckpt_name)
    elif args.task in ['generate', 'track']:
      model = BartForConditionalGeneration.from_pretrained(ckpt_name)

  if ckpt_path is None:
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
  print('Loading best finetuned model ...')
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
