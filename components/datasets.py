import os, pdb, sys
import numpy as np
import random
import mmap
import torch
from torch.utils.data import Dataset
from assets.static_vars import device

class BaseInstance:
  def __init__(self, embed_data, utt_text, label_text, label_id):
    self.input_id = embed_data['input_ids']
    self.input_mask = embed_data['attention_mask']
    if 'token_type_ids' in embed_data:
      self.segment_id = embed_data['token_type_ids']
    else:  # since roberta tokenizer does not return segment ids
      self.segment_id = np.zeros(len(self.input_mask))

    self.utterance = utt_text
    self.label = label_text
    self.label_id = label_id

  def __repr__(self):
    return f'utt: {self.utterance}\nlabel: {self.label}'

class BaseDataset(Dataset):
  def __init__(self, args, examples, tokenizer, split):
    self.split = split
    self.shuffle = (split == 'train')
    self.data = examples
    self.size = len(self.data)

    self.tokenizer = tokenizer
    self.task = args.task

  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    return self.data[idx]

  def _pad_right(self, targets):
    max_vec_len = max([len(vector) for vector in targets.input_ids])
    assert(max_vec_len < 24)

    padded = []
    for vector in targets.input_ids:
      diff = max_vec_len - len(vector)
      for i in range(diff):
        vector.append(-100)  # id of -100 means to not pay attention on training
      padded.append(vector)

    target_tensor = torch.tensor(padded).to(device)
    return target_tensor

  def collate_func(self, examples):
    """transforms a batch of examples into a features dict that can be fed directly into a model"""
    pad_style = 'max_length' if self.split == 'test' else 'longest' # sequence in the batch

    dialogues, labels, cands = [], [], []
    for example in examples:
      dialogues.append(example['dialogue'])
      labels.append(example['label'])
    inputs = self.tokenizer(dialogues, padding=pad_style,
                              truncation=True, return_tensors='pt').to(device)

    if self.task == 'generate':
      targets = inputs["input_ids"]
    else:
      targets = torch.tensor(labels, dtype=torch.long, device=device) # or torch.float of BCEWithLogits
    return inputs, targets

class MetaLearnDataset(BaseDataset):

  def collate_func(self, examples):
    """transforms a batch of examples into a features dict that can be fed into a GPT model"""
    dialogues, extras = [], []
    eos = self.tokenizer.eos_token

    if self.split == 'train':
      for example in examples:
        context = example['context']
        value = example['label']
        dialog = context + '<sep>' + example['prompt'] + '<label>' + value + eos
        dialogues.append(dialog)
      inputs = self.tokenizer(dialogues, padding=True, max_length=1024,
                                truncation=True, return_tensors='pt').to(device)
      targets = inputs['input_ids']
      return inputs, targets

    elif self.split in ['dev', 'test']:
      for example in examples:
        context = example['context']
        dialog = context + '<sep>' + example['prompt'] + '<label>'
        dialogues.append(dialog)
        extras.append(example['extra'])
      inputs = self.tokenizer(dialogues, padding=True, max_length=1000,
                                truncation=True, return_tensors='pt').to(device)
      return inputs, extras

class InContextDataset(BaseDataset):

  def collate_func(self, examples):
    """transforms a batch of examples into a features dict that can be fed directly into a model"""
    pad_style = 'max_length' if self.split == 'test' else 'longest' # sequence in the batch

    contexts, prompts, labels = [], [], []
    for example in examples:
      contexts.append(example['context'])
      prompts.append(example['prompt'])
      labels.append(example['label'])

    inputs = self.tokenizer(contexts, prompts, padding=pad_style,
                              truncation='only_first', return_tensors='pt').to(device)
    targets = torch.tensor(labels, dtype=torch.long, device=device) # or torch.float of BCEWithLogits
    return inputs, targets

class FineTuneDataset(BaseDataset):

  def collate_func(self, examples):
    """transforms a batch of examples into a features dict that can be fed into a T5 or BART model"""
    dialogues, labels = [], []

    for example in examples:
      dialog = example['context'] + '<sep>' + example['prompt']
      dialogues.append(dialog)
      labels.append(example['label'] if self.split == 'train' else example['extra'])

    # self.tokenizer.pad_token = self.tokenizer.eos_token
    inputs = self.tokenizer(dialogues, padding='longest', max_length=1000,
                                truncation=True, return_tensors='pt').to(device)
    if self.split == 'train':
      targets = self.tokenizer(labels) # we do not want tensors
      target_tensor = self._pad_right(targets)
      return inputs, target_tensor
    else:
      return inputs, labels
