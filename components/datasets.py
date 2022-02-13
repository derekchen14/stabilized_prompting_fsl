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
  def __init__(self, examples, tokenizer, task, split):
    self.split = split
    self.shuffle = (split == 'train')
    self.data = examples
    self.size = len(self.data)

    self.tokenizer = tokenizer
    self.task = task

  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    return self.data[idx]

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
    """transforms a batch of examples into a features dict that can be fed directly into a model"""
    dialogues, labels = [], []
    eos = self.tokenizer.eos_token

    if self.split == 'train':
      
      for example in examples:
        input_text = example['dialogue'] + example['prompt'] + example['flattened'] + eos
        dialogues.append(input_text)
      inputs = self.tokenizer(dialogues, padding='longest',
                                truncation=True, return_tensors='pt').to(device)
      targets = inputs['input_ids']
      return inputs, targets

    elif self.split in ['dev', 'test']:

      for example in examples:
        input_text = example['dialogue']
        dialogues.append(input_text)
        labels.append(example['structured'])
      inputs = self.tokenizer(dialogues, padding='max_length',
                                truncation=True, return_tensors='pt').to(device)
      targets = inputs['input_ids']
      return inputs, targets, labels

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
    """transforms a batch of examples into a features dict that can be fed directly into a model"""
    pad_style = 'max_length' if self.split == 'test' else 'longest' # sequence in the batch

    dialogues, prompts, labels = [], [], []
    for example in examples:
      dialogues.append(example['dialogue'])
      prompts.append(example['prompt'])
      labels.append(example['label'])

    inputs = self.tokenizer(dialogues, prompts, padding=pad_style,
                              truncation='only_first', return_tensors='pt').to(device)
    targets = torch.tensor(labels, dtype=torch.long, device=device) # or torch.float of BCEWithLogits
    return inputs, targets