import os, pdb, sys
import numpy as np
import random
import mmap
import torch
from torch.utils.data import Dataset

from assets.static_vars import device
from utils.prompt import find_prompt
from utils.meta_learn import select_context

class BaseDataset(Dataset):
  def __init__(self, args, examples, tokenizer, split):
    self.split = split
    self.shuffle = (split == 'train')
    self.data = examples
    self.size = len(self.data)

    self.tokenizer = tokenizer
    self.task = args.task
    self.max_length = args.max_len
    self.model_type = args.model
    self.prompt_style = args.prompt_style

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

  def collate_lm(self, examples):
    raise NotImplementedError

  def collate_seq2seq(self, examples):
    raise NotImplementedError

  def collate_func(self, examples):
    if self.model_type == 'gpt':
      return self.collate_lm(examples)
    elif self.model_type in ['bart', 't5']:
      return self.collate_seq2seq(examples)

class InContextDataset(BaseDataset):

  def __init__(self, args, examples, support_set, tokenizer, split):
    super().__init__(args, examples, tokenizer, split)
    self.support = support_set

  def select_context(self, dialog, target):
    current_size = len(dialog)
    while current_size < self.max_length:
      # TODO: find more context based on embedding of dialog and closest support embedding
      context_example = random.choice(self.support)
      context_prompt = select_prompt(context_example['target'])
      added_context = example['dialogue'] + context_prompt
      added_size = len(added_context)
      current_size += added_size

      contexts.append(added_context)
    additional_context = ' '.join(added_context)
    return additional_context

  def collate_lm(self, examples):
    """ train and dev splits should not occur since you do not need gradient based training """
    assert(self.split not in ['train', 'dev'])

    for example in examples:
      target = example['target']
      prompt = select_prompt(target)
      dialog = example['dialogue'] + prompt
      additional_context = self.select_context(example['dialogue'], target)

      contexts.append(additional_context)
      dialogues.append(dialog)
      labels.append(target)

    inputs = self.tokenizer(contexts, dialogues, padding=pad_style,
                              truncation='only_first', return_tensors='pt').to(device)
    # targets = torch.tensor(labels, dtype=torch.long, device=device) # or torch.float of BCEWithLogits
    return inputs, labels

class MetaLearnDataset(InContextDataset):

  def collate_lm(self, examples):
    """
    train - use support dataset
    dev - use support dataset, do not include the label
    test - use the query dataset, do not include the label
    """
    contexts, dialogues, labels = [], [], []

    if self.split == 'train':
      eos = self.tokenizer.eos_token
      for example in examples:
        target = example['target']
        prompt = select_prompt(target)
        dialog = example['dialogue'] + prompt + target['value'] + eos
        additional_context = self.select_context(example['dialogue'], target)

        contexts.append(additional_context)
        dialogues.append(dialog)
      inputs = self.tokenizer(contexts, dialogues, padding=pad_style,
                                truncation='only_first', return_tensors='pt').to(device)
      labels = inputs['input_ids']

    elif self.split == 'dev':
      for example in examples:
        target = example['target']
        prompt = select_prompt(target)
        dialog = example['dialogue'] + prompt
        additional_context = self.select_context(example['dialogue'], target)

        contexts.append(additional_context)
        dialogues.append(dialog)
        labels.append(target)
      inputs = self.tokenizer(contexts, dialogues, padding=pad_style, max_length=1010,
                                truncation='only_first', return_tensors='pt').to(device)

    elif self.split == 'test':
      inputs,labels = super().collate_lm(examples)

    return inputs, labels


class FineTuneDataset(BaseDataset):

  def collate_seq2seq(self, examples):
    """transforms a batch of examples into a features dict that can be fed into a T5 or BART model"""
    dialogues, labels = [], []

    for example in examples:
      dialog = example['context'] + '<sep>' + example['prompt']
      dialogues.append(dialog)
      labels.append(example['label'] if self.split == 'train' else example['target'])

    # self.tokenizer.pad_token = self.tokenizer.eos_token
    inputs = self.tokenizer(dialogues, padding='longest', max_length=1000,
                                truncation=True, return_tensors='pt').to(device)
    if self.split == 'train':
      targets = self.tokenizer(labels) # we do not want tensors
      target_tensor = self._pad_right(targets)
      return inputs, target_tensor
    else:
      return inputs, labels

  def collate_lm(self, examples):
    """transforms a batch of examples into a features dict that can be fed into a GPT model"""
    dialogues, labels = [], []
    eos = self.tokenizer.eos_token

    if self.split == 'train':
      for example in examples:
        context = example['dialogue']
        target = example['target']
        domain, slot, value = target['domain'], target['slot'], target['value']
        prompt = f" The {slot} for {domain} is "
        dialog = context + '<sep>' + prompt + value + eos
        dialogues.append(dialog)
      inputs = self.tokenizer(dialogues, padding=True, max_length=1024,
                                truncation=True, return_tensors='pt').to(device)
      targets = inputs['input_ids']
      return inputs, targets

    elif self.split in ['dev', 'test']:
      for example in examples:
        context = example['dialogue']
        target = example['target']
        domain, slot, value = target['domain'], target['slot'], target['value']
        prompt = f" The {slot} for {domain} is "
        dialog = context + '<sep>' + prompt
        dialogues.append(dialog)
        labels.append(target)
      inputs = self.tokenizer(dialogues, padding=True, max_length=1000,
                                truncation=True, return_tensors='pt').to(device)
      return inputs, labels
