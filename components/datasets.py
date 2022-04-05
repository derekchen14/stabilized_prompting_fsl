import os, pdb, sys
import numpy as np
import random
import mmap
import torch
from torch.utils.data import Dataset

from assets.static_vars import device, DATASETS
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
    self.max_len = args.maximum_length
    self.ctx_len = args.context_length
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

  def add_support(self, supports, left_out):
    self.supported_datasets = ['sgd', 'dstc']
    # self.supported_datasets = [name for name, _ in DATASETS.items() if name != left_out]
    for support_name, support_data in supports.items():
      assert(support_name in self.supported_datasets)
      setattr(self, f"{support_name}_data", support_data['data'])
      setattr(self, f"{support_name}_ont", support_data['ont'])

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
    print("good enough for part 1")
    sys.exit()
    current_size = len(dialog)
    contexts = []

    while current_size < self.max_length:
      # TODO: find more context based on embedding of query and closest support embedding
      context_example = search_similar_context(dialog, self.support, target)
      context_target = context_example['target']
      context_label = context_target['value']
      context_prompt = find_prompt(context_target)
      added_context = example['dialogue'] + context_prompt + context_label
      added_size = len(added_context)
      current_size += added_size
      contexts.append(added_context)

    additional_context = ' <sep> '.join(contexts)
    return additional_context

  def collate_lm(self, examples):
    """ train and dev splits should not occur since you do not need gradient based training """
    assert(self.split not in ['train', 'dev'])

    for example in examples:
      target = example['target']
      prompt = select_prompt(target)
      dialog = example['dialogue'] + prompt
      # additional_context = self.select_context(example['dialogue'], target)

      contexts.append(additional_context)
      dialogues.append(dialog)
      labels.append(target)

    inputs = self.tokenizer(contexts, dialogues, padding=pad_style,
                              truncation='only_first', return_tensors='pt').to(device) 
   
    trick = inputs['input_ids']
    treat = self.tokenizer.batch_decode(trick)
    for entry in treat:
      print(entry.replace('<pad>', '|'))
      pdb.set_trace()

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
        dialog = example['history'] + prompt + target['value'] + eos
        additional_context = self.select_context(example['current'], target)

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

    for example in examples:
      history = example['history'][-self.ctx_len:]
      current_utt = example['current']
      target = example['target']
      domain, slot, value = target['domain'], target['slot'], target['value']
      prompt = select_prompt(target)

      if self.split == 'train':
        dialog = history + ' ' + current_utt + '<sep>' + prompt + value + eos
        max_length = self.max_len
      elif self.split in ['dev', 'test']:
        dialog = history + ' ' + current_utt + '<sep>' + prompt
        max_length = self.max_len - 10
      dialogues.append(dialog)
      labels.append(target)

    inputs = self.tokenizer(dialogues, padding=True, max_length=max_length,
                              truncation=True, return_tensors='pt').to(device)
    if self.split == 'train':
      return inputs, inputs['input_ids']
    else:
      return inputs, labels
