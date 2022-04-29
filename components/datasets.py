import os, pdb, sys
import numpy as np
import random
import mmap
import torch
from torch.utils.data import Dataset

from assets.static_vars import device, DATASETS
from utils.make_prompt import find_prompt

class BaseDataset(Dataset):
  def __init__(self, args, examples, tokenizer, split):
    self.split = split
    self.shuffle = (split == 'train')
    self.data = examples
    self.size = len(self.data)

    self.tokenizer = tokenizer
    self.supported_datasets = []
    
    self.task = args.task
    self.max_len = args.maximum_length
    self.ctx_len = args.context_length
    self.model_type = args.model
    self.detective = None

  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    return self.data[idx]

  def _pad_right(self, targets):
    max_vec_len = max([len(vector) for vector in targets.input_ids])
    assert(max_vec_len < 12)

    padded = []
    for vector in targets.input_ids:
      diff = max_vec_len - len(vector)
      for i in range(diff):
        vector.append(-100)  # id of -100 means to not pay attention on training
      padded.append(vector)

    target_tensor = torch.tensor(padded).to(device)
    return target_tensor

  def add_support(self, supports, left_out):
    # self.supported_datasets = [name for name, _ in DATASETS.items() if name != left_out]
    for support_name, support_data in supports.items():
      if support_name != left_out:
        setattr(self, f"{support_name}_data", support_data['data'])
        setattr(self, f"{support_name}_ont", support_data['ont'])
        self.supported_datasets.append(support_name)

  def add_detective(self, detective):
    if self.detective is None:
      self.detective = detective

  def collate_lm(self, args, examples):
    raise NotImplementedError

  def collate_seq2seq(self, args, examples):
    raise NotImplementedError

  def collate(self, args, examples):
    if self.model_type == 'gpt':
      return self.collate_lm(args, examples)
    elif self.model_type in ['bart', 't5']:
      return self.collate_seq2seq(args, examples)

  def collate_func(self, examples):
    return examples


class InContextDataset(BaseDataset):

  def move_elsewhere(self):
    if len(self.supported_datasets) > 0:
      for name, _ in DATASETS.items():
        support = getattr(self, f"{name}_data")
        context_example = self.detective.search(example, support)

  def select_context(self, args, example, joined_utts):
    bpe_tokens = self.tokenizer(joined_utts)
    current_size = len(bpe_tokens['input_ids'])
    assert(current_size > 1)
    eos_token = self.tokenizer.eos_token
    
    model_input_length = 2048 if args.size == 'large' else 1024
    max_allowed = model_input_length - 12

    self.detective.reset()
    contexts = []
    while current_size < max_allowed:
      exemplar = self.detective.search(example)
      ctx_domain, ctx_slot, ctx_label = exemplar['dsv']
      ctx_prompt = find_prompt(args.prompt_style, ctx_domain, ctx_slot)
      added_context = f"{exemplar['history']} {ctx_prompt} {ctx_label}"
      contexts.append(added_context)

      tokenized_context = self.tokenizer(added_context)
      current_size += len(tokenized_context['input_ids'])

    additional_context = eos_token.join(contexts)
    cleaned_context = self.remove_special(additional_context)
    return cleaned_context + eos_token

  def remove_special(self, text):
    text = text.replace('<agent>', 'agent:')
    text = text.replace('<customer>', 'customer:')
    text = text.replace('<none>', 'none')
    text = text.replace('<label>', 'answer:')
    text = text.replace('<sep>', ';')
    text = text.replace('<remove>', 'none')
    text = text.replace('<pad>', '[PAD]')
    return text

  def collate_lm(self, args, examples):
    """ train and dev splits should not occur since you do not need gradient based training """
    assert(self.split not in ['train', 'dev'])
    contexts, dialogues, labels = [], [], []

    for example in examples:
      joined_utts = ' '.join(example['utterances'])
      target = example['target']
      prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])
      context = self.select_context(args, example, joined_utts)
      dialog = self.remove_special(f'{joined_utts} {prompt}')

      contexts.append(context)
      dialogues.append(dialog)
      labels.append(target)

    inputs = self.tokenizer(contexts, dialogues, padding=True,
                              truncation='only_first', return_tensors='pt').to(device) 
    """ 
    trick = inputs['input_ids']
    treat = self.tokenizer.batch_decode(trick)
    for label, entry in zip(labels, treat):
      print(entry)
      pdb.set_trace()
    """
    return inputs, labels

class MetaLearnDataset(InContextDataset):

  def collate_lm(self, args, examples):
    """
    train - use support dataset
    dev - use support dataset, do not include the label
    test - use the query dataset, do not include the label
    """
    contexts, dialogues, labels = [], [], []

    if self.split == 'train':
      eos = self.tokenizer.eos_token
      for example in examples:
        history = ' '.join(example['utterances'])
        target = example['target']
        prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])
        dialog = history + prompt + target['value'] + eos
        additional_context = self.select_context(example)

        contexts.append(additional_context)
        dialogues.append(dialog)
      inputs = self.tokenizer(contexts, dialogues, padding=True,
                                truncation='only_first', return_tensors='pt').to(device)
      labels = inputs['input_ids']

    elif self.split == 'dev':
      for example in examples:
        target = example['target']
        prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])
        dialog = ' '.join(example['utterances']) + prompt
        additional_context = self.select_context(example)

        contexts.append(additional_context)
        dialogues.append(dialog)
        labels.append(target)

      max_length = self.max_len - 12
      inputs = self.tokenizer(contexts, dialogues, padding=True, max_length=max_length,
                                truncation='only_first', return_tensors='pt').to(device)

    elif self.split == 'test':
      inputs, labels = super().collate_lm(examples)

    return inputs, labels


class FineTuneDataset(BaseDataset):

  def collate_seq2seq(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into a T5 or BART model"""
    dialogues, labels = [], []

    for example in examples:
      dialog = ' '.join(example['utterances'])
      dialogues.append(dialog)
      labels.append(example['target']['value'] if self.split == 'train' else example['target'])

    max_length = self.max_len - 12
    inputs = self.tokenizer(dialogues, padding='longest', max_length=max_length,
                                truncation=True, return_tensors='pt').to(device)
    if self.split == 'train':
      targets = self.tokenizer(labels) # we do not want tensors
      target_tensor = self._pad_right(targets)
      return inputs, target_tensor
    else:
      return inputs, labels

  def collate_lm(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into a GPT model"""
    dialogues, labels = [], []
    eos = self.tokenizer.eos_token

    for example in examples:
      dialog = ' '.join(example['utterances'])
      target = example['target']
      prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])

      if self.split == 'train':
        dialog += f" {prompt} {target['value']} {eos}"
        max_length = self.max_len
      elif self.split in ['dev', 'test']:
        dialog += f" {prompt} "
        max_length = self.max_len - 12
      dialogues.append(dialog)
      labels.append(target)

    inputs = self.tokenizer(dialogues, padding=True, max_length=max_length,
                              truncation=True, return_tensors='pt').to(device)
    if self.split == 'train':
      return inputs, inputs['input_ids']
    else:
      return inputs, labels
