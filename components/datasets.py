import os, pdb, sys
import numpy as np
import random
import mmap
import re
import torch
from torch.utils.data import Dataset

from assets.static_vars import device, DATASETS
from utils.make_prompt import find_prompt
from utils.help import standardize_format

class BaseDataset(Dataset):
  def __init__(self, args, examples, tokenizer, split):
    self.split = split
    self.shuffle = (split == 'train')
    self.data = self._unravel(examples, split)
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

  def _unravel(self, examples, split):
    # examples are grouped by conversation and turn by default
    data = []
    for convo_id, conversation in examples.items():
      if split == 'test':
        data.append(conversation)
      else:
        for global_id, turn in conversation.items():
          for example in turn:
            example['global_id'] = global_id
            data.append(example)
    return data

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
    """ replaces the query set data with the support set data for training """
    # self.supported_datasets = [name for name, _ in DATASETS.items() if name != left_out]
    query_set = self.data
    support_set = []
    for support_name, support_data in supports.items():
      if support_name != left_out:
        self.supported_datasets.append(support_name)
        for example in support_data[self.split]:
          example['corpus'] = support_name
          support_set.append(example)
        setattr(self, f"{support_name}_ont", support_data['ont'])
    
    self.leftout = query_set
    self.data = support_set
    self.size = len(self.data)

  def add_detective(self, detective):
    if self.detective is None:
      self.detective = detective

  @staticmethod
  def state_to_string(prev_state):
    """ Transform the dialog state (dict) into a string to be used for context """
    prev_state_string = ''
    for dom_slot in prev_state:
      domain, slot = dom_slot.split("-")
      domain, slot = standardize_format(domain, slot)
      if prev_state[dom_slot] == '<none>':
        continue
      prev_state_string += f'{domain} {slot} {prev_state[dom_slot]} , '
    return prev_state_string[:-2].strip()

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
      exemplar = self.detective.search(example, args.dataset)
      ctx_domain, ctx_slot, ctx_label = exemplar['dsv']
      ctx_prompt = find_prompt(args.prompt_style, ctx_domain, ctx_slot)
      state_string = super().state_to_string(exemplar['prev_state'])
      added_context = f"{state_string} {exemplar['history']} {ctx_prompt} {ctx_label}"
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
      if args.use_pre_state:
        state_string = super().state_to_string(example['prev_state'])
        dialog = state_string + ' ' + dialog

      contexts.append(context)
      dialogues.append(dialog)
      labels.append(target)

    inputs = self.tokenizer(contexts, dialogues, padding=True,
                              truncation='only_first', return_tensors='pt').to(device) 
    return inputs, labels

class MetaLearnDataset(BaseDataset):

  def select_context(self, args, example, joined_utts, use_oracle=False):
    bpe_tokens = self.tokenizer(joined_utts)
    current_size = len(bpe_tokens['input_ids'])
    eos_token = self.tokenizer.eos_token
    
    model_input_length = 2048 if args.size == 'large' else 1024
    max_allowed = model_input_length - 12

    self.detective.reset()
    contexts = []
    while current_size < max_allowed:
      exemplar = self.detective.search(example, example['corpus'], use_oracle)
      ctx_domain, ctx_slot, ctx_label = exemplar['dsv']
      ctx_prompt = find_prompt(args.prompt_style, ctx_domain, ctx_slot)
      added_context = f"{exemplar['history']} {ctx_prompt} {ctx_label}"
      contexts.append(added_context)

      tokenized_context = self.tokenizer(added_context)
      current_size += len(tokenized_context['input_ids'])

    additional_context = eos_token.join(contexts)
    return additional_context + eos_token

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
        state_str = super().state_to_string(example['prev_state'])
        history = ' '.join(example['utterances'])
        target = example['target']
        prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])

        dialog = state_str + history + prompt + ' ' + target['value'] + eos
        additional_context = self.select_context(args, example, history, True)

        contexts.append(additional_context)
        dialogues.append(dialog)
      inputs = self.tokenizer(contexts, dialogues, padding=True,
                                truncation='only_first', return_tensors='pt').to(device)
      labels = inputs['input_ids']

    elif self.split == 'dev':
      for example in examples:
        state_str = super().state_to_string(example['prev_state'])
        target = example['target']
        prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])

        dialog = state_str + ' '.join(example['utterances']) + prompt
        additional_context = self.select_context(args, example, dialog)

        contexts.append(additional_context)
        dialogues.append(dialog)
        labels.append(target)

      max_length = self.max_len - 12
      inputs = self.tokenizer(contexts, dialogues, padding=True, max_length=max_length,
                                truncation='only_first', return_tensors='pt').to(device)

    elif self.split == 'test':
      inputs, labels = self.collate_lm(examples)
    return inputs, labels


class FineTuneDataset(BaseDataset):

  def collate_seq2seq(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into a T5 or BART model"""
    dialogues, labels = [], []

    for example in examples:
      dialog = ' '.join(example['utterances'])
      state_string = super().state_to_string(example['prev_state'])
      dialog = f"{state_string} {dialog}"
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
      if self.split != 'test':
        target['global_id'] = example['global_id']

      state_string = super().state_to_string(example['prev_state'])
      prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])

      if self.split == 'train':
        dial_input = f"{dialog} {prompt} {target['value']} {eos}"
        max_length = self.max_len
      elif self.split in ['dev', 'test']:
        dial_input = f"{dialog} {prompt}"
        max_length = self.max_len - 12
      if args.use_pre_state:
        dial_input = f"{state_string}{dial_input}"
      dialogues.append(dial_input)
      labels.append(target)
    inputs = self.tokenizer(dialogues, padding=True, max_length=max_length,
                              truncation=True, return_tensors='pt').to(device)
    if self.split == 'train':
      return inputs, inputs['input_ids']
    else:
      return inputs, labels
