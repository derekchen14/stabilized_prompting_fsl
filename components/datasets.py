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
    self.args = args
    self.split = split
    self.shuffle = (split == 'train')
    self.data = self._unravel(examples, split)

    self.tokenizer = tokenizer
    self.supported_datasets = []
    
    self.task = args.task
    self.max_len = args.maximum_length
    self.ctx_len = args.context_length
    self.model_type = args.model
    self.detective = None


  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


  def _unravel(self, examples, split):
    # examples are grouped by conversation and turn by default
    data = []
    for convo_id, conversation in examples.items():
      if split == 'test':
        if len(conversation) > 0:  # some convos have no examples after filtering
          data.append(conversation)
      else:
        for global_id, turn in conversation.items():
          for example in turn:
            data.append(example)
    return data

  def _pad_right(self, targets):
    max_vec_len = max([len(vector) for vector in targets.input_ids])
    assert(max_vec_len < 24)
    if max_vec_len > 16:
      max_vec_len = 17

    padded = []
    for vector in targets.input_ids:
      if len(vector) > 16:
        vector = vector[:17]
      else:
        diff = max_vec_len - len(vector)
        for i in range(diff):
          vector.append(-100)  # id of -100 means to not pay attention on training
      padded.append(vector)

    target_tensor = torch.tensor(padded).to(device)
    return target_tensor

  def select_context(self, args, example, history, detective_id=0):
    bpe_tokens = self.tokenizer(history)
    current_size = len(bpe_tokens['input_ids'])
    eos = self.tokenizer.eos_token
    
    model_input_length = 512 if args.model == 't5' else 1024
    max_allowed = model_input_length - 16

    self.detective[detective_id].reset()
    contexts = []
    while current_size < max_allowed:
      exemplar = self.detective[detective_id].search(example)
      ctx_domain, ctx_slot, ctx_label = exemplar['dsv']
      ctx_prompt = find_prompt(args.prompt_style, ctx_domain, ctx_slot)
      state_str = self.__class__.state_to_string(exemplar['prev_state'])
      context = f"{state_str}{exemplar['history']} {ctx_prompt} {ctx_label}{eos}"
      contexts.append(context)

      tokenized_context = self.tokenizer(context)
      current_size += len(tokenized_context['input_ids'])

    self.detective[detective_id].num_exemplars.append(len(contexts))
    additional_context = ' '.join(contexts)
    return additional_context

  def add_detective(self, detective):
    if self.detective is None:
      if type(detective) is list:
        self.detective = detective
      else:
        self.detective = [detective]
    search_methods = " ".join([detective.search_method for detective in self.detective])
    print(f"Using {search_methods} distance to search ...")

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

  def remove_special(self, text):
    text = text.replace('<agent>', ' agent:')
    text = text.replace('<customer>', ' customer:')
    text = text.replace('<none>', 'none')
    text = text.replace('<label>', 'answer:')
    text = text.replace('<sep>', ';')
    text = text.replace('<remove>', 'none')
    # text = text.replace('<pad>', '[PAD]')
    return text

  def collate(self, args, examples):
    """ train and dev splits should not occur since you do not need gradient based training """
    assert(self.split not in ['train', 'dev'])
    contexts, dialogues, labels = [], [], []

    for example in examples:
      target = example['target']
      prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])

      state_str = super().state_to_string(example['prev_state'])
      history = ' '.join(example['utterances'])

      ensemble_range = args.ensemble if args.ensemble > 1 else 1
      for detective_id in range(ensemble_range):
        additional_context = self.remove_special(self.select_context(args, example, history, detective_id))
        dialog = self.remove_special(f"{state_str}{history} {prompt} <extra_id_0>")

        contexts.append(additional_context)
        dialogues.append(dialog)
        
        target['history'] = history
        labels.append(target)

    inputs = self.tokenizer(contexts, dialogues, padding=True, max_length=512,
                              truncation='only_first', return_tensors='pt').to(device) 
    return inputs, labels


class MetaLearnDataset(BaseDataset):

  def _filter_set(self, support_data):
    new_support = []
    relevant_domains = ['hotel', 'rental', 'rideshare', 'event', 'travel', 'restaurant']
    if self.split == 'train':
      for example in support_data['train']:
        if example['target']['domain'] in relevant_domains:
          new_support.append(example)
      support_data['train'] = new_support
    elif self.split == 'dev':
      for example in support_data['dev']:
        if example['target']['domain'] in relevant_domains:
          new_support.append(example)
      support_data['dev'] = new_support
    return support_data

  def add_support(self, supports, left_out):
    """ replaces the query set data with the support set data for training """
    # self.supported_datasets = [name for name, _ in DATASETS.items() if name != left_out]
    query_set = self.data
    support_set = []
    for support_name, support_data in supports.items():
      if support_name != left_out:
        if support_name == 'sgd':
          support_data = self._filter_set(support_data)
        self.supported_datasets.append(support_name)
        support_set.extend(support_data[self.split])
        setattr(self, f"{support_name}_ont", support_data['ont'])
        if self.split == 'train':
          print(f"Added {DATASETS[support_name]} as a support dataset.")
    if self.split == 'train':
      print(f"Moved {DATASETS[left_out]} to become the query dataset.")
    
    self.leftout = query_set
    self.data = support_set

  def collate_seq2seq(self, args, examples):
    contexts, dialogues, labels = [], [], []

    for example in examples:
      history = ' '.join(example['utterances'])
      target = example['target']
      prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])
      state_str = super().state_to_string(example['prev_state'])
      
      ensemble_range = args.ensemble if args.ensemble > 1 else 1
      for detective_id in range(ensemble_range):
        additional_context = self.select_context(args, example, history, detective_id)
        dialog = f"{state_str}{history} {prompt}"

        contexts.append(additional_context)
        dialogues.append(dialog)

        if self.split == 'train':
          labels.append(target['value'])
        else:
          target['history'] = history
          labels.append(target)
    """ max length is hardcoded to 512, which is the max allowed
    there is no need to subtract 16 since we do not need to save any space for the target value
    instead the sequence of target is taken care of by the decoder """
    inputs = self.tokenizer(contexts, dialogues, padding=True, max_length=512,
                              truncation='longest_first', pad_to_multiple_of=8, return_tensors='pt').to(device)
    # inputs = self.tokenizer(contexts, dialogues, padding=True, max_length=512, truncation='longest_first', pad_to_multiple_of=8, return_tensors='pt')
    if self.split == 'train':
      targets = self.tokenizer(labels)
      target_tensor = self._pad_right(targets)
      return inputs, target_tensor
    else:
      return inputs, labels

  def collate_lm(self, args, examples):
    """
    train - use support dataset
    dev - use support dataset, do not include the label
    test - use the query dataset, do not include the label
    """
    contexts, dialogues, labels = [], [], []
    eos = self.tokenizer.eos_token

    for example in examples:
      history = ' '.join(example['utterances'])
      target = example['target']
      prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])
      state_str = super().state_to_string(example['prev_state'])

      if self.split == 'train':
        additional_context = self.select_context(args, example, history)
        dialog = f"{state_str}{history} {prompt} {target['value']}{eos}"
        max_len = self.max_len
      elif self.split in ['dev', 'test']:
        additional_context = self.select_context(args, example, history)
        dialog = f"{state_str}{history} {prompt}"
        max_len = self.max_len - 16

      contexts.append(additional_context)
      dialogues.append(dialog)
      labels.append(target)
      
    inputs = self.tokenizer(contexts, dialogues, padding=True, max_length=max_len,
                                truncation='only_first', return_tensors='pt').to(device)
    if self.split == 'train':
      labels = inputs['input_ids']
    return inputs, labels

class PreTrainDataset(MetaLearnDataset):
  """ Uses the meta learn dataset pre-processing to add different corpora as support sets.
  At the same time, uses the standard fine-tuning method to collate since not doing ICL"""

  def collate_seq2seq(self, args, examples):
    dialogues, labels = [], []

    for example in examples:
      history = ' '.join(example['utterances'])
      target = example['target']
      state_str = super().state_to_string(example['prev_state'])
      prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])
      
      dialog = f"{state_str}{history} {prompt}"
      dialogues.append(dialog)

      if self.split == 'train':
        labels.append(target['value'])
      else:
        target['history'] = history
        labels.append(target)

    inputs = self.tokenizer(dialogues, padding='longest', max_length=512,
                              truncation=True, pad_to_multiple_of=8, return_tensors='pt').to(device)
    if self.split == 'train':
      targets = self.tokenizer(labels) # we do not want tensors
      target_tensor = self._pad_right(targets)
      return inputs, target_tensor
    else:
      return inputs, labels

class FineTuneDataset(BaseDataset):

  def collate_seq2seq(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into a T5 or BART model"""
    dialogues, labels = [], []

    for example in examples:
      history = ' '.join(example['utterances'])
      target = example['target']
      state_str = super().state_to_string(example['prev_state'])
      prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])
      
      dialog = f"{state_str}{history} {prompt}"
      dialogues.append(dialog)

      if self.split == 'train':
        labels.append(target['value'])
      else:
        target['history'] = history
        labels.append(target)

    inputs = self.tokenizer(dialogues, padding='longest', max_length=512,
                              truncation=True, pad_to_multiple_of=8, return_tensors='pt').to(device)
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
      history = ' '.join(example['utterances'])
      target = example['target']
      state_str = super().state_to_string(example['prev_state'])
      prompt = find_prompt(args.prompt_style, target['domain'], target['slot'])

      if self.split == 'train':
        dialog = f"{state_str}{history} {prompt} {target['value']}{eos}"
        max_length = self.max_len
      elif self.split in ['dev', 'test']:
        dialog = f"{state_str}{history} {prompt}"
        max_length = self.max_len - 16
      
      dialogues.append(dialog)
      labels.append(target)

    inputs = self.tokenizer(dialogues, padding=True, max_length=max_length,
                              truncation=True, return_tensors='pt').to(device)

    if self.split == 'train':
      return inputs, inputs['input_ids']
    else:
      return inputs, labels
