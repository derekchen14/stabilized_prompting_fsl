import os, pdb, sys
import numpy as np
import random

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from collections import defaultdict
from assets.static_vars import device

class BaseModel(nn.Module):
  def __init__(self, args, encoder, tokenizer):
    self.name = 'classify'
    self.encoder = encoder
    self.model_type = args.model
    self.tokenizer = tokenizer

    self.verbose = args.verbose
    self.debug = args.debug
    self.weight_decay = args.weight_decay
    self.dropout = nn.Dropout(args.drop_rate)

    self.dense = nn.Linear(args.embed_dim, args.hidden_dim)
    self.gelu = nn.GELU()
    self.classify = nn.Linear(args.hidden_dim, len(ontology)) 
    self.softmax = nn.LogSoftmax(dim=1)
    self.criterion = nn.CrossEntropyLoss()  # performs LogSoftmax and NegLogLike Loss

  def forward(self, inputs, targets, outcome='logit'):
    if self.model_type == 'roberta':
      """ By default, the encoder returns result of (batch_size, seq_len, vocab_size) under 'logits'
      When the output_hs flag is turned on, the output will also include a tuple under 'hidden_states'
      The tuple has two parts, the first is embedding output and the second is hidden_state of last layer
      """
      enc_out = self.encoder(**inputs, output_hidden_states=True) 
      cls_token = enc_out['hidden_states'][1][:, 0, :]
    else:
      enc_out = self.encoder(**inputs)
      cls_token = enc_out['last_hidden_state'][:, 0, :]   # batch_size, embed_dim
    
    hidden1 = self.dropout(cls_token)
    hidden2 = self.dense(hidden1)
    hidden3 = self.gelu(hidden2)
    hidden4 = self.dropout(hidden3)
    logits = self.classify(hidden4)  # batch_size, num_classes
    logits = logits.squeeze()
    
    loss = self.criterion(logits, targets)
    output = logits if outcome == 'logit' else self.softmax(logits)
    return output, loss

class GenerateModel(BaseModel):
  # Main model for general classification prediction
  def __init__(self, args, core, tokenizer):
    super().__init__(args, core, tokenizer)
    self.name = 'generate'

  def forward(self, inputs, targets, outcome='logit'):
    enc_out = self.encoder(**inputs)
    cls_token = enc_out['last_hidden_state'][:, 0, :]   # batch_size, embed_dim
    
    logits = cls_token
    loss = self.criterion(logits, targets)
    output = logits if outcome == 'logit' else self.softmax(logits)
    return output, loss

