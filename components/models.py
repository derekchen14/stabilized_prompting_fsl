import os, pdb, sys
import numpy as np
import random
import json

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from tqdm import tqdm as progress_bar
from collections import defaultdict
from assets.static_vars import device
from sentence_transformers import SentenceTransformer
from sentence_transformers.model_card_templates import ModelCardTemplate
from typing import List, Dict, Tuple, Type, Callable
    
class BaseModel(nn.Module):
  def __init__(self, args, encoder, tokenizer, ontology):
    super(BaseModel, self).__init__()
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
  def __init__(self, args, core, tokenizer, ontology):
    super().__init__(args, core, tokenizer, ontology)
    self.name = 'generate'

  def forward(self, inputs, targets, outcome='logit'):
    enc_out = self.encoder(**inputs)
    cls_token = enc_out['last_hidden_state'][:, 0, :]   # batch_size, embed_dim
    
    logits = cls_token
    loss = self.criterion(logits, targets)
    output = logits if outcome == 'logit' else self.softmax(logits)
    return output, loss

class SentenceBERT(SentenceTransformer):

  def qualify(self, features, utterances):
    chosen_id = random.randint(0, len(utterances))
    chosen_utt = utterances[chosen_id]
    chosen_embed = features['sentence_embedding'][chosen_id].unsqueeze(0)

    comparables = []
    for sent_embed, utterance in zip(features['sentence_embedding'], utterances):
      with torch.no_grad():
        score = torch.cosine_similarity(chosen_embed, sent_embed.unsqueeze(0))
      comp = (utterance, round(score.item(), 3))
      comparables.append(comp)
    comparables.sort(key=lambda x: x[1], reverse=True)

    print("Target utterance:", chosen_utt)
    print(f"Out of {len(utterances)} utterances, the 3 closest are:")
    count = 1
    for close, score in comparables[1:4]:
      print(f"   {count})", close, score)
      count += 1
    print(f"And the three furthest are:")
    count = 1
    for far, score in comparables[-3:]:
      print(f"   {count})", far, score)
      count += 1

  def fit(self, train_objective: Tuple[object, nn.Module],
      evaluator, epochs: int = 1,
      steps_per_epoch = None,
      scheduler_name: str = 'WarmupLinear',
      warmup_steps: int = 10000,
      optimizer_class = optim.AdamW,
      optimizer_params : Dict[str, object]= {'lr': 3e-5},
      weight_decay: float = 0.01,
      logging_steps: int = 0,
      evaluation_steps: int = 0,
      output_path: str = None,
      save_best_model: bool = True,
      max_grad_norm: float = 3,
      do_qual: bool=False,
      callback: Callable[[float, int, int], None] = None,
      checkpoint_path: str = None,
      checkpoint_save_steps: int = 2000,
      checkpoint_save_total_limit: int = 0):
    """
    Train the model with the given training objective
    Each training objective is sampled in turn for one batch.
    We sample only as many batches from each objective as there are in the smallest one
    to make sure of equal training with each dataset.

    :param train_objectives: Tuples of (DataLoader, LossFunction). Only accepts on tuple now.
    :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model 
            performance during training on held-out dev data. Used to determine the best model that is saved to disc.
    :param epochs: Number of epochs for training
    :param steps_per_epoch: Number of training steps per epoch. If set to None (default), 
            one epoch is equal the DataLoader size from train_objectives.
    :param scheduler_name: Learning rate scheduler. Available schedulers: 
            constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
    :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), 
            the learning rate is increased from o up to the maximal learning rate. 
            After these many training steps, the learning rate is decreased linearly back to zero.
    :param optimizer_class: Optimizer
    :param optimizer_params: Optimizer parameters
    :param weight_decay: Weight decay for model parameters
    :param logging_steps: If > 0, evaluate the model using evaluator after each number of training steps
    :param evaluation_steps: If > 0 and do qualify print out the closest relations per batch
    :param output_path: Storage path for the model and evaluation files
    :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
    :param max_grad_norm: Used for gradient normalization.
    :param callback: Callback function that is invoked after each evaluation.
        It must accept the following three parameters in this order:
        `score`, `epoch`, `steps`
    :param checkpoint_path: Folder to save checkpoints during training
    :param checkpoint_save_steps: Will save a checkpoint after so many steps
    :param checkpoint_save_total_limit: Total number of checkpoints to store
    """

    ##Add info to model card
    dataloader, loss_model = train_objective
    info_loss_functions =  ModelCardTemplate.get_train_objective_info(dataloader, loss_model)
    info_loss_functions = "\n\n".join([text for text in info_loss_functions])
    eval_name = evaluator.__class__.__module__
    
    info_fit_parameters = {"evaluator": eval_name, "epochs": epochs, "steps_per_epoch": steps_per_epoch,
        "scheduler": scheduler_name, "warmup_steps": warmup_steps, "weight_decay": weight_decay,
        "optimizer_class": str(optimizer_class), "optimizer_params": optimizer_params, 
        "evaluation_steps": evaluation_steps, "logging_steps": logging_steps, "max_grad_norm": max_grad_norm}
    print(info_fit_parameters)
    ifp = json.dumps(info_fit_parameters, indent=4, sort_keys=True)

    self._model_card_text = None
    self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", ifp)
    self.best_score = -9999999
    self.to(self._target_device)
    loss_model.to(self._target_device)

    # Use smart batching
    dataloader.collate_fn = self.smart_batching_collate
    if steps_per_epoch is None or steps_per_epoch == 0:
      steps_per_epoch = len(dataloader)
    num_train_steps = int(steps_per_epoch * epochs)

    # Prepare optimizers
    param_optimizer = list(loss_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
    scheduler = self._get_scheduler(optimizer, scheduler=scheduler_name, 
              warmup_steps=warmup_steps, t_total=num_train_steps)

    global_step = 0
    data_iterators = []
    tok = self._first_module().tokenizer
    
    for epoch in progress_bar(range(epochs), desc="Epoch", total=epochs):
      training_steps = 0
      loss_model.zero_grad()
      loss_model.train()
      chosen_batch = random.randint(0, 100-1) # len(dataloader)

      losses = []
      for features, labels in dataloader:
        if labels.dtype == torch.int64:
          labels = labels.type(torch.float32)

        loss_value = loss_model(features, labels)
        losses.append(loss_value.item())
        loss_value.backward()

        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        training_steps += 1
        global_step += 1

        if logging_steps > 0 and training_steps % logging_steps == 0:
          avg_loss = round(np.mean(losses), 3) 
          print(f"Step {training_steps}/{steps_per_epoch}, Loss: {avg_loss}")
        if checkpoint_path is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
          print("Saving checkpoint")
          self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
        if do_qual and training_steps == chosen_batch:
          fzero = features[0]
          utterances = tok.batch_decode(fzero['input_ids'], skip_special_tokens=True)

      if do_qual:
        self.qualify(loss_model.model, fzero, utterances)
      avg_loss = round(np.mean(losses), 3)
      def caller(raw_score, epoch, steps):
        score = round(raw_score, 3)
        print(f"Step {steps}/{steps_per_epoch}, Loss: {avg_loss}, Score: {score}")
      self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, caller)

    if checkpoint_path is not None:
      self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


