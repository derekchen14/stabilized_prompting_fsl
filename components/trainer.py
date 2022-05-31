#!/usr/bin/env python3
#
import os, sys, pdb
import numpy as np
import random
import math

from torch import nn, no_grad
from tqdm import tqdm as progress_bar
from components.logger import ExperienceLogger
from components.detector import ExemplarDetective

from utils.help import *
from utils.process import process_data, get_dataloader
from utils.arguments import solicit_params
from utils.load import load_tokenizer, load_model, load_data, load_best_model, load_support
from utils.evaluate import eval_quantify, eval_qualify, test_quantify, parse_output
from assets.static_vars import debug_break
from transformers.deepspeed import deepspeed_init, deepspeed_reinit, is_deepspeed_zero3_enabled
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach



class DSTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    labels = inputs.pop("labels")
    # forward pass
    outputs = model(**inputs, labels=labels)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    return (loss, outputs) if return_outputs else loss


  def prediction_step(
      self,
      model,
      inputs,
      prediction_loss_only,
      ignore_keys = None,
  ):
    # do not compute the loss during evaluation
    has_labels = all(inputs.get(k) is not None for k in self.label_names)
    inputs = self._prepare_inputs(inputs)
    if ignore_keys is None:
        if hasattr(self.model, "config"):
            ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
        else:
            ignore_keys = []

    # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
    if has_labels:
        labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
        if len(labels) == 1:
            labels = labels[0]
    else:
        labels = None

    with torch.no_grad():
        loss = None
        del inputs["labels"]
        with self.autocast_smart_context_manager():
          outputs = model(**inputs)
        if isinstance(outputs, dict):
          logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
        else:
          logits = outputs
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
          self._past = outputs[self.args.past_index - 1]

    if prediction_loss_only:
        return (loss, None, None)

    logits = nested_detach(logits)
    if len(logits) == 1:
        logits = logits[0]

    return (loss, logits, labels)