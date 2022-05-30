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