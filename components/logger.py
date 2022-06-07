import os, pdb, sys
import torch
import random
import logging
import re
import glob
import shutil
import math

import pandas as pd
import numpy as np
import time as tm
from datetime import datetime
from collections import defaultdict
from utils.help import model_match

class ExperienceLogger:
  def __init__(self, args, ontology, save_dir):
    self.args = args
    self.learning_rate = args.learning_rate
    self.model_type = args.model

    self.save_path = save_dir
    self.ontology = ontology
    self.optimizer = None
    self.scheduler = None

    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(__name__)
    log_name = f'{args.prompt_style}_lr{args.learning_rate}_clen{args.context_length}.log'
    log_path = os.path.join(save_dir, log_name)
    self.logger.addHandler(logging.FileHandler(log_path))
    self.logger.debug(args)
    self.log_info(args)

    self.global_step = 0
    self.eval_step = 0
    self.log_interval = args.log_interval
    self.epoch = 1   # epoch count
    self.num_epochs = args.n_epochs

    self.best_score = { 'epoch': 1, 'chunk': 1 }
    self.metric = 'jga'
    self.best_score[self.metric] = 0
    self.do_save = args.do_save
    self.differences = []
    self.past_metrics = []

    self.logging_loss = 0.0
    self.tr_loss = 0.0
    self.eval_loss = 0.0

    self.start_time_chunk = None
    self.chunk_num = 0

  def log_info(self, text):
    self.logger.info(text)

  def start_train(self, total_step):
    self.logger.info("***** Running training *****")
    self.logger.info("  Num Epochs = %d" % self.args.n_epochs)
    self.logger.info("  Total train batch size  = %d" % self.args.batch_size)
    self.logger.info("  Total optimization steps = %d" % total_step)
    self.logger.info("  Running experiment for {}".format(self.style))

  def start_epoch(self, dataloader, percent):
    self.logger.info(f"Starting epoch {self.epoch} of {self.num_epochs}")
    self.start_time = tm.time()
    self.num_steps = len(dataloader)
    self.breakpoint = int(self.num_steps * percent)

  def end_epoch(self):
    self.epoch += 1
    self.end_time = tm.time()

    raw_diff = self.end_time - self.start_time
    minute_diff = round(raw_diff / 60.0, 3)
    self.differences.append(minute_diff)
    avg_diff = round(np.average(self.differences), 3)

    met = round(self.best_score[self.metric] * 100, 2)
    self.logger.info(f"Best epoch is {self.best_score['epoch']} with {met}% accuracy")
    self.logger.info(f"Current epoch took {minute_diff} min, average is {avg_diff} min")

    return self.early_stop(met)

  def start_chunk(self):
    self.logger.info(f"Starting chunk {self.chunk_num}")
    self.start_time_chunk = tm.time()

  def end_chunk(self):
    self.chunk_num += 1
    self.end_time_chunk = tm.time()

    if self.start_time_chunk is None:
      raw_diff = self.end_time_chunk - self.start_time
    else:
      raw_diff = self.end_time_chunk - self.start_time_chunk
    minute_diff = round(raw_diff / 60.0, 3)

    met = round(self.best_score[self.metric] * 100, 2)
    self.logger.info(f"Best chunk is {self.best_score['chunk']} with {met}% accuracy")
    self.logger.info(f"Current chunk took {minute_diff} min")

    return self.early_stop(met)


  def early_stop(self, metric):
    below_threshold = False
    
    if self.epoch > 3 and self.args.debug:
      below_threshold = True

    self.past_metrics.append(metric)
    if len(self.past_metrics) >= 4:
      trail = self.past_metrics[-4:]
      if all(x == trail[0] for x in trail):
        below_threshold = True

    if below_threshold:
      if self.args.checkpoint_interval > 0:
        self.logger.info(f"Ran out of patience, early stopped at chunk {self.chunk_num}")
      else:
        self.logger.info(f"Ran out of patience, early stopped at epoch {self.epoch}")
    return below_threshold

  def start_eval(self, num_batches, eval_interval):
    self.eval_step = 0
    if eval_interval == 'whole':
      self.interval_checkpoint = math.ceil(num_batches / 1)  # redundant
    elif eval_interval == 'half':
      self.interval_checkpoint = math.ceil(num_batches / 2)
    elif eval_interval == 'quarter':
      self.interval_checkpoint = math.ceil(num_batches / 4)
    elif eval_interval == 'tenth':
      self.interval_checkpoint = math.ceil(num_batches / 10)
    self.final_step = num_batches

    # to allow for more variety in qualitative review
    self.previous_outputs = []
    self.previous_targets = []

  def log_eval(self, qualify, output_strings, target_dicts):
    self.eval_loss = 0  # no loss, since inference only
    self.eval_step += 1
    self.past_history = []

    is_done = self.eval_step >= self.final_step
    is_checkpoint = self.eval_step % self.interval_checkpoint == 0 

    self.previous_outputs.append(output_strings)
    self.previous_targets.append(target_dicts)

    if len(self.previous_targets) > 5:
      position = random.randint(0,5)
      self.previous_targets.pop(position)
      self.previous_outputs.pop(position)

    if qualify and is_checkpoint:
      position = random.randint(0,len(self.previous_targets) - 1)
      selected_outputs = self.previous_outputs.pop(position)
      selected_targets = self.previous_targets.pop(position)

      for out_str, target in zip(selected_outputs, selected_targets):
        replaced = out_str.replace("<pad>","")
        
        if 'history' in target:
          history = target['history']
          prompt_and_pred = replaced  # just the prediction for seq2seq models
        else:
          try:
            history, prompt_and_pred = replaced.rsplit('<sep>', 1)
          except(ValueError):
            history = replaced
            prompt_and_pred = replaced[-20:]

        global_id = target['global_id']
        if global_id not in self.past_history:
          print(history, global_id)
          self.past_history.append(global_id)
        print('  predicted:', prompt_and_pred.strip(), ', actual:', target['value'])
    return is_done or is_checkpoint

  def train_stop(self, args, step, debug_break):
    if args.debug and step >= debug_break*args.log_interval:
      return True
    if step > self.breakpoint:
      print(f"Training stopped early at step {step} to save time")
      return True
    return False

  def log_train(self, step, scheduler):
    self.global_step += 1
    now = datetime.now().strftime("%d-%H:%M:%S")

    step_report = f'{step}/{self.num_steps}'
    adjusted_lr = round(scheduler.get_last_lr()[0], 6)
    lr_report = f"Learning rate: {adjusted_lr}"
    current_loss = 10000 * ((self.tr_loss - self.logging_loss) / self.log_interval)
    loss_report = 'Mean loss: %.5f' % current_loss
    self.logging_loss = self.tr_loss

    if self.global_step < 100 and self.global_step % 10 == 0:
      print(self.global_step)
    if step % self.log_interval == 0 and step > 1:
      print(f"[{now}] Steps: {step_report}, {lr_report}, {loss_report}")

  def save_best_model(self, model, tokenizer, prune_keep):
    if self.do_save and self.best_score[self.metric] > 0.1:
      learning_rate = str(self.args.learning_rate)
      accuracy = str(self.best_score[self.metric] * 10000)[:3]
      style = self.args.prompt_style
      context_length = self.args.context_length
      ckpt_name = f'{style}_lr{learning_rate}_clen{context_length}_epoch{self.epoch}_acc{accuracy}.pt'
      ckpt_path = os.path.join(self.save_path,ckpt_name)

      # model_to_save = model.module if hasattr(model, 'module') else model
      # torch.save(model_to_save.state_dict(), ckpt_path)   # Standard Pytorch method
      model.save_pretrained(ckpt_path)
      # tokenizer.save_pretrained(ckpt_path)  # Huggingface method, creates a new folder
      print(f"Saved a model at {ckpt_path}")
      if prune_keep > 0:
        self.prune_saves(num_keep=prune_keep)

  def prune_saves(self, is_directory=False, num_keep=5):
    # files = [f for f in os.listdir(self.save_path) if f.endswith('.pt')]
    folders = glob.glob(os.path.join(self.save_path, "*pt"))
    
    if len(folders) > num_keep:
      # scores_and_files = []
      acc_and_folders = []
      # for fname in files:
      for fname in folders:
        re_str = r'acc([0-9]{3})\.pt$'
        regex_found = re.findall(re_str, fname)
        if regex_found:
          accuracy = int(regex_found[0])
          # prune only ckpt under the same arguments
          if model_match(fname, self.args):
            acc_and_folders.append((accuracy, fname))

      # scores_and_files.sort(key=lambda tup: tup[0], reverse=True)  # largest to smallest
      acc_and_folders.sort(key=lambda tup: tup[0], reverse=True)
      # for _, file in scores_and_files[num_keep:]:
      for _, folder in acc_and_folders[num_keep:]:
        # os.remove(file)
        shutil.rmtree(folder) # for recursive removal
        # print(f'removed {file} due to pruning')
        print(f'removed {folder} due to pruning')

  def update_optimization(self, optimizer, scheduler):
    self.optimizer = optimizer
    self.scheduler = scheduler
