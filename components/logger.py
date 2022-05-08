import os, pdb, sys
import torch
import random
import logging
import re
import glob
import shutil

import pandas as pd
import numpy as np
import time as tm
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

    self.best_score = { 'epoch': 1 }
    self.metric = 'jga'
    self.best_score[self.metric] = 0
    self.do_save = args.do_save
    self.differences = []
    self.past_metrics = []

    self.logging_loss = 0.0
    self.tr_loss = 0.0
    self.eval_loss = 0.0

  def log_info(self, text):
    self.logger.info(text)

  def start_train(self, total_step):
    self.logger.info("***** Running training *****")
    self.logger.info("  Num Epochs = %d" % self.args.n_epochs)
    self.logger.info("  Total train batch size  = %d" % self.args.batch_size)
    self.logger.info("  Total optimization steps = %d" % total_step)
    self.logger.info("  Running experiment for {}".format(self.style))

  def start_epoch(self, dataloader):
    self.logger.info(f"Starting epoch {self.epoch} of {self.num_epochs}")
    self.start_time = tm.time()
    self.num_steps = len(dataloader)

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

  def log_eval(self, qualify, output_strings, target_dicts):
    self.eval_loss = 0  # no loss, since inference only
    self.eval_step += 1

    is_done = self.eval_step == self.final_step
    is_checkpoint = self.eval_step % self.interval_checkpoint == 0 

    if qualify and is_checkpoint:
      for out_str, target in zip(output_strings, target_dicts):
        print(out_str.replace("<pad>",""))
        print(target['value'])

    return is_done or is_checkpoint

  def log_train(self, step, train_metric=''):
    self.global_step += 1

    if self.global_step < 100 and self.global_step % 10 == 0:
      print(self.global_step)
    if self.global_step < 1000 and self.global_step % 100 == 0:
      if self.log_interval <= 500:
        print(self.global_step)

    if self.global_step % self.log_interval == 0:
      current_loss = (self.tr_loss - self.logging_loss) / self.log_interval
      self.logging_loss = self.tr_loss

      step_report = f'[{step+1}/{self.num_steps}] '
      loss_report = 'Mean_loss: %.3f, ' % current_loss
      metric_report = f'{self.metric}: {train_metric}'
      # print(step_report + loss_report + metric_report)
      print(step_report + loss_report)

  def save_best_model(self, model, tokenizer, prune_keep):
    if self.do_save and self.best_score[self.metric] > 0.1:
      learning_rate = str(self.args.learning_rate)
      accuracy = str(self.best_score[self.metric] * 10000)[:3]
      style = self.args.prompt_style
      context_length = self.args.context_length
      ckpt_name = f'{style}_epoch{self.epoch}_lr{learning_rate}_clen{context_length}_acc{accuracy}.pt'
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
