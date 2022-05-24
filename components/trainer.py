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
  def train(self, exp_logger, detective):
    # dataset, dev_dataset = datasets['train'], datasets['dev']
    train_dataloader = get_dataloader(self.args, self.train_dataset)
    # train_dataloader = get_train_dataloader()
    total_steps = len(train_dataloader) // self.args.grad_accum_steps * self.args.n_epochs


    if self.args.deepspeed:
      deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
          self, num_training_steps=total_steps, resume_from_checkpoint=False
      )
      self.model = deepspeed_engine.module
      self.model_wrapped = deepspeed_engine
      self.deepspeed = deepspeed_engine
      self.optimizer = optimizer
      self.lr_scheduler = lr_scheduler
    else:
      self.optimizer, self.scheduler = setup_optimization(self.args, self.model, total_steps)

    exp_logger.update_optimization(self.optimizer, self.scheduler)
    
    if self.args.task == 'meta_learn':
      self.train_dataset.add_detective(detective)
      self.eval_dataset.add_detective(detective)

    for epoch_count in range(exp_logger.num_epochs):
      exp_logger.start_epoch(train_dataloader, self.args.percent)
      self.model.train()

      for step, batch in enumerate(train_dataloader):
        inputs, targets = self.dataset['train'].collate(self.args, batch)
        review_inputs(self.args, inputs, targets, self.tokenizer)

        if self.deepspeed:
          kwargs = dict(device=self.args.device)
          kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
          inputs =  inputs.to(**kwargs)
          targets = targets.to(**kwargs)

        outputs = self.model(**inputs, labels=targets)
        exp_logger.tr_loss += outputs.loss.item()
        loss = outputs.loss / self.args.grad_accum_steps
        if self.deepspeed:
          loss = self.deepspeed.backward(loss)
        else:
          loss.backward()

        if (step + 1) % self.args.grad_accum_steps == 0:
          nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
          if self.deepspeed:
            self.deepspeed.step()
          else:
            exp_logger.optimizer.step()  # backprop to update the weights
            exp_logger.scheduler.step()  # Update learning rate schedule
          self.model.zero_grad()
          exp_logger.log_train(step)
        if exp_logger.train_stop(self.args, step, debug_break): break

      if self.args.task == 'meta_learn' and self.args.do_leave:
        run_leftout(self.args, self.model, self.eval_dataset, exp_logger)
      eval_res = self.run_eval(self.args, self.model, self.dataset['dev'], exp_logger)
      if eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
        exp_logger.best_score = eval_res
        exp_logger.save_best_model(self.model, self.tokenizer, self.args.prune_keep)
      early_stop = exp_logger.end_epoch()
      if early_stop: break

    return self.model


  def run_eval(self, args, model, dataset, exp_logger):
    tokenizer = dataset.tokenizer
    dataloader = get_dataloader(args, dataset, 'dev')
    num_batches = debug_break if args.debug else len(dataloader)
    exp_logger.start_eval(num_batches, args.eval_interval)
    all_outputs, all_targets = [], []
    
    ''' goes through model generation without backprop, rather than classification '''
    for batch in progress_bar(dataloader, total=len(dataloader)):
      inputs, target_dict = dataset.collate(args, batch)
      all_targets.extend(target_dict)   # notice this is "extend", not "append"

      maxl = inputs['input_ids'].shape[1] + 12
      with no_grad():
        # defaults to greedy sampling, for param details see https://huggingface.co/docs/transformers/
        #        v4.15.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate 
        outputs = model.generate(**inputs, max_length=maxl, early_stopping=True,
                            repetition_penalty=args.threshold, temperature=args.temperature)
      output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)
      all_outputs.extend(output_strings)

      if exp_logger.log_eval(args.qualify, output_strings, target_dict):
        results = eval_quantify(args, all_outputs, all_targets, exp_logger, tokenizer)
      if args.debug and exp_logger.eval_step >= debug_break: break

    return results

  def predict(self, exp_logger, detective):
    ontology, tokenizer = exp_logger.ontology, self.dataset['test'].tokenizer
    self.dataset['test'].add_detective(detective)
    exp_logger.start_eval(len(self.dataset['test']), self.args.eval_interval)
    
    if self.args.task in ['meta_learn', 'fine_tune']:
      model = load_best_model(self.args, exp_logger, tokenizer)
    else:
      model = load_model(self.args, ontology, tokenizer, exp_logger.save_path)

    all_targets = defaultdict(list)
    prior_pred_state = defaultdict(dict)
    for conversation in progress_bar(self.dataset['test'].data, total=len(self.dataset['test'])):
      for global_id, turn in conversation.items():
        # turn is a list of examples

        batches = batchify(self.args, turn, global_id, prior_pred_state)
        for batch in batches:
          inputs, target_dict = self.dataset['test'].collate(self.args, batch)
          review_inputs(self.args, inputs, inputs['input_ids'], tokenizer)
          all_targets[global_id].extend(target_dict) #  all the target labels for this turn 

          if self.args.task == 'in_context':
            maxl = 2048 if self.args.size == 'large' else 1024
          else:
            maxl = inputs['input_ids'].shape[1] + 12

          with no_grad():
            outputs = model.generate(**inputs, max_length=maxl, repetition_penalty=self.args.threshold,
                                                early_stopping=True, temperature=self.args.temperature, 
                                                forced_eos_token_id=tokenizer.eos_token_id)
          output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=False)
         
          for target, output_str in zip(target_dict, output_strings):
            state_key = f"{target['domain']}-{target['slot']}"
            pred_value = parse_output(self.args, output_str)
            prior_pred_state[global_id][state_key] = pred_value
      if exp_logger.log_eval(self.args.qualify, output_strings, target_dict):
        results = test_quantify(self.args, prior_pred_state, all_targets, exp_logger, tokenizer)
        self.dataset['test'].detective.report(self.args.verbose, self.args.task)
    
    if self.args.do_save:
      output_name = f'{self.args.prompt_style}_lr{self.args.learning_rate}_clen{self.args.context_length}.json'
      json.dump(results, open(os.path.join(save_path, output_name), 'w'), indent=2)