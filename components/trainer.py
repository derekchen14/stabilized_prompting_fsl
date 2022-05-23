#!/usr/bin/env python3
#
import os, sys, pdb
import numpy as np
import random

from torch import nn, no_grad
from tqdm import tqdm as progress_bar
from components.logger import ExperienceLogger
from components.detector import ExemplarDetective

from utils.help import *
from utils.process import process_data, get_dataloader
from utils.arguments import solicit_params
from utils.load import load_tokenizer, load_model, load_data, load_best_model, load_support
from utils.evaluate import eval_quantify, eval_qualify, test_quantify, parse_output
from assets.static_vars import device, debug_break, STOP_TOKENS
from transformers import (
  CONFIG_MAPPING,
  MODEL_FOR_CAUSAL_LM_MAPPING,
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  HfArgumentParser,
  Trainer,
  TrainingArguments,
  default_data_collator,
  set_seed,
)



class DSTrainer(Trainer):
  """docstring for Trainer"""
  def __init__(
    self,
    model: Union[PreTrainedModel, nn.Module] = None,
    training_args: TrainingArguments = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    args_customize = None,
    dataset = None,
    ):
    super(Trainer, self).__init__(
      model=model,
      args=training_args,
      train_dataset=datasets["train"] if training_args.do_train else None,
      eval_dataset=datasets["test"] if training_args.do_eval else None,
      tokenizer=tokenizer,
      data_collator=default_data_collator,
      is_model_parallel=args.parallel,)
    self.args = args_customize
    self.dataset = dataset

  def get_train_dataloader(self):
      return get_dataloader(self.args, dataset['train'])

  def get_eval_dataloader(self):
      return get_dataloader(self.args, dataset['dev'])

  def get_test_dataloader(self):
      return get_dataloader(self.args, dataset['test'])

  def train(self):
    detective = ExemplarDetective(self.args, self.dataset['train'])
    # dataset, dev_dataset = datasets['train'], datasets['dev']
    # train_dataloader = get_dataloader(self.args, self.dataset)
    train_dataloader = get_train_dataloader()
    total_steps = len(train_dataloader) // self.args.grad_accum_steps * self.args.n_epochs
    optimizer, scheduler = setup_optimization(self.args, self.model, total_steps)
    exp_logger.update_optimization(optimizer, scheduler)
    
    if self.args.task == 'meta_learn':
      self.dataset['train'].add_detective(detective)
      dev_dataset.add_detective(detective)

    for epoch_count in range(exp_logger.num_epochs):
      exp_logger.start_epoch(train_dataloader, self.args.percent)
      self.model.train()

      for step, batch in enumerate(train_dataloader):
        inputs, targets = self.dataset['train'].collate(self.args, batch)
        review_inputs(self.args, inputs, targets, self.dataset['train'].tokenizer)
        outputs = self.model(**inputs, labels=targets)
        exp_logger.tr_loss += outputs.loss.item()
        loss = outputs.loss / self.args.grad_accum_steps
        loss.backward()

        if (step + 1) % self.args.grad_accum_steps == 0:
          nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
          exp_logger.optimizer.step()  # backprop to update the weights
          exp_logger.scheduler.step()  # Update learning rate schedule
          self.model.zero_grad()
          exp_logger.log_train(step)
        if exp_logger.train_stop(self.args, step, debug_break): break

      if self.args.task == 'meta_learn' and self.args.do_leave:
        run_leftout(self.args, self.model, self.dataset['dev'], exp_logger)
      eval_res = run_eval(self.args, self.model, self.dataset['dev'], exp_logger)
      if eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
        exp_logger.best_score = eval_res
        exp_logger.save_best_model(self.model, tokenizer, self.args.prune_keep)
      early_stop = exp_logger.end_epoch()
      if early_stop: break

  return self.model

  def evaluate(self):
    detective = ExemplarDetective(self.args, self.dataset['train'])
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