import os, pdb, sys
import re
import json
import torch
import numpy as np
import pandas as pd
import random

from numpy.linalg import norm
from tqdm import tqdm as progress_bar
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score
from assets.static_vars import device, debug_break, GENERAL_TYPO
# metric = load_metric('bleu')  'bertscore', ''  

def parse_output(args, generated_string):
  """accepts the full string generated by the model and 
  returns a parsed string for comparison with the target"""
  if args.model in ['bart', 't5']:
    # [:7] is to truncate the "<agent>" token. Might not be needed for us
    pred_string = generated_string[7:]
  elif args.model == 'gpt':
    if args.task == 'in_context':
      return parse_in_context(generated_string)
    pred_string = parse_gpt(args.prompt_style, generated_string)

  eos_index = len(pred_string)
  for tok in ['<pad>', '<sep>', '</s>', '<|endoftext|>']:
    cand_index = pred_string.find(tok)  # returns -1 if not found
    if 0 < cand_index and cand_index < eos_index:
      eos_index = cand_index

  pred_string = pred_string[:eos_index]
  parsed_str = normalize_text(pred_string)
  return parsed_str

def parse_in_context(generated_string):
  """ unlike a typical parse output, the in context string has no special tokens """
  parts = generated_string.split('<|endoftext|>')
  if len(parts[-1]) > 14:
    current_example = parts[-1]  # failed to generate a eos_token
  else:
    current_example = parts[-2]

  try:
    prompt_with_pred = current_example.split(';')[1]
  except(IndexError):
    prompt_with_pred = current_example

  try:
    pred_string = prompt_with_pred.split(' is ')[1]
  except(IndexError):
    pred_string = prompt_with_pred.split()[-1]
  
  parsed_str = normalize_text(pred_string)
  return parsed_str

def parse_gpt(style, generated_string):
  if style in ['schema', 'statement', 'naive', 'human']:
    prompt_with_pred = generated_string.split('<sep>')[1]
    pred_string = prompt_with_pred.split(' is')[1]
  elif style == 'question':
    prompt_with_pred = generated_string.split('<sep>')[1]
    pred_string = prompt_with_pred.split('? ')[1]
  elif style in ['none', 'random']:
    pred_string = generated_string.split('<label>')[1]
  return pred_string.strip().replace(' <pad>', '')

def calculate_prec_rec(predicted_outputs, labels):
  label_keys = ['intents', 'requests', 'slots', 'values']  # services is part of input
  match, possible_matches = 0, 0.001
  found, possible_found = 0, 0.001

  for pred_string, label_dict in zip(predicted_outputs, labels):
    parsed = parse_pred_output(pred_string, label_keys)

    for key in label_keys:
      predicted_items = parsed[key]
      target_items = label_dict[key]

      for pi in predicted_items:
        if pi in target_items:
          match += 1
        possible_matches += 1

      for ti in target_items:
        if ti in predicted_items:
          found += 1
        possible_found += 1

  precision = round(match / possible_matches, 3)
  recall = round(found / possible_found, 3)
  return precision, recall


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;=?@\[\]\\^`{|}~_\']')

def normalize_text(s):
  # Lower text and remove punctuation, articles and extra whitespace.
  s = s.lower().strip()
  s = re_punc.sub(' ', s)
  s = re_art.sub(' ', s)
  s = ' '.join(s.split())
  return s

def group_by_convo(args, predictions, targets, use_history=False):
  """groups all predictions by conversation, parses the output, and then sorts by turn
  predictions: list of the raw string out from the model
  targets: list of targets extracted from examples
  """
  convos = defaultdict(dict)
  for pred, target in zip(predictions, targets):
    convo_id, turn_string = target['global_id'].split('_')
    turn_count = int(turn_string)
    parsed = parse_history(args, pred) if use_history else parse_output(args, pred)
    turn_tuple = (parsed, target['domain'], target['slot'], target['value'])
    if turn_count not in convos[convo_id]:
      convos[convo_id][turn_count] = []
    convos[convo_id][turn_count].append(turn_tuple)

  return convos

def sort_by_turn(conversations):
  """ returns sorted conversations which is 
   a dict {key is convo_id : value is a list of turns in order }
   each turn has key of slot, and value is a tuple of (pred_val, target_val)
  """
  sorted_convos = defaultdict(list) 
  for convo_id, turns in conversations.items():
    max_turns = max(list(turns.keys())) + 1  # largest turn count in the convo
    
    for turn_index in range(max_turns):
      if turn_index in turns:
        turn_data = turns[turn_index]
        group_by_ds = {}
        for pred_val, domain, slot, target_val in turn_data:
          group_by_ds[f'{domain}_{slot}'] = (pred_val, target_val)
        sorted_convos[convo_id].append(group_by_ds)  # in order due to looping by turn_count
  return sorted_convos

def fill_carryover(conversations, use_history=False):
  """ Automatically carry over slots when the current estimate is none"""
  filled = defaultdict(list) 
  for convo_id, turns in conversations.items():

    carry = {}
    for turn_data in turns:
      dialog_state = {}
      for domain_slot, turn_data in turn_data.items():
        if use_history:
          package, target_val = turn_data
          history, pred_val = package
        else:
          pred_val, target_val = turn_data
          history = ''

        target_val = normalize_text(target_val)
        if pred_val in GENERAL_TYPO:
          pred_val = GENERAL_TYPO[pred_val]
        if pred_val == '<none>' and domain_slot in carry:
          pred_val = carry[domain_slot] # then carry over the old value
        elif pred_val == '<remove>':
          pred_val = '<none>'

        dialog_state[domain_slot] = (history, pred_val, target_val)
        carry[domain_slot] = pred_val  # store it for the next round
      
      filled[convo_id].append(dialog_state)
  return filled

def parse_history(args, generated_string):
  """parses the output for evaluation , only works with GPT"""
  """
  TODO: cannot run, needs to be fixed
  """
  history, pred_string = generated_string.split('<label>')
  pred_string.replace('<pad>', '').strip()
  
  eos_index = len(pred_string)
  for tok in ['<pad>', '<sep>', '<|endoftext|>']:
    cand_index = pred_string.find(tok)  # returns -1 if not found
    if 0 < cand_index and cand_index < eos_index:
      eos_index = cand_index
  pred_string = pred_string[:eos_index]
  parsed_str = normalize_text(pred_string)
  return history[10:], parsed_str

def calculate_jga(results, final_preds):
  """ Does not currently calculate JGA, just gives a sketch 
  should return a results dictionary that contains the JGA and anything else you want to log
  """
  possible, correct = 0, 0
  joint_possible, joint_correct = 0, 0

  for convo_id, filled_turns in final_preds.items():
    
    turn_correct = True
    for dialog_state in filled_turns:
      for domain_slot, turn_data in dialog_state.items():
        _, pred_val, target_val = turn_data
        if target_val != '<none>':
          if pred_val == target_val:
            correct += 1
          else:
            turn_correct = False
          possible += 1

      if turn_correct:
        joint_correct += 1
      joint_possible += 1

  results['accuracy'] = round(float(correct) / possible, 3)
  results['jga'] = round(float(joint_correct) / joint_possible, 3)
  return results

def eval_quantify(args, predictions, targets, exp_logger, tokenizer):
  results = {'epoch': exp_logger.epoch }  # 'loss': exp_logger.eval_loss  (no loss by default)

  if args.style == 'dataset':
    # the left out query set is MWOZ or SGD
    grouped_preds = group_by_convo(args, predictions, targets)
    sorted_preds = sort_by_turn(grouped_preds)
    final_preds = fill_carryover(sorted_preds)
    results = calculate_jga(results, final_preds)

  elif args.style == 'domain':
    # the left out query set is hotel, attraction, taxi, etc.
    pass
  exp_logger.log_info(results)
  return results

def eval_qualify(args, predictions, targets, exp_logger):
  grouped_preds = group_by_convo(args, predictions, targets, use_history=True)
  sorted_preds = sort_by_turn(grouped_preds)
  final_preds = fill_carryover(sorted_preds, use_history=True)

  errors = Counter()
  for convo_id, filled_turns in final_preds.items():
    for dialog_state in filled_turns:
      for domain_slot, turn_data in dialog_state.items():
        history, pred_val, target_val = turn_data
        if target_val != '<none>' and pred_val != target_val:

          if random.random() < 0.005 and args.verbose:
            print(history)
            print(f'{domain_slot}, pred: {pred_val}, actual: {target_val}')
          val_key = f"{domain_slot}|{pred_val}|{target_val}"
          errors[val_key] += 1

  for error, count in errors.most_common(10):
    print(error, count)

  if args.do_save:
    save_filepath = os.path.join(exp_logger.save_path, 'qualify.txt')
    with open(save_filepath, 'w') as file:
      for error, count in errors.items():
        file.writeline((error, count))
    print("results written to", save_filepath)
  return errors

def dst_breakdown(predictions, labels, results):
  label_keys = ['intents', 'requests', 'slots', 'values']

  matches = defaultdict(float)
  poss_match = defaultdict(float)
  founds = defaultdict(float)
  poss_found = defaultdict(float)

  for prediction, label_dict in zip(predictions, labels):
    parsed = parse_pred_output(prediction, label_keys)

    for key in label_keys:
      predicted_items = parsed[key]
      target_items = label_dict[key]

      for pi in predicted_items:
        if pi in target_items:
          matches[key] += 1
        poss_match[key] += 1

      for ti in target_items:
        if ti in predicted_items:
          founds[key] += 1
        poss_found[key] += 1

      # to avoid divide by zero
      matches[key] += 0.001
      poss_match[key] += 0.001
      founds[key] += 0.001
      poss_found[key] += 0.001

  aggregate_match, aggregate_poss_match = 0, 0  
  aggregate_found, aggregate_poss_found = 0, 0

  for lk in label_keys:
    match = matches[lk]
    possible_matches = poss_match[lk]
    found = founds[lk]
    possible_found = poss_found[lk]

    aggregate_match += match
    aggregate_poss_match += possible_matches
    aggregate_found += found
    aggregate_poss_found += possible_found

    result[f'{lk}_prec'] = round(match / possible_matches, 3)
    result[f'{lk}_rec'] = round(found / possible_found, 3)

  results['precision'] = round(aggregate_match / aggregate_poss_match, 3)
  results['recall'] = round(aggregate_found / aggregate_poss_found, 3)
  return results

if __name__ == '__main__':
  args = solicit_params()
  args.multiwoz_version = '2.1'
  args.use_action = True
  args.use_knowledge = True

  random.seed(14)
  np.random.seed(14)
  # joint_acc, _ = eval_dst(args)
  joint_acc, _ = eval_confidence(args)
  print('joint accuracy: {}%'.format(joint_acc))

