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
  original = generated_string
  ending_tokens = ['<pad>', '<sep>', '</s>', '<|endoftext|>']

  if args.model in ['bart', 't5']:  
    if args.task in ['in_context', 'meta_learn']:
      ending_tokens.extend(['.', ',', ';', '[PAD]'])
    pred_string = generated_string
    if pred_string.startswith('<pad> '):
      # [:6] is to truncate the "<pad>" token.
      pred_string = pred_string[6:]
    if pred_string.startswith('<extra_id_0> '):
      # [13:] is to truncate the "<extra_id_0>" token.
      pred_string = pred_string[13:]

  elif args.model == 'gpt':
    if args.task in ['in_context', 'meta_learn']:
      generated_string = drop_exemplars(generated_string)
    pred_string = parse_gpt(args.prompt_style, args.task, generated_string)

  eos_index = len(pred_string)
  for tok in ending_tokens:
    cand_index = pred_string.find(tok)  # returns -1 if not found
    if 0 < cand_index and cand_index < eos_index:
      eos_index = cand_index

  pred_string = pred_string[:eos_index]
  parsed_str = normalize_text(pred_string)
  return parsed_str

def drop_exemplars(generated_string):
  parts = generated_string.split('<|endoftext|>')
  candidate_part = parts[-1].replace('<pad>', '').replace('<sep>', '')
  if len(candidate_part) > 14:
    current_example = candidate_part  # failed to generate a eos_token
  else:
    current_example = parts[-2]
  return current_example

def parse_gpt(style, task, current_example):
  if task == 'in_context':  # in-context string has no special tokens
    separator, label_sep = ';', ':'
  else:
    separator, label_sep = '<sep>', '<label>'

  try:
    prompt_with_pred = current_example.split(separator)[1]
  except(IndexError):
    prompt_with_pred = current_example

  try:
    if style in ['schema', 'statement', 'naive', 'human']:
      pred_string = prompt_with_pred.split(' is ')[1]
    elif style == 'question':
      pred_string = prompt_with_pred.split('? ')[1]
    elif style in ['none', 'random']:
      pred_string = prompt_with_pred.split(label_sep)[1]
  
  except(IndexError):
    # pred is very likely incorrect, so we feed something just to prevent code from breaking 
    pred_string = prompt_with_pred.split()[-1]
  return pred_string.replace(' <pad>', '').strip()

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
  if s in ['none', '<none>', 'remove', '<remove>']:
    return '<none>'
  s = s.lower().strip()
  s = re_punc.sub(' ', s)
  s = re_art.sub(' ', s)
  parts = s.split()
  s = ' '.join(parts[:6])
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
          clean_pred = GENERAL_TYPO[pred_val] if pred_val in GENERAL_TYPO else pred_val
          clean_target = normalize_text(target_val)
          group_by_ds[f'{domain}_{slot}'] = (clean_pred, clean_target)
        sorted_convos[convo_id].append(group_by_ds)  # in order due to looping by turn_count
  return sorted_convos

def calculate_jga(results, final_preds, verbose):
  """ Return a results dictionary that contains the JGA and accuracy """
  possible, correct = 0, 0
  joint_possible, joint_correct = 0, 0
  errors = Counter()
  normal_dict = {
    "true": "yes",
    "false": "none",
    "any":"<none>",
  }

  for convo_id, filled_turns in final_preds.items():
    
    turn_correct = True
    for dialog_state in filled_turns:
      for domain_slot, turn_data in dialog_state.items():
        pred_val, target_val = turn_data
        if pred_val in normal_dict:
          pred_val = normal_dict[pred_val]
        if target_val in normal_dict:
          target_val = normal_dict[target_val]
        
        if target_val != '<none>':
          # if pred_val == target_val:
          if pred_val.startswith(target_val):
            correct += 1
          else:
            turn_correct = False
            errors[f"{pred_val}-{target_val}"] += 1
          possible += 1

      if turn_correct:
        joint_correct += 1
      joint_possible += 1

  if verbose:
    for error in errors.most_common(10):
      print(error)
  results['accuracy'] = round(float(correct) / possible, 3)
  results['jga'] = round(float(joint_correct) / joint_possible, 3)
  return results

def test_quantify(args, predictions, targets, exp_logger, tokenizer):
  """predictions is a dict with keys of global_ids
    the keys is another dict with keys in 'domain-slot' format
    the value is the parsed, predicted value associated with that 'domain-slot'
  targets is also a dict with keys of global ids
    the values are list of the target labels for that turn
    the labels are dict with keys of domain, slot, value
  the final preds dict should have keys of convo_id (but not critical)
    the values is a list of turns, where each item is a dialog state
    each dialog state is dict with keys of domain_slot and a set of values
    the set is composed of None, predicted label and target label
  """
  results = {'split': 'test'}
  final_preds = defaultdict(list)

  for global_id, preds in predictions.items():
    turn_targets = targets[global_id]
    convo_id, turn_count = global_id.split('_')
    labels = {f"{tt['domain']}-{tt['slot']}": tt['value'] for tt in turn_targets}

    dialog_state = {}
    for domain_slot, pred_val in preds.items():
      target_val = normalize_text(labels[domain_slot])
      if pred_val in GENERAL_TYPO:
        pred_val = GENERAL_TYPO[pred_val]
      if pred_val == 'any':
        pred_val = '<none>'
      if target_val == 'any':
        target_val = '<none>'
      dialog_state[domain_slot] = (pred_val, target_val)
    final_preds[convo_id].append(dialog_state)

  results = calculate_jga(results, final_preds, args.verbose)
  exp_logger.log_info(results)
  return results

def eval_quantify(args, predictions, targets, exp_logger, tokenizer):
  results = {'epoch': exp_logger.epoch, 'chunk': exp_logger.chunk_num}  # 'loss': exp_logger.eval_loss  (no loss by default)

  if args.style == 'dataset':
    # the left out query set is MWOZ or SGD
    grouped_preds = group_by_convo(args, predictions, targets)
    final_preds = sort_by_turn(grouped_preds)
    results = calculate_jga(results, final_preds, args.verbose)

  elif args.style == 'domain':
    # the left out query set is hotel, attraction, taxi, etc.
    pass
  exp_logger.log_info(results)
  return results

def eval_qualify(args, all_outputs, all_targets):
  assert(len(all_outputs) == len(all_targets))
  if args.qualify:
    positions = random.sample(range(len(all_outputs)), 10)  # sample 10 random examples

    for pos in positions:
      output_str, target = all_outputs[pos], all_targets[pos]
      print(output_str)
      print(f'--- Target: {target} ---')

def dst_breakdown(predictions, labels, results):
  label_keys = ['intents', 'requests', 'slots', 'values']

  matches = defaultdict(float)
  poss_match = defaultdict(float)
  founds = defaultdict(float)
  possnormalize_text(_found = defaultdict(float))

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

  # prompt_style = 'none'
  # task = 'meta_learn'
  # size = 'medium'
  # model = 'gpt'
  # result_dir = "/results/"
  # output_name = f"{prompt_style}_five_test.json"
  # result_path = os.path.join('mwoz', task, f'{model}_{size}')


  # with open(os.path.join(result_dir, result_path, output_name)) as df:
  #   output_results = json.load(df)
  # prior_pred_state = output_results['preds']
  # all_targets = output_results['targets']

  # joint_acc, _ = test_results(args,  prior_pred_state, all_targets,)
  # joint_acc, _ = eval_dst(args)
  joint_acc, _ = eval_confidence(args)
  print('joint accuracy: {}%'.format(joint_acc))

