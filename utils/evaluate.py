import os, pdb, sys
import re
import json
import torch
import numpy as np
import pandas as pd
import random

from torch import nonzero
from numpy.linalg import norm
# from lexical_diversity import lex_div
from tqdm import tqdm as progress_bar
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score
from assets.static_vars import device, debug_break
# metric = load_metric('bleu')  'bertscore', ''  
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from mwzeval.metrics import Evaluator

def parse_output(args, generated_string):
  """accepts the full string generated by the model and 
  returns a parsed string for comparison with the target"""
  if args.model in ['bart', 't5']:
    pred_string = generated_string[7:]
  elif args.model == 'gpt':
    pred_string = parse_gpt(args.prompt_style, generated_string)

  eos_index = len(pred_string)
  for tok in ['<pad>', '<sep>', '</s>', '<|endoftext|>']:
    cand_index = pred_string.find(tok)  # returns -1 if not found
    if 0 < cand_index and cand_index < eos_index:
      eos_index = cand_index

  pred_string = pred_string[:eos_index]
  parsed_str = normalize_text(pred_string)

  if args.verbose and random.random() < 0.01: 
    print(generated_string[10:110])
    print(parsed_str)
  return parsed_str 

def parse_gpt(style, generated_string):
  if style in ['schema', 'question', 'informed', 'naive', 'human']:
    prompt_with_pred = generated_string.split('<sep>')[1]
    pred_string = prompt_with_pred.split(' is ')[1]
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

def group_by_convo(args, predictions, targets):
  """groups all predictions by conversation, parses the output, and then sorts by turn
  predictions: list of the raw string out from the model
  targets: list of targets extracted from examples
  """
  convos = defaultdict(dict)
  for pred, target in zip(predictions, targets):
    convo_id, turn_string = target['global_id'].split('_')
    turn_count = int(turn_string)
    parsed = parse_output(args, pred)
    val = normalize_text(target['value'])
    turn_tuple = (parsed, target['domain'], target['slot'], val)
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

def fill_carryover(conversations):
  """ Automatically carry over slots when the current estimate is none"""
  filled = defaultdict(list) 
  for convo_id, turns in conversations.items():

    carry = {}
    for turn_data in turns:
      dialog_state = {}
      for domain_slot, turn_data in turn_data.items():
        pred_val, target_val = turn_data
        if pred_val == '<none>' and domain_slot in carry:
          pred_val = carry[domain_slot] # then carry over the old value
        elif pred_val == '<remove>':
          pred_val = '<none>'

        dialog_state[domain_slot] = (pred_val, target_val)
        carry[domain_slot] = pred_val  # store it for the next round
      
      filled[convo_id].append(dialog_state)
  return filled

def calculate_jga(results, final_preds):
  """ Does not currently calculate JGA, just gives a sketch 
  should return a results dictionary that contains the JGA and anything else you want to log
  """
  possible, correct = 0, 0
  joint_possible, joint_correct = 0, 0

  for convo_id, filled_turns in final_preds.items():
    
    turn_correct = True
    for dialog_state in filled_turns:
      for domain_slot, (pred_val, target_val) in dialog_state.items():
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

def eval_quantify(args, predictions, targets, exp_logger, tokenizer, split):
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

def eval_qualify(args, predictions, targets, contexts, exp_logger, tokenizer):
  preds = torch.argmax(predictions, axis=1) 
  
  results = []
  for pred, target, context in zip(preds, targets, contexts):
    input_text = tokenizer.decode(context) # , skip_special_tokens=True)
    cleaned_text = clean_text(input_text, tokenizer)
    target_text = exp_logger.ontology[target]
    pred_text = exp_logger.ontology[pred]

    if pred_text != target_text:
      if args.do_save:
        res = ' '.join(cleaned_text) + '---' + pred_text + '---' + target_text + '\n'
        results.append(res)
      else:
        for line in cleaned_text:
          print(line)
        print('predicted:', pred_text, ', actual:', target_text)
        pdb.set_trace()

  if args.do_save:
    save_filepath = os.path.join(exp_logger.save_path, 'qualify.txt')
    with open(save_filepath, 'w') as file:
      file.writelines(results)
    print(len(results), "results written to", save_filepath)
  return results

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

