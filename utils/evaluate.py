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

def parse_output(generated_string, task):
  """accepts the full string generated by the model and 
  returns a structured representation for comparison with the target"""
  parsed = defaultdict(list)
  context_str, pred_str = generated_string.split('<label>')
  pred_string = pred_str.replace(' <pad>', '') 

  if '<' in pred_string:  # represents the start of a special token
    eos_index = pred_string.index('<')
  else:
    return parsed  # we found nothing
  pred_string = pred_string[:eos_index].strip()

  if task == 'sgd':
    return parse_sgd(pred_string, parsed)
  elif task == 'tt':
    return parse_tt(pred_string, parsed)
  elif task == 'mwoz20':
    return parse_mwoz(pred_string, parsed)
  elif task == 'mwoz22':
    return pred_string

def parse_sgd(pred_string, parsed):
  for pred in pred_string.split(';'):
    try:
      remaining, value = pred[:-1].split("=")
      intent, slot = remaining.split("(")
    except(ValueError):
      continue

    if slot == 'request':
      parsed['requests'].append(value)
    else:
      parsed['slots'].append(slot)
      parsed['values'].append(value)
    parsed['intents'].append(intent)

  return parsed

def parse_mwoz(pred_string, parsed):
  domain_preds = pred_string.split("<sep>") # and the

  for dpred in domain_preds:
    dpred = dpred.strip()
    try:
      position = dpred.index(":")  # state
      domain = dpred[:position].strip()
      prediction = dpred[position+1:]
    except(ValueError):
      continue

    for pred in prediction.strip().split(';'):
      for swap in swaps:
        if swap in pred:
          replacement = swaps[swap]
          pred = pred.replace(swap, replacement)

      parts = pred.split()
      if len(parts) < 2:
        continue
      elif len(parts) == 2:
        slot, val = parts
      else:
        slot = parts[0]
        val = ' '.join(parts[1:])

      if val != 'none':
        parsed[domain].append((slot, val))

  return parsed

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

def group_by_convo(predictions, extras, model_type):
  convos = defaultdict(dict)
  for pred, extra in zip(predictions, extras):
    if model_type == 'gpt':
      pred_value = parse_output(pred, task='mwoz22')
    elif model_type in ['bart', 't5']:
      pred_value = pred.strip()

    if isinstance(pred_value, str) and len(pred_value) > 0 and pred_value != "none":
      convo_id, turn_count = extra['convo_id'], extra['turn_count']
      turn_tuple = (pred_value, extra['dsv'], extra['active_domains'])
      if turn_count not in convos[convo_id]:
        convos[convo_id][turn_count] = []
      convos[convo_id][turn_count].append(turn_tuple)
  return convos

def pred_to_dialog_state(grouped_preds):
  generated_convos = {}
  for convo_id, convo_turns in grouped_preds.items():
    # convo turns is a dictionary with keys of turn_count and values of turn_tuples
    max_turn = max(list(convo_turns.keys()))   # largest turn count in the convo
    generated_turns = order_by_turn_group_by_ds(convo_turns, max_turn+1)
    generated_convos[convo_id] = generated_turns
  return generated_convos

def order_by_turn_group_by_ds(convo_turns, max_turns):
  generated_turns = []

  for turn_index in range(max_turns):
    if turn_index in convo_turns:
      turn_tuples = convo_turns[turn_index]
      gen_turn = {'state': defaultdict(dict), 'active_domains': turn_tuples[0][2]}
      
      # Group by domain
      response_tokens = []
      for pred_value, dsv, _ in turn_tuples:
        domain, slot, target_value = dsv
        gen_turn['state'][domain][slot] = pred_value
        response_tokens.extend([domain, slot, target_value])
      gen_turn['response'] = ' '.join(response_tokens)  # filler to satisfy evaluator
      generated_turns.append(gen_turn)  # in order due to looping by turn_count
    
  return generated_turns

def calculate_mwoz_2_2(predictions, extras, model_type):
  # Group by conversation and turn
  grouped_preds = group_by_convo(predictions, extras, model_type)
  # Generated_convos must be a dictionary where the key is the convo_id
  # the value of the prediction should be a list of dicts, where each dict is a dialog state
  generated_convos = pred_to_dialog_state(grouped_preds)
  # Create evaluator and run evalation
  # evaluator = Evaluator(bleu=False, success=False, richness=False, dst=True)
  # return evaluator.evaluate(generated_convos)

def eval_quantify(args, predictions, targets, exp_logger, tokenizer, split):
  results = {'epoch': exp_logger.epoch }  # 'loss': exp_logger.eval_loss  (no loss by default)

  if args.dataset == 'mwoz22':
    mwoz_res = calculate_mwoz_2_2(predictions, targets, args.model)
    results['slot_f1'] = round(mwoz_res['dst']['slot_f1'], 3)
    results['precision'] = round(mwoz_res['dst']['slot_precision'], 3)
    results['recall'] = round(mwoz_res['dst']['slot_recall'], 3)
    results['accuracy'] = round(mwoz_res['dst']['joint_accuracy'], 3)

  exp_logger.log_info(results)
  return results

def clean_text(input_text, tokenizer):
  input_text = input_text.replace(tokenizer.pad_token, '')
  input_text = input_text.replace(tokenizer.bos_token, '')
  input_text = input_text.replace(tokenizer.eos_token, '')
  cleaned = input_text.split(" <")
  return cleaned

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

