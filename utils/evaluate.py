import os, pdb, sys
import re
import json
import torch
import numpy as np
import pandas as pd
import random

from torch import nonzero
from numpy.linalg import norm
from lexical_diversity import lex_div
from tqdm import tqdm as progress_bar
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score
from assets.static_vars import device, debug_break
# metric = load_metric('bleu')  'bertscore', ''  
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(hypotheses, references, smoothing, word_tokenize):
    score = 0
    for hyp, ref in zip(hypotheses, references):
      ref_tokens = word_tokenize(ref)
      hyp_tokens = word_tokenize(hyp)  
      # wrap with a list since technically, you can have many references
      s = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
      score += s
    avg_bleu = float(score) / len(hypotheses)
    return avg_bleu

def calculate_bow(embedder, hypotheses, references):
  """ hypotheses is a list of sentences with batch_size len
  references is a list of lists with batch_size len, the inner list contains list of sentences
  returns the bag-of-words embedding score, taken by averaging all cosine distances """
  cos_sims = []

  for hyp, refsource in zip(hypotheses, references):
    hyp_emb = embedder.embed_sentence(hyp)

    max_similarity = 0
    for ref in refsource:
      ref_emb = embedder.embed_sentence(ref)
      cosine_similarity = np.inner(hyp_emb, ref_emb)/(norm(hyp_emb) * norm(ref_emb))
      if cosine_similarity > max_similarity:
        max_similarity = cosine_similarity        
    cos_sims.append(max_similarity)

  return round(np.mean(cos_sims), 3)

def parse_pred_output(full_string, label_keys):
  # parse the predicted output of the model into a structured representation
  parsed = defaultdict(list)

  pred_string = full_string.split('<label>')[1]   # first half is the context string
  pred_string = pred_string.replace(' <pad>', '') 

  if '<' in pred_string:  # represents the start of a special token
    eos_index = pred_string.index('<')
  else:
    return parsed  # we found nothing

  pred_string = pred_string[:eos_index].strip()
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

def calculate_ranking(predictions, labels):
  results = {}

  for rank in [1,5,10]:
    level = -rank   # select the top 5 rather than bottom 5
    num_correct, num_possible = 0, 0
    # vectorized version possible, but a lot less readable
    for pred, label in zip(predictions, labels):
      top_k_indexes = np.argpartition(pred, kth=level)[level:]
      if label in top_k_indexes:
        num_correct += 1
      if label >= 0:    # -1 means the turn was take-action or end-of-convo
        num_possible += 1

    rank_name = f'recall@{rank}'
    results[rank_name] = round(num_correct / num_possible, 3)

  return results

def eval_quantify(args, predictions, targets, exp_logger, tokenizer, split):
  results = {'epoch': exp_logger.epoch, 'loss': round(exp_logger.eval_loss, 3) }

  if args.task == 'rg': # response generation
    hypotheses, references = [], []
    for pred_batch, targ_batch in zip(predictions, targets):
      for pred, target in zip(pred_batch, targ_batch):
        pred_ids = torch.argmax(pred, axis=1)
        hyp = tokenizer.decode(pred_ids, skip_special_tokens=True)
        hypotheses.append(hyp)
        ref = tokenizer.decode(target, skip_special_tokens=True)
        references.append([ref])
    results['bow_similarity'] = calculate_bow(exp_logger.embedder, hypotheses, references)

  elif args.task == 'ir':  # information retrival
    rank_res = calculate_ranking(predictions, targets)
    for rr_key, rr_val in rank_res.items():
        results[rr_key] = rr_val

  elif args.task == 'dst': # dialogue state tracking
    all_inputs, all_outputs = predictions
    precision, recall = calculate_prec_rec(all_outputs, targets)
    results['precision'] = precision
    results['recall'] = recall
    raw_f1 = (2 * precision * recall) / (precision + recall + 1e-7)
    results['f1_score'] = round(raw_f1, 3)

  else: # classification
    preds = torch.argmax(predictions, axis=1)
    correct = torch.sum(preds == targets)
    total = len(targets)
    acc = correct.item() / float(total)
    results['accuracy'] = round(acc, 4)
  
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

