import os, pdb, sys
import numpy as np
import random

def search_for_similar(example, candidates):
  """
  candidates is a list of examples

  TODO: use TF_IDF / SBERT / Roberta
  embedding = roberta_model(dialog)

  closest = none
  distance = 100
  for cand in candidates:
    current_dist = cosine_distance(embedding, cand)
    if current_dist < distance:
      closest = cand
      distance = current_distance
  return closest
  """
  target = example['target']
  gid = target['global_id']
  domain = target['domain']
  slot = target['slot']

  # this is the oracle version where we cheat by using the target domain and slot
  acceptable = False
  while not acceptable:
    candidate = random.choice(candidates)
    ct = candidate['target']
    cand_history = ' '.join(candidate['utterances'])

    matching_ds = domain == ct['domain'] and slot == ct['slot']
    new_example = gid != ct['global_id']
    not_empty = ct['value'].lower() in cand_history.lower()

    if new_example and matching_ds and not_empty:
      acceptable = True
  
  return candidate
