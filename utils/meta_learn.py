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
    cand = random.choice(candidates)
    cand_gid = cand['target']['global_id']
    cand_dom = cand['target']['domain']
    cand_slot = cand['target']['slot']

    if gid != cand_gid and domain == cand_dom and slot == cand_slot:
      acceptable = True
  
  return cand
