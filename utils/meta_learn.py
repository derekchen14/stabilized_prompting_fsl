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

  acceptable = False
  while not acceptable:
    cand = random.choice(candidates)
    cand_gid = cand['target']['global_id']
    cand_dom = cand['target']['domain']

    if gid != cand_gid and domain == cand_dom:
      acceptable = True
  
  return cand
