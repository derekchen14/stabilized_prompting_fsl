import os, pdb, sys
import numpy as np
import random

def search_similar_context(dialog, candidates, target):
  # use TF_IDF / SBERT / Roberta
  similar = random.choice(candidates)
  return similar