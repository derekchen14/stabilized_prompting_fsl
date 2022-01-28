import os, pdb, sys
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm as progress_bar

class Embedder(object):
  def __init__(self, full_path, size):
    self.path = full_path
    self.embeddings = {}

    with open(full_path, 'r') as file:
      for line in progress_bar(file, total=400000):
        row = line.split()
        token = row[0]
        self.embeddings[token] = np.array( [float(x) for x in row[1:]] )

    unk = np.empty(size)
    for key in ["unk", "unknown", "missing", "none", "error", "na", "empty"]:
      unk += self.embeddings[key]
    self.unk = unk / 7

  def _vec(self, token):
    try:
      return self.embeddings[token]
    except KeyError:
      return self.unk

  def embed_word(self, word):
    return self._vec(word)

  def embed_sentence(self, sentence):
    if len(sentence.strip()) == 0:
      return self.unk

    embs = [self._vec(word) for word in word_tokenize(sentence)]
    avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
    assert not np.any(np.isnan(avg_emb))
    return avg_emb
