import os, pdb, sys
import numpy as np
import random
import pickle as pkl
from numpy.linalg import norm

from sentence_transformers import SentenceTransformer
from tqdm import tqdm as progress_bar
from assets.static_vars import device

class ExemplarDetective(object):

  def __init__(self, args, data):
    self.search_method = args.search
    self.left_out = args.left_out
    self.num_shots = args.num_shots
    
    if args.task != 'fine_tune':
      self.check_embed_cache(args, data, 'mpnet')  # roberta

  def check_embed_cache(self, args, data, embed_method):
    left_out = 'none' if args.left_out == '' else args.left_out
    ctx_len = args.context_length
    cache_file = f'{embed_method}_{args.style}_{left_out}_lookback{ctx_len}_embeddings.pkl'
    cache_path = os.path.join(args.input_dir, 'cache', args.dataset, cache_file)

    if os.path.exists(cache_path):
      self.candidates = pkl.load( open( cache_path, 'rb' ) )
      print(f"Loaded {len(self.candidates)} embeddings from {cache_path}")
    else:
      self.embed_candidates(data, cache_path, embed_method)

  def embed_candidates(self, data, cache_path, embed_method):
    samples = self._sample_shots(data)
    ckpt_name = 'all-mpnet-base-v2' if embed_method == 'mpnet' else 'all-distilroberta-v1'
    model = SentenceTransformer(f'sentence-transformers/{ckpt_name}')
    self.embed_model = model.to(device)
    print(f'Creating new embeddings with {embed_method} from scratch ...')

    self.candidates = []
    for exp in progress_bar(data, total=len(data)):
      history = ' '.join(exp['utterances'])
      cand = {   # embedding is a 768-dim numpy array
        'embedding': self.embed_model.encode(history),
        'gid': target['global_id'],
        'history': history,
        'dsv': (target['domain'], target['slot'], target['value'])
      }
      self.candidates.append(cand)
    pkl.dump(self.candidates, open(cache_path, 'wb'))

  def _sample_shots(self, data):
    full_size = len(data)

    if self.num_shots == 'full':
      return data
    elif self.num_shots == 'ten':
      num_samples = int(full_size * 0.1)
    elif self.num_shots == 'five':
      num_samples = int(full_size * 0.05)
    elif self.num_shots == 'one':
      num_samples = int(full_size * 0.01)
    elif self.num_shots == 'point':
      num_samples = int(full_size * 0.001)

    samples = np.random.choice(data, size=num_samples, replace=False)
    return samples

  def search(self, example):
    """ returns the closest exemplars from the candidate pool not already chosen"""
    if self.search_method == 'oracle':
      return self.oracle_search(example)
    elif self.search_method == 'tuned':
      pass
    else:
      return self.distance_search(example)

  def reset(self):
    self.selected_gids = []
    self.sorted_exemplars = []

  def oracle_search(self, example):
    self.selected_gids.append(target['global_id'])
    target = example['target']
    acceptable = False

    while not acceptable:
      exemplar = random.choice(self.candidates)
      hist = exemplar['history'].lower()
      domain, slot, value = exemplar['dsv']

      matching_ds = domain == target['domain'] and slot == target['slot']
      new_example = exemplar['gid'] not in self.selected_gids
      not_empty = value.lower() in hist or slot in hist or value == 'any'

      if new_example and matching_ds and not_empty:
        self.selected_gids.append(exemplar['gid'])
        acceptable = True

    return exemplar

  def distance_search(self, example):
    if len(self.sorted_exemplars) == 0:
      history = ' '.join(example['utterances'])
      exp_embed = self.embed_model.encode(history)
      cand_embeds = [cand['embedding'] for cand in self.candidates]

      if self.search_method == 'cosine':
        self.cosine(exp_embed, cand_embeds)
      elif self.search_method == 'euclidean':
        self.euclidean(exp_embed, cand_embeds)
      elif self.search_method == 'mahalanobis':
        self.mahalanobis(exp_embed, cand_embeds)
      # we can never fit more than 60 in-context examples, so we can stop sorting there
      nearest_indexes = np.argpartition(self.distances, 60)[:60]
      
      for near_id in nearest_indexes:
        exemplar = self.candidates[near_id]
        self.sorted_exemplars.append(exemplar)

    acceptable = False
    while not acceptable:
      exemplar = self.sorted_exemplars.pop(0)
      domain, slot, value = exemplar['dsv']
      global_id = exemplar['gid']
      if global_id not in self.selected_gids and value != '<none>':
        self.selected_gids.append(global_id)
        acceptable = True

    return exemplar

  def cosine(self, example, exemplars):
    for exp in exemplars:
      cos_sim = np.dot(example, exp) / (norm(example)*norm(exp))
      self.distances.append(1 - cos_sim)

  def mahalanobis(self, example, exemplars):
    num_exp, hidden_dim = exemplars.shape
    covar = torch.zeros(hidden_dim, hidden_dim)
    for exp in progress_bar(exemplars, total=num_exp, desc="Covariance matrix"):
      diff = (example - exp).expand_dims(axis=1)     # hidden_dim, 1
      covar += np.dot(diff, diff.T)                  # hidden_dim, hidden_dim
    inv_cov_matrix = np.linalg.inv(covar)

    for exp in progress_bar(exemplars, total=num_exp, desc="Calculating distances"):
      difference = example - exp
      left_term = np.dot(difference, inv_cov_matrix)
      score = np.dot(left_term, difference.T)    
      distance = np.sqrt(abs(score))
      self.distances.append(distance)

  #@staticmethod
  def euclidean(self, example, exemplars):
    differences = example - exemplars
    self.distances = np.sqrt(np.sum((differences) ** 2, axis=1))