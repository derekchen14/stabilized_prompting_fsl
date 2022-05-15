import os, pdb, sys
import numpy as np
import random
import pickle as pkl
from numpy.linalg import norm

from tqdm import tqdm as progress_bar
from assets.static_vars import device, DATASETS
from utils.load import load_sent_transformer
from collections import defaultdict

class ExemplarDetective(object):

  def __init__(self, args, data):
    self.search_method = args.search
    self.left_out = args.left_out
    self.num_shots = args.num_shots
    self.candidates = defaultdict(list)

    self.regular_loops = 0
    self.missed_loops = 0
    self.num_exemplars = []

    if args.task == 'in_context':
      corpus = args.dataset
      self.check_embed_cache(args, data, corpus, 'mpnet')  # roberta
    elif args.task == 'meta_learn':
      for corpus, full_name in DATASETS.items():
        self.check_embed_cache(args, data, corpus, 'mpnet')

  def check_embed_cache(self, args, data, corpus, embed_method):
    ctx_len = args.context_length
    cache_file = f'{embed_method}_{args.style}_{corpus}_lookback{ctx_len}_embeddings.pkl'
    cache_path = os.path.join(args.input_dir, 'cache', args.dataset, cache_file)
    self.embed_model = load_sent_transformer(args, embed_method)

    if os.path.exists(cache_path) and not args.ignore_cache:
      self.candidates[corpus] = pkl.load( open( cache_path, 'rb' ) )
      num_cands = len(self.candidates[corpus])
      print(f"Loaded {num_cands} embeddings from {cache_path}")
    else:
      samples = self._sample_shots(data)
      print(f'Creating new embeddings with {embed_method} from scratch ...')
      self.embed_candidates(args, samples, cache_path, corpus)

  def embed_candidates(self, args, samples, cache_path, corpus):
    histories = [' '.join(exp['utterances']) for exp in samples]
    embeddings = self.embed_model.encode(histories)

    for emb, exp, hist in zip(embeddings, samples, histories):
      target = exp['target']
      cand = {   # embedding is a 768-dim numpy array
        'embedding': emb,
        'gid': target['global_id'],
        'history': hist,
        'dsv': (target['domain'], target['slot'], target['value']),
        'prev_state': exp['prev_state'],
      }
      self.candidates[corpus].append(cand)
    pkl.dump(self.candidates[corpus], open(cache_path, 'wb'))

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
    corpus = example['corpus']
    if self.search_method == 'oracle' or use_oracle:
      return self.oracle_search(example, corpus)
    else:
      return self.distance_search(example, corpus)

  def reset(self):
    self.selected_gids = []
    self.distances = []
    self.sorted_exemplars = []

  def report(self, verbose, task):
    if task == 'in_context':
      avg_context_size = round(np.mean(self.num_exemplars), 2)
      print(f"Found an average of {avg_context_size} exemplars for each input example")

      if verbose:
        # total loops should equal number of exemplars chosen
        regular = self.regular_loops
        total_loops = self.missed_loops + self.regular_loops
        rate = round((regular / total_loops) * 100, 2)
        print(f"Loop success rate is {regular} out of {total_loops} exemplars ({rate}%)")

  def oracle_search(self, example, corpus):
    target = example['target']
    self.selected_gids.append(target['global_id'])

    acceptable = False
    loops = 0

    while not acceptable:
      exemplar = random.choice(self.candidates[corpus])
      hist = exemplar['history'].lower()
      domain, slot, value = exemplar['dsv']

      matching_ds = domain == target['domain'] and slot == target['slot']
      new_example = exemplar['gid'] not in self.selected_gids
      not_empty = value.lower() in hist or slot in hist or value == 'any'

      if new_example and matching_ds and not_empty:
        self.selected_gids.append(exemplar['gid'])
        acceptable = True
        self.regular_loops += 1
      loops += 1
      if loops > 10000:
        self.missed_loops += 1
        acceptable = True

    return exemplar

  def distance_search(self, example, corpus):
    if len(self.sorted_exemplars) == 0:
      history = ' '.join(example['utterances'])
      exp_embed = self.embed_model.encode(history, show_progress_bar=False)
      cand_embeds = [cand['embedding'] for cand in self.candidates[corpus]]

      if self.search_method == 'cosine':
        self.cosine(exp_embed, cand_embeds)
      elif self.search_method == 'euclidean':
        self.euclidean(exp_embed, cand_embeds)
      elif self.search_method == 'mahalanobis':
        self.mahalanobis(exp_embed, cand_embeds)
      # we can never fit more than 60 in-context examples, so we can stop sorting there
      nearest_indexes = np.argpartition(self.distances, 60)[:60]
      
      for near_id in nearest_indexes:
        exemplar = self.candidates[corpus][near_id]
        self.sorted_exemplars.append(exemplar)

    loops = 0
    acceptable = False
    while not acceptable:
      exemplar = self.sorted_exemplars.pop(0)
      domain, slot, value = exemplar['dsv']
      global_id = exemplar['gid']

      if global_id not in self.selected_gids and value != '<none>':
        self.selected_gids.append(global_id)
        self.regular_loops += 1
        acceptable = True

      if loops > 50:
        self.missed_loops += 1
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
