import os, sys, pdb
import numpy as np
import random
import math

from torch.utils.data import DataLoader
from torch import nn

from utils.help import *
from utils.arguments import solicit_params
from utils.load import load_tokenizer, load_data, load_sent_transformer
from assets.static_vars import device, debug_break
from sentence_transformers import LoggingHandler, losses, util, InputExample, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

class DomainSlotValueLoss(nn.Module):
  """https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveLoss.py"""
  def __init__(self, model):
    super(DomainSlotValueLoss).__init__()
    self.model = model
    self.loss_function = nn.MSELoss()

  def _convert_labels(self, labels):
    """ returns a list of labels where each transformed label is a 3-part tuple
    the three parts in the tuple are 3 booleans
    indicating whether the pair is matching in [domain, slot and value] """
    converted = []
    for label in labels:

      score_str = bin(label)[2:]
      if len(score_str) == 1:
        bin_str = '00' + score_str
      if len(score_str) == 2:
        bin_str = '0' + score_str
      if len(score_str) == 3:
        bin_str = score_str

      bin_list = [int(x) > 0.5 for x in bin_str]
      converted.append(bin_list)
    return converted

  def forward(self, sentence_features, labels):
    # sentence_features is Iterable[Dict[str, Tensor]]
    # labels is an integer representing a binary encoding
    matches = self._convert_labels(labels)
    embeddings = [self.model(sent_feat)['sentence_embedding'] for sent_feat in sentence_features]
    output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
    return self.loss_fct(output, labels.view(-1))

def fit_model(args, model, dataloader, evaluator):
  if args.sbert_loss == 'cosine':
    loss_function = losses.CosineSimilarityLoss(model=model)
  elif args.sbert_loss == 'contrast':
    loss_function = losses.ContrastiveLoss(model=model)
  elif args.sbert_loss == 'custom':
    loss_function = DomainSlotValueLoss(model=model)

  warm_steps = math.ceil(len(dataloader) * args.n_epochs * 0.1) # 10% of train data for warm-up
  ckpt_name = f'lr{args.learning_rate}_k{args.kappa}_{args.finetune}.pt'
  ckpt_path = os.path.join(args.output_dir, 'sbert', ckpt_name)

  model.fit(train_objectives=[(dataloader, loss_function)],
          evaluator=evaluator,
          epochs=args.n_epochs,
          evaluation_steps=args.log_interval,
          warmup_steps=warm_steps,
          output_path=ckpt_path)
  return model

def build_evaluator(args, dev_samples):
  print("Building the evaluator which is cosine similarity by default")
  evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, 
      show_progress_bar=True, name='mwoz', write_csv=args.do_save)
  # InformationRetrievalEvaluator, RerankingEvaluator
  # https://www.sbert.net/docs/package_reference/evaluation.html
  # alternatively, where sentences1, sentences2, and scores are lists of equal length
  # evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
  return evaluator

def add_special_tokens(model):
  word_embedding_model = model._first_module()
  tokens = ["<agent>", "<customer>", "<label>", "<none>", "<remove>", "<sep>", "<pad>"]
  word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
  word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
  return model

def mine_for_samples(args):
  # Load raw dialogue data from cache
  cache_file = f'mpnet_{args.style}_none_lookback3_embeddings.pkl'
  cache_path = os.path.join(args.input_dir, 'cache', args.dataset, cache_file)  
  samples = pkl.load( open( cache_path, 'rb' ) )

  # Look for positive and negative example pairs to be used as training data based on similarity
  all_pairs = compute_scores(args, samples):
  all_pairs.sort(key=lambda exp: exp.label)
  selected_pairs = all_pairs[:args.kappa] + all_pairs[-args.kappa:]
  return selected_pairs

def find_positives_negatives(samples):
  num_samples = len(samples)
  all_pairs = []
  for i in range(num_samples):
    for j in range(num_samples - 1):
      s_i = samples[i]
      s_j = samples[j]
      sim_score = domain_slot_sim(s_i, s_j)
      pair = {'si': s_i['history'], 'sj': s_j['history'], 'score': sim_score}
      all_pairs.append(pair)
  return all_pairs

def compute_scores(args, samples):
  """ InputExample is an object that requires the following API params: 
  texts = a tuple of two example pairs
    Each example is a dialogue string, which has three parts:
      (1) prior dialog state
      (2) two utterances of dialog context
      (3) domain of the current turn
    These serve as both the query and the key, the value is literally the slot value
  label = the score related to that example pair ???
    if the pair is positive, then the score should be high (ie. close to 1.0)
    if it is a negative pair, then the score should be low (ie. close to 0.0)
  """
  num_samples = len(samples)
  all_pairs = []
  for i in range(num_samples):
    for j in range(num_samples - 1):
      s_i = samples[i]
      s_j = samples[j]
      if args.sbert_loss == 'cosine':
        sim_score = domain_slot_sim(s_i, s_j)
      elif args.sbert_loss == 'contrast':
        sim_score = 1 if s_i['dsv'][1] == s_j['dsv'][1] else 0
      elif args.sbert_loss == 'custom':
        sim_score = encode_as_bits(s_i, s_j)
        
      pair = InputExample(texts=[s_i['history'], s_j['history']], label=sim_score)
      all_pairs.append(pair)
  return all_pairs

def domain_slot_sim(a, b):
  domain_a, slot_a, value_a = a['dsv']
  domain_b, slot_b, value_b = b['dsv']

  sim_score = 0
  if domain_a == domain_b:
    sim_score += 0.3
  if slot_a == slot_b:   # not else if, this is cumulative
    sim_score += 0.4
    if value_a == value_b:   # this is conditioned on already matching the slot
      sim_score += 0.3
  return sim_score

def encode_as_bits(a, b):
  matches = ['0'] * 5  # start with 5 leading zeros
  for property_a, property_b in zip(a['dsv'], b['dsv']):
    if property_a == property_b:
      matches.append('1')
    else:
      matches.append('0')
  match_str = ''.join(matches)
  sim_score = int(match_str, 2)

  return sim_score

def state_change_sim(a, b):
  diff = a - b
  sim_score = float(diff) / 5.0  # Normalize score to range 0 ... 1
  aa, bb = row['sentence1'], row['sentence2']
  return sim_score

def run_test(model, datasets):
  test_samples = datasets['test']
  test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
  ckpt_name = f'lr{args.learning_rate}_k{args.kappa}_{args.finetune}.pt'
  ckpt_path = os.path.join(args.output_dir, 'sbert', ckpt_name)
  test_evaluator(model, output_path=ckpt_path)


if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  set_seed(args)

  model = load_sent_transformer(args, for_train=True)
  model = add_special_tokens(model)

  train_samples = mine_for_samples(args)
  dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
  dev_samples = mine_for_samples(args, split='dev')
  evaluator = build_evaluator(args, dev_samples)
  fit_model(args, model, dataloader, evaluator)