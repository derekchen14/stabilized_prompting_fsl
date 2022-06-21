import os, sys, pdb
import numpy as np
import random
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from collections import Counter
from utils.help import *
from utils.arguments import solicit_params
from utils.load import load_tokenizer, load_data, load_sent_transformer
from assets.static_vars import device, debug_break
from sentence_transformers import LoggingHandler, losses, util, InputExample, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, InformationRetrievalEvaluator

class IdentityLoss(nn.Module):
  """Custom contrastive loss for DST"""
  def __init__(self, model):
    super().__init__()
    self.model = model

  def __getitem__(self, key):
      return getattr(self, key, None)

  def forward(self, sentence_features, targets):
    return torch.zeros(1, requires_grad=False)

class DomainSlotValueLoss(nn.Module):
  """Custom contrastive loss for DST"""
  def __init__(self, args, model):
    super().__init__()
    self.model = model
    self.loss_function = nn.MSELoss()

    self.batch_size = args.batch_size
    self.register_buffer("temperature", torch.tensor(args.temperature))
    self.verbose = args.verbose

  def __getitem__(self, key):
      return getattr(self, key, None)

  def _convert_to_labels(self, targets):
    """ returns a list of labels where each transformed target is a 4-part tuple
    targets are a 3-part tensor, which changes into a tensor of 4 floats
    indicating whether the pair is matching in [negative, domain, slot, value] """
    positives = (torch.sum(targets, dim=1) > 0.5).float()  # sum along 
    labels = torch.cat([positives.unsqueeze(1), targets], dim=1)
    if len(labels) != self.batch_size:
      print('size', len(labels))
    # assert len(labels) == self.batch_size
    return labels

  def forward(self, sentence_features, targets):
    # sentence_features is Iterable[Dict[str, Tensor]]
    # labels is an integer representing a binary encoding
    labels = self._convert_to_labels(targets)
    reps = [self.model(sent_feat)['sentence_embedding'] for sent_feat in sentence_features]
    distances = 1 - torch.cosine_similarity(reps[0], reps[1])

    pos_values, pos_slots, pos_domains, negatives = 0.0, 0.0, 0.0, 0.0
    for distance, label in zip(distances, labels):
      pos_values += label[3] * distances.pow(2)
      pos_slots += label[2] * F.relu(0.3 - distance).pow(2)
      pos_domains += label[1] * F.relu(0.8 - distance).pow(2)
      negatives += label[0] * F.relu(1.0 - distance).pow(2)

    loss = 0.5 * (pos_values + pos_slots + pos_domains + negatives)
    return loss.mean()

def fit_model(args, model, dataloader, evaluator):
  if args.loss_function in ['cosine', 'zero_one']:
    loss_function = losses.CosineSimilarityLoss(model=model)
  elif args.loss_function == 'contrast':
    loss_function = losses.ContrastiveLoss(model=model)
  elif args.loss_function == 'custom':
    loss_function = DomainSlotValueLoss(args, model)
  elif args.loss_function == 'default':
    loss_function = IdentityLoss(model)

  warm_steps = math.ceil(len(dataloader) * args.n_epochs * 0.1) # 10% of train data for warm-up
  ckpt_name = f'lr{args.learning_rate}_k{args.kappa}_{args.loss_function}.pt'
  ckpt_path = os.path.join(args.output_dir, 'sbert', ckpt_name)

  # By default, uses AdamW optimizer with learning rate of 3e-5, WarmupCosine scheduler
  model.fit(train_objective=(dataloader, loss_function),
          evaluator=evaluator,
          epochs=args.n_epochs,
          logging_steps=args.log_interval,
          checkpoint_save_steps=args.checkpoint_interval,
          warmup_steps=warm_steps,
          save_best_model=args.do_save,
          optimizer_params={'lr': args.learning_rate},
          output_path=ckpt_path,
          checkpoint_path=ckpt_path,
          do_qual=args.qualify,
          args=args)
  return model

def build_evaluator(args, dev_samples):
  # https://www.sbert.net/docs/package_reference/evaluation.html
  print("Building the evaluator which is cosine similarity by default")
  # if args.loss_function == 'custom':
  queries, corpus, relevant_docs = dev_samples
  evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs,
                   precision_recall_at_k=[1, 5, 10], name=args.loss_function)
  # else:
  #   evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, 
  #         show_progress_bar=False, name='mwoz', write_csv=args.do_save)
  # alternatively, where sentences1, sentences2, and scores are lists of equal length
  # evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
  return evaluator

def add_special_tokens(model):
  word_embedding_model = model._first_module()
  tokens = ["<agent>", "<customer>", "<label>", "<none>", "<remove>", "<sep>", "<pad>"]
  word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
  word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
  print("Model loaded")
  return model

def load_from_cache(args):
  cache_file = f'mpnet_mwoz_{args.num_shots}_default_embeddings.pkl'
  cache_path = os.path.join(args.input_dir, 'cache', args.dataset, cache_file)  
  samples = pkl.load( open( cache_path, 'rb' ) )
  return samples

def mine_for_samples(args):
  # Load raw dialogue data from cache
  samples = load_from_cache(args)
  # Look for positive and negative example pairs to be used as training data based on similarity
  all_pairs = compute_scores(args, samples)
  # all_pairs.sort(key=lambda exp: exp.label)
  divide = int(len(all_pairs) * 0.8)
  selected_pairs = all_pairs[:divide]
  dev_pairs = all_pairs[divide:]
  print('train size', len(selected_pairs), 'dev size', len(dev_pairs))
  return selected_pairs, dev_pairs

def mine_for_queries(args):
  # Load raw dialogue data from cache
  samples = load_from_cache(args)
  # Look for positive and negative example pairs to be used as training data based on similarity
  all_pairs = compute_scores(args, samples)
  divide = int(len(all_pairs) * 0.8)
  selected_pairs = all_pairs[:divide]
  # Gather the queries, corpus and relevant docs
  divide = int(len(samples) * 0.5)
  dev_data = gather_docs(args, samples[divide:])
  return selected_pairs, dev_data

def select_test_data(args):
  samples = load_from_cache(args)
  total_size = args.kappa * args.batch_size
  test_data = random.sample(samples, total_size)
  return test_data

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

def gather_docs(args, samples):
  """ queries is a dict with key of qid and value of utterance string
  corpus is a dict with key of cid and value of utterance string
  relevant docs is a dict with key of qid and value is a list of cids that are relevant
  """
  corpus, queries = {}, {}
  relevant_docs = defaultdict(list)
  print(f"Gathered {len(samples)} queries")

  for out_sample in progress_bar(samples, total=len(samples), desc='Gathering queries'):
    cid = out_sample['gid']
    dialogue = out_sample['history']
    corpus[cid] = dialogue

    for in_sample in samples:
      qid = in_sample['gid']
      if cid == qid: continue

      domain, slot, value = in_sample['dsv']
      valid = slot in ['internet', 'parking']  # default to False most of the time
      for part in value.lower().replace(':', ' ').replace('|', ' ').split():
        if part in in_sample['history'].lower():
          valid = True
      if not valid: continue

      sim_score = domain_slot_sim(in_sample, out_sample)
      if sim_score > 0.6:
        relevant_docs[qid].append(cid)
        queries[qid] = in_sample['history']
      
  return queries, corpus, relevant_docs

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
  pair_ids = set()
  breakdown = Counter()
  all_pairs = []
  for i in progress_bar(range(num_samples), total=num_samples, desc='Computing scores'):
    s_i = samples[i]
    candidates = random.sample(samples, args.kappa)
    # print("KEY:", s_i['history'], s_i['dsv'])
    
    for s_j in candidates:
      if s_i['gid'] == s_j['gid']: continue
      domain, slot, value = s_j['dsv']

      valid = slot in ['internet', 'parking']  # default to False most of the time
      for part in value.lower().replace(':', ' ').replace('|', ' ').split():
        if part in s_j['history'].lower():
          valid = True
      if not valid: continue

      joint_id = create_joint_id(s_i, s_j)
      if joint_id in pair_ids:
        continue
      else:
        pair_ids.add(joint_id)

      if args.loss_function == 'cosine':
        sim_score = domain_slot_sim(s_i, s_j)
        target = sim_score
        threshold = 0.4
      elif args.loss_function in ['zero_one', 'default']:
        sim_score = zero_one_sim(s_i, s_j)
        target = sim_score
        threshold = 0.4
      elif args.loss_function == 'contrast':
        sim_score = 1 if s_i['dsv'][1] == s_j['dsv'][1] else 0
        target = sim_score
        threshold = 0.2
      elif args.loss_function == 'custom':
        target = encode_as_bits(s_i, s_j)
        sim_score = sum(target)
        threshold = 0.3

      
      if sim_score == 0 and random.random() > threshold:
        continue  # only keep a portion of negatives to keep things balanced
      breakdown[sim_score] += 1
      pair = InputExample(texts=[s_i['history'], s_j['history']], label=target)
      all_pairs.append(pair)
    """
      if random.random() < 0.3:
        print("   --", s_j['history'], sim_score, s_j['dsv'])
    if i >= 10:
      pdb.set_trace()
    """
  print(breakdown)
  return all_pairs

def create_joint_id(a, b):
  small = min(a['gid'], b['gid'])
  large = max(a['gid'], b['gid'])
  joint_id = small + '-' + large
  return joint_id

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

def zero_one_sim(a, b):
  domain_a, slot_a, value_a = a['dsv']
  domain_b, slot_b, value_b = b['dsv']

  if domain_a == domain_b and slot_a == slot_b:
    return 1.0
  else:
    return 0.0

def encode_as_bits(a, b):
  matches = []
  for property_a, property_b in zip(a['dsv'], b['dsv']):
    if property_a == property_b:
      matches.append(1.0)
    else:
      matches.append(0.0)
  return matches
  # match_str = ''.join(matches)
  # sim_score = int(match_str, 2)
  # score_str = bin(target)[2:]

def state_change_sim(a, b):
  diff = a - b
  sim_score = float(diff) / 5.0  # Normalize score to range 0 ... 1
  aa, bb = row['sentence1'], row['sentence2']
  return sim_score

def test_collate(batch):
  utterances, states = [], []
  for example in batch:
    utterances.append(example['history'])
    states.append(example['dsv'])
  return utterances, states

def test_model(args, model, dataloader):
  count = 0
  for utterances, states in dataloader:
    features = model._first_module().tokenize(utterances)  # dict with inputs and attn_mask
    features['input_ids'] = features['input_ids'].to(device)
    features['attention_mask'] = features['attention_mask'].to(device)
    
    with torch.no_grad():
      outputs = model(features)
    model.qualify(outputs, utterances)
    
    count += 1
    if count > 10: break
    print("\n   ---------------   \n")

if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  set_seed(args)

  if args.do_eval and args.use_tuned:
    use_fine_tune = True
  elif args.loss_function == 'default':
    use_fine_tune = False
  else:
    use_fine_tune = False

  model = load_sent_transformer(args, use_tuned=use_fine_tune)
  model = add_special_tokens(model)
  model.to(device)

  if args.do_train:
    # train_samples, dev_samples = mine_for_samples(args)
    train_samples, dev_samples = mine_for_queries(args)
    dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
    evaluator = build_evaluator(args, dev_samples)
    fit_model(args, model, dataloader, evaluator)

  elif args.do_eval:
    test_samples = select_test_data(args)
    dataloader = DataLoader(test_samples, shuffle=True, batch_size=args.batch_size)
    dataloader.collate_fn = test_collate
    test_model(args, model, dataloader)

"""
  def sup_con_loss(self, sentence_features, labels):
    embeds = [self.model(sent_feat)['sentence_embedding'] for sent_feat in sentence_features]
    # embeds[0] has shape batch_size, embed_dim
    assert len(embeds) == 2

    # transform labels vector into a matrix of 0.0s and 1.0s (aka. floats)
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Reshapes the embeddings to merge the batch_size and number of positive classes
    anchor_count = embeds.shape[1]   # should be 3 for Domain, Slot and Value
    anchor_feature = torch.cat(torch.unbind(embeds, dim=1), dim=0)  # only embed_dim remains

    anchor_dot_contrast = torch.div(
      torch.matmul(anchor_feature, contrast_feature.T),
      self.temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
      torch.ones_like(mask),
      1,
      torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
      0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss

  def simclr_loss(self, sentence_features, labels):
    embeds = [self.model(sent_feat)['sentence_embedding'] for sent_feat in sentence_features]
    z_i = F.normalize(embeds[0], dim=1)
    z_j = F.normalize(embeds[1], dim=1)
    rep = torch.cat([z_i, z_j], dim=0)

    similarity_matrix = F.cosine_similarity(rep.unsqueeze(1), rep.unsqueeze(0), dim=2)
    if self.verbose: print("Similarity matrix", similarity_matrix.shape)
  
    def pairwise_loss(i, j):
      z_i_, z_j_ = rep[i], rep[j]
      sim_i_j = similarity_matrix[i, j]
      if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
          
      numerator = torch.exp(sim_i_j / self.temperature)
      default_ones = torch.ones((2 * self.batch_size, ))
      one_for_not_i = default_ones.to(emb_i.device).scatter_(0, torch.tensor([i]), 0.0)
      if self.verbose: print(f"one for not i", one_for_not_i.shape)
      
      partition_func = torch.exp(similarity_matrix[i, :] / self.temperature)
      denominator = torch.sum(one_for_not_i * partition_func)
      if self.verbose: print("Denominator", denominator.shape)
          
      loss_ij = -torch.log(numerator / denominator)
      if self.verbose: print(f"loss({i},{j})={loss_ij}")
      return loss_ij.squeeze(0)

    N = self.batch_size
    total_loss = 0.0
    for k in range(0, N):
        total_loss += self.pairwise_loss(k, k + N) + self.pairwise_loss(k + N, k)
    total_loss *= 1.0 / (2*N)
    return total_loss
  """
