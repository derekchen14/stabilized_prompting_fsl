import os, sys, pdb
import numpy as np
import random
import math

from utils.help import *
from torch.utils.data import DataLoader
from utils.arguments import solicit_params
from utils.load import load_tokenizer, load_data, load_sent_transformer
from assets.static_vars import device, debug_break
from sentence_transformers import LoggingHandler, losses, util, InputExample, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

def fit_model(args, model, dataloader, evaluator):
  loss_function = losses.ContrastiveLoss(model=model)
  # loss_function = losses.CosineSimilarityLoss(model=model)
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

def build_evaluator(args):
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

def mine_for_negatives(args):
  cache_file = f'mpnet_{args.style}_none_lookback3_embeddings.pkl'
  cache_path = os.path.join(args.input_dir, 'cache', args.dataset, cache_file)  
  samples = pkl.load( open( cache_path, 'rb' ) )

  all_pairs = compute_scores(samples):
  selected_pairs = select_top_k(all_pairs)
  return selected_pairs

def compute_scores(samples):
  num_samples = len(samples)
  all_pairs = []
  for i in range(num_samples):
    for j in range(num_samples - 1):
      s_i = samples[i]
      s_j = samples[j]
      # sim_score = state_change_sim(s_i, s_j)
      sim_score = domain_slot_sim(s_i, s_j)
      pair = {'si': s_i['history'], 'sj': s_j['history'], 'score': sim_score}
      all_pairs.append(pair)
  return all_pairs

def select_top_k(all_pairs, k=1000):
  all_pairs.sort(key=lambda pair: pair['score'])
  
  selected_pairs = []
  for high_pair in all_pairs[:k]:
    exp = InputExample(texts=[high_pair['si'], high_pair['sj']], label=high_pair['score'])
    selected_pairs.append(exp)
  for low_pair in all_pairs[-1000:]:
    exp = InputExample(texts=[low_pair['si'], low_pair['sj']], label=low_pair['score'])
    selected_pairs.append(exp)

  return selected_pairs

def domain_slot_sim(a, b):
  domain_a, slot_a, value_a = a['dsv']
  domain_b, slot_b, value_b = b['dsv']

  sim_score = 0
  if domain_a == domain_b:
    sim_score += 0.3
  if slot_a == slot_b:   # not else if, this is cumulative
    sim_score += 0.7
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

  dataloader = DataLoader(datasets['train'], shuffle=True, batch_size=args.batch_size)
  model = load_sent_transformer(args, for_train=True)
  model = add_special_tokens(model)
  evaluator = build_evaluator(args)
  fit_model(args, model, dataloader, evaluator)