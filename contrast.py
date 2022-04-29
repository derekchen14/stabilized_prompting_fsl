import os, sys, pdb
import numpy as np
import random
import math

from utils.help import *
from torch.utils.data import DataLoader
from utils.arguments import solicit_params
from utils.load import load_tokenizer, load_data, load_sent_transformer
from assets.static_vars import device, debug_break
from sentence_transformers import LoggingHandler, losses, util, InputExample
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

def mine_for_negatives(args):
  # tokenizer = load_tokenizer(args)
  raw_data = load_data(args)

  datasets = {}
  for split, rows in raw_data.items():
    examples = []
    for row in rows:
      score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
      aa, bb = row['sentence1'], row['sentence2']
      examples.append(InputExample(texts=[aa, bb], label=score))
    datasets[split] = examples

  return datasets

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
  
  datasets = mine_for_negatives(args)
  dataloader = DataLoader(datasets['train'], shuffle=True, batch_size=args.batch_size)
  model = load_sent_transformer(args, for_train=True)
  evaluator = build_evaluator(args)
  fit_model(args, model, dataloader, evaluator)