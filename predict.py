import os, pdb, sys
import json
import numpy as np
import random

def display_errors(wrongs):
  samples = np.random.choice(wrongs, 5, replace=False)
  for sample in samples:
    print(sample['speaker'], sample['current'])
    print(sample['label'], sample['score'], sample['prediction'])

def grade_predictions(results):
  wrongs = []

  true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
  for res in results:

    if res['prediction']:
      if res['label']:
        true_pos += 1   # predicted positive, is truly positive
      else:
        false_pos += 1  # predicted positive, is actually negative

    else:
      if res['label']:
        false_neg += 1   # predicted negative, should have picked positive
        wrongs.append(res)
      else:
        true_neg += 1    # predicted negative, got it correct

  calculate_acc(false_pos, false_neg, results)
  score = calculate_f1(true_pos, false_pos, true_neg, false_neg)
  display_errors(wrongs)

def calculate_acc(false_pos, false_neg, results):
  accuracy = (false_pos + false_neg) / len(results)
  acc = round(accuracy * 100, 2)
  print(f"Accuracy: {acc}%")

def calculate_f1(true_pos, false_pos, true_neg, false_neg):
  epsilon = 1e-9
  precision = true_pos / (true_pos + false_pos + epsilon)
  recall = true_pos / (true_pos + false_neg + epsilon)
  f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

  prec = round(precision, 3)
  rec = round(recall, 3)
  score = round(f1_score, 3)

  print(f"precision: {prec}, recall: {rec}, f1_score: {score}")
  return score

def predict_saliency(annotations):
  results = []
  for convo_id, examples in annotations.items():
    for exp in examples:
      speaker, previous, current = exp['speaker'], exp['previous'], exp['current']
      score = 0.5

      for number in [' 1 ', ' 2 ', ' 3 ', ' 4 ', ' 5 ']:
        if number in current:
          score += 0.3

      for phrase in ['reference', 'postcode', 'thank', 'anything else', 'phone number']:
        if phrase in current.lower():
          score -= 0.3

      if speaker == 'agent':
        if len(current) < 20:
          score -= 0.2

      if speaker == 'customer':
        score += 0.1
        if len(current) < 10:
          score -= 0.2

      if len(current) < 5:
        score -= 0.1

      exp['score'] = score
      exp['prediction'] = score > 0.5
      results.append(exp)

  return results

def load_annotations():
  annotation_path = os.path.join('results', 'annotations', "saliency_final.json")
  annotations = json.load(open(annotation_path, 'r'))
  return annotations

if __name__ == "__main__":
  annotations = load_annotations()
  results = predict_saliency(annotations)
  grade_predictions(results)