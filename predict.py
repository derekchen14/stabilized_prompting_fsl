import os, pdb, sys
import json
import numpy as np
import random
import re

def display_errors(wrongs, sample_size=5):
  if len(wrongs) == 0:
    print("No more false negatives!")
    sys.exit()
  elif len(wrongs) < sample_size:
    print(wrongs)
    print("These are all the mistakes left")
    sys.exit()

  samples = np.random.choice(wrongs, sample_size, replace=False)
  for sample in samples:
    print(sample['speaker'], sample['current'])
    print(sample['convo_id'], sample['label'], sample['score'], sample['prediction'])

def many_capital_letters(current):
  num_caps = 0
  for token in current.split():
    if token[0].isupper():
      num_caps += 1
  return num_caps > 2

def predict_saliency(annotations):
  results = []
  for convo_id, examples in annotations.items():
    for exp in examples:
      speaker, previous, current = exp['speaker'], exp['previous'], exp['current']
      score = 0.5

      if re.search(r"\s\d\s", current):  # digit surrounded by whitespace
        score += 0.3
      if re.search(r"\d\d:\d\d", current):  # HH:MM time
        score += 0.2
      for domain in ['restaurant', 'taxi', 'hotel', 'attraction', 'train']:
        if domain in current.lower():
          score += 0.2
      for number in ['one', 'two', 'three', 'four', 'five', 'six']:
        if number in current.lower():
          score += 0.1
      for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
        if day in current.lower():
          score += 0.1
      for direction in ['north', 'south', 'east', 'west']:
        if direction in current.lower():
          score += 0.1
      for phrase in ['looking for']:
        if phrase in current.lower():
          score += 0.2
      for phrase in ['do not', "don't care", "don't have", 'preference', 'yes', 'but']:
        if phrase in current.lower():
          score += 0.1
      if many_capital_letters(current):
        score += 0.1

      for phrase in ['reference', 'postcode', 'thank', ' else', 'phone number', 
                        'booking', 'contact number']:
        if phrase in current.lower():
          score -= 0.2
      if re.search(r"(\d|[A-Z]){7,}", current):  # reference number of at least 6 characters
        score -= 0.2
      if speaker == 'agent':
        if len(current) < 20:
          score -= 0.1
        elif len(current) < 10:
          score -= 0.2
      if speaker == 'customer':
        score += 0.1
        if len(current) < 10:
          score -= 0.2
        if current[-1] == '?':
          score -= 0.05
      if len(current) < 5:
        score -= 0.1

      exp['score'] = round(score, 2)
      exp['prediction'] = score >= 0.5
      exp['convo_id'] = convo_id
      results.append(exp)

  return results

def grade_predictions(results):
  wrongs = []

  true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
  for res in results:

    if res['prediction']:
      if res['label']:
        true_pos += 1   # predicted positive, is truly positive
      else:
        false_pos += 1  # predicted positive, is actually negative
        wrongs.append(res)

    else:
      if res['label']:
        false_neg += 1   # predicted negative, should have picked positive
      else:
        true_neg += 1    # predicted negative, got it correct

  calculate_acc(true_pos, true_neg, results)
  score = calculate_f1(true_pos, false_pos, true_neg, false_neg)
  display_errors(wrongs)

def calculate_acc(true_pos, true_neg, results):
  accuracy = (true_pos + true_neg) / len(results)
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

def load_annotations():
  annotation_path = os.path.join('results', 'annotations', "saliency_final.json")
  annotations = json.load(open(annotation_path, 'r'))
  return annotations

if __name__ == "__main__":
  annotations = load_annotations()
  results = predict_saliency(annotations)
  grade_predictions(results)