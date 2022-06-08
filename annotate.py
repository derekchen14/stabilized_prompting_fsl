import os, pdb, sys
import json
from collections import defaultdict
from nltk.tokenize import sent_tokenize

def save_results(results, convo_id, version=1):
  save_path = os.path.join('results', 'annotations', f"saliency_v{version}.json")
  json.dump(results, open(save_path, 'w'), indent=4)
  size = len(results[convo_id])
  print(f"Saved {size} more annotations for a total of {len(results)} conversations")

def create_prior():
  prior_path = os.path.join('results', 'annotations', "saliency_v1.json")
  if os.path.exists(prior_path):
    prior_results = json.load(open(prior_path, 'r'))
    return prior_results
  else:
    return {}

def load_data():
  data_path = os.path.join('assets', 'mwoz', 'dev.json')
  convos = json.load(open(data_path, 'r'))
  return convos

def annotate_data(data, results):
  speakers = ['customer', 'agent']
  
  for convo_id, conversation in data.items():
    if convo_id in results: 
      continue
    else:
      results[convo_id] = []

    prev_sent = ""
    speaker_id = 0
    for turn in conversation['log']:

      speaker = speakers[speaker_id]
      print(speaker)

      for sentence in sent_tokenize(turn['text']):
        annotation = input(sentence + " --> ")

        annotation = annotation.strip()
        example = {'speaker': speaker, 'previous': prev_sent, 'current': sentence}
        if annotation in ['s', 'salient', 'S', 'd', 'True', 'true', 'y']:
          example['label'] = True
        elif annotation in ['n', 'not', 'N', 'h', 'False', 'false']:
          example['label'] = False
        else:
          continue
        
        results[convo_id].append(example)
        prev_sent = sentence
      speaker_id = 1 - speaker_id

    save_results(results, convo_id)
    response = input("End of conversation. Continue?")
    if response not in ['y', 'yes', 'ok', 'c', 'continue']:
      sys.exit()

if __name__ == "__main__":
  data = load_data()
  prior = create_prior()
  annotate_data(data, prior)