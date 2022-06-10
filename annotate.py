import os, pdb, sys
import json
import random
from collections import defaultdict
from nltk.tokenize import sent_tokenize


dev_list=[
  "MUL0129.json",
  "MUL0178.json",
  "MUL0476.json",
  "MUL0602.json",
  "MUL0603.json",
  "MUL1125.json",
  "MUL1160.json",
  "MUL1227.json",
  "MUL1381.json",
  "MUL2020.json",
  "MUL2251.json",
  "MUL2344.json",
  "MUL2418.json",
  "MUL2690.json",
  "PMUL0134.json",
  "PMUL0187.json",
  "PMUL0287.json",
  "PMUL0626.json",
  "PMUL0689.json",
  "PMUL1159.json",
  "PMUL1181.json",
  "PMUL1557.json",
  "PMUL1579.json",
  "PMUL1599.json",
  "PMUL1635.json",
  "PMUL1879.json",
  "PMUL2389.json",
  "PMUL2748.json",
  "PMUL2804.json",
  "PMUL3363.json",
  "PMUL3466.json",
  "PMUL3470.json",
  "PMUL3554.json",
  "PMUL4029.json",
  "PMUL4053.json",
  "PMUL4126.json",
  "PMUL4711.json",
  "SNG0019.json",
  "SNG01172.json",
  "SNG01297.json",
  "SNG02214.json",
  "SNG0271.json",
  "SNG0314.json",
  "SNG0494.json",
  "SNG0907.json",
  "SNG0910.json",
  "SNG1046.json",
  "SNG1069.json"
]
def save_results(results, convo_id, version=1):
  save_path = os.path.join('results', 'annotations', f"saliency_v{version}.json")
  json.dump(results, open(save_path, 'w'), indent=4)
  size = len(results[convo_id])
  print(f"Saved {size} more annotations for a total of {len(results)} conversations")

def create_prior(version=1):
  prior_path = os.path.join('results', 'annotations', f"saliency_v{version}.json")
  if os.path.exists(prior_path):
    prior_results = json.load(open(prior_path, 'r'))
    return prior_results
  else:
    return {}

def load_data():
  data_path = os.path.join('assets', 'mwoz', 'dev.json')
  convos = json.load(open(data_path, 'r'))
  return convos

def annotate_data(data, results, version):
  speakers = ['customer', 'agent']
  data_keys = list(data.keys())
  random.shuffle(data_keys)

  for convo_id in data_keys:
    conversation = data[convo_id]
    if convo_id in results or convo_id in dev_list: 
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

    save_results(results, convo_id, version)
    response = input("End of conversation. Continue?")
    if response not in ['y', 'yes', 'ok', 'c', 'continue']:
      sys.exit()

if __name__ == "__main__":
  version = 2
  data = load_data()
  prior = create_prior(version)
  annotate_data(data, prior, version)