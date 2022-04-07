import os, pdb, sys
import random
import json
import math
import pickle as pkl
import numpy as np

from assets.static_vars import device, DATASETS, GENERAL_TYPO, DOMAIN_SLOTS
from components.datasets import MetaLearnDataset, InContextDataset, FineTuneDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm as progress_bar
from collections import defaultdict

def check_cache(args):
  cache_file = f'{args.dataset}_{args.task}.pkl'
  cache_path = os.path.join(args.input_dir, 'cache', args.dataset, cache_file)
  use_cache = not args.ignore_cache

  if os.path.exists(cache_path) and use_cache:
    res = pkl.load( open( cache_path, 'rb' ) )
    if args.do_train:
      print(f"Loaded {len(res['train'])} train and {len(res['dev'])} dev examples from {cache_path}")
    elif args.do_eval:
      print(f"Loaded {len(res['test'])} test examples from {cache_path} for evaluation")
    return res, True
  else:
    print(f'Creating new dataset for {args.dataset.upper()} from scratch ...')
    return cache_path, False

def extract_label(targets):
  # returns a list of (domain, slot, value) tuples when the domain is an active 
  swaps = {'not mentioned': 'none', 'dontcare': 'any', '': 'none'}
  labels = []

  for domain, domain_data in targets.items():
    domain_data = targets[domain]
    active_domain = False

    for slot, value in domain_data['semi'].items():
      if len(value) > 0:
        active_domain = True

    if active_domain:
      for slot, value in domain_data['book'].items():
        if not isinstance(value, list):
          if value in swaps:
            value = swaps[value]
          labels.append((domain, slot, value))
      for slot, value in domain_data['semi'].items():
        if value in swaps:
          value = swaps[value]
        if value in GENERAL_TYPO:
          value = GENERAL_TYPO[value]
        labels.append((domain, slot, value))
  return labels

def build_mwoz21(args, data, label_set):
  # written for MultiWoz v2.1, 2.3 and 2.4
  examples = []
  speakers = ["<customer>", "<agent>"]

  for convo_id, conversation in progress_bar(data.items(), total=len(data)):
    text_so_far = []
    speaker_id = 0
    goals = conversation['goal']
    if len(goals['police']) > 0 or len(goals['hospital']) > 0:
      continue

    for turn in conversation['log']:
      text = turn['text']
      speaker = speakers[speaker_id]
      utterance = f"{speaker} {text}"
      
      if speaker == '<agent>':
        domain, d_tracker = extract_domain(targets, label_set, d_tracker)
        context = ' '.join(text_so_far)
        targets = extract_label(turn['metadata'])
        for domain, slot, value in targets:
          target = {'domain': domain, 'slot': slot, 'value': value}
          example = {'history': context, 'current': utterance, 'target': target}
          examples.append(example)

      text_so_far.append(utterance)  # add agent utterance afterwards
      
      speaker_id = 1 - speaker_id
      if len(text_so_far) > args.context_len:
        text_so_far = text_so_far[-args.context_len:]

  return examples

def build_mwoz(args, data):
  ''' Written for raw v2.2 mwoz. This follows the schema format built by SGD'''
  examples = []
  speakers = {'user': '<customer>', 'system': '<agent>'}
  allowed_domains = list(DOMAIN_SLOTS.keys())

  for conversation in progress_bar(data, total=len(data)):
    text_so_far = []

    for turn in conversation['turns']:
      text = turn['utterance']
      speaker = speakers[turn['speaker']]
      utterance = f"{speaker} {text}"
      text_so_far.append(utterance)
      
      if len(turn['frames']) > 0 and speaker == '<customer>':
        act_dom = [fr['service'] for fr in turn['frames'] if fr['state']['active_intent'] != "NONE"]
        
        for frame in turn['frames']:
          current_domain = frame['service']
          if current_domain in allowed_domains:

            slotvals = frame['state']['slot_values']
            if len(slotvals) > 0:
              active_slots = [domain_slot.split('-')[1] for domain_slot, _ in slotvals.items()]
              
              for slot in DOMAIN_SLOTS[current_domain]:
                if slot in active_slots:
                  domain_slot = '-'.join([current_domain, slot])
                  value = slotvals[domain_slot][0]
                else:
                  value = 'none'

                history = ' '.join(text_so_far[:-1])
                target = {'domain': current_domain, 'slot': slot, 'value': value,
                        'global_id': conversation['dialogue_id'] + '_' + turn['turn_id'] }

                example = {'history': history, 'current': utterance, 'target': target}
                examples.append(example)
      
      if len(text_so_far) > 10:
        text_so_far = text_so_far[-10:]
  return examples

def interact_mwoz(data, mapping):
  examples = []
  speakers = ["Customer: ", "Agent: "]

  for convo_id, conversation in progress_bar(data.items(), total=len(data)):
    text_so_far = []
    speaker_id = 0
    
    for turn in conversation['log']:
      text = turn['text']
      speaker = speakers[speaker_id]
      utterance = f"{speaker} {text}"
      text_so_far.append(utterance)
      context = ' '.join(text_so_far)
      
      if speaker == 'Agent: ':
        examples.append({'context': context, 'prompt': 'n/a', 'label': 'n/a'})  
      
      speaker_id = 1 - speaker_id
      if len(text_so_far) > 10:
        text_so_far = text_so_far[-10:]

  return examples

def create_abcd_mappings(ontology):
  intent_map = {}
  for flow, subflows in ontology['intents'].items():
    for intent in subflows:
      intent_map[intent] = flow

  enumerate_map = {}
  for slot, values in ontology['values']['enumerable'].items():
    enumerate_map[slot] = True
  for slot in ontology['values']['non_enumerable']:
    enumerate_map[slot] = False

  validity_map = {}
  for action, slots in ontology['actions']['has_slotval'].items():
    validity_map[action] = True
  for action in ontology['actions']['empty_slotval']:
    validity_map[action] = False

  mappings = {'intent': intent_map, 'enum': enumerate_map, 'valid': validity_map}
  return mappings

def extract_state(intent, action, value, ontology, mappings):
  target = {}
  valid = False
  valid_actions = ontology['actions']['has_slotval']

  cand_value = value.lower().strip()
  if mappings['valid'][action]:
    valid = True
    candidate_slots = valid_actions[action]

    target['domain'] = mappings['intent'][intent]
    if len(candidate_slots) == 1:
      target['slot'] = candidate_slots[0]
      target['value'] = cand_value
    else:
      for cand_slot in candidate_slots:
        if mappings['enum'][cand_slot]:
          cand_values = ontology['values']['enumerable'][cand_slot]
          if cand_value in cand_values:
            target['slot'] = cand_slot
            target['value'] = cand_value
        elif cand_slot in ontology['values']['non_enumerable']:
          target['slot'] = cand_slot
          target['value'] = cand_value

  return target, valid

def build_abcd(args, data, ontology):
  examples = []
  mappings = create_abcd_mappings(ontology)

  for convo in progress_bar(data, total=len(data)):
    # each convo has keys: convo_id, scenario, original, delexed, conversation
    utt_so_far = []
    for turn in convo['conversation']:
      # each turn has keys: speaker, text, targets, turn_count, candidates
      speaker = turn['speaker']

      if speaker == 'action':  # skip action turns
        intent, nextstep, action, value, utt_rank = turn['targets']
        # each target is a 5-part list: intent, nextstep, action, value, utt_rank
        target, valid = extract_state(intent, action, value, ontology, mappings)
        target['global_id'] = str(convo['convo_id']) + '_' + str(turn['turn_count'])
  
        if valid:
          context = ' '.join(utt_so_far[:-1])
          current_utt = utt_so_far[-1]
          example = {'history': context, 'current': current_utt, 'target': target}
          examples.append(example)  
      else:
        text = turn['text']
        utt_so_far.append(f"<{speaker}> {text}")

    if len(utt_so_far) > 10:
      utt_so_far = utt_so_far[-10:]

  return examples

def build_dstc(args, data):
  ''' extra contains the structured label as a value '''
  examples = []

  for convo in progress_bar(data, total=len(data)):
    text_so_far = []

    for turn in convo['conversation']:
      target = {
        'global_id': convo['guid'] + '_' + turn['turn'],
        'domain': 'restaurant' }

      if turn['speaker'] == 'agent':
        sys_text = f"<agent> {turn['text']}"
        text_so_far.append(sys_text)
  
      elif turn['speaker'] == 'user':
        user_text = f"<customer> {turn['text']}"
        history = ' '.join(text_so_far)
        text_so_far.append(user_text)

        for slot, value in turn['inform'].items():
          # TODO: add negatives to predict "none"
          target['slot'] = slot
          target['value'] = value
          examples.append({'history': history, 'current': user_text, 'target': target})

      if len(text_so_far) > 10:
        text_so_far = text_so_far[-10:]

  return examples

def build_gsim(data, mapping):
  examples = []

  for conversation in progress_bar(data, total=len(data)):
    dialog_id = conversation['dialogue_id']
    domain = dialog_id.split('_')[0]
    text_so_far = []    

    for turn_count, turn in enumerate(conversation['turns']):
      if 'system_utterance' in turn:
        sys_text = turn['system_utterance']['text']
        sys_utt = f"<agent> {sys_text}"
        text_so_far.append(sys_utt)


      user_text = turn['user_utterance']['text']
      user_utt = f"<customer> {user_text}"
      context = ' '.join(text_so_far)
      text_so_far.append(user_utt)

      for state in turn['dialoge_state']:
        target = {'domain': domain, 
                    'slot': state['slot'],
                   'value': state['value'],  
               'global_id': dialog_id + '_' + str(turn_count + 1) }
        example = {'history': context, 'current': user_utt, 'target': target}
        examples.append(example)

  return examples

def build_sgd(args, data, mapping, split):
  examples = []
  prompt = "The topic of conversation is about"

  for conversation in progress_bar(data, total=len(data)):
    text_so_far = []

    for turn_count, turn in enumerate(conversation['turns']):    
      text = turn['utterance']

      if turn['speaker'] == 'SYSTEM':
        sys_text = f"<agent> {text}"
        text_so_far.append(sys_text)
  
      elif turn['speaker'] == 'USER':
        user_utt = f"<customer> {text}"
        text_so_far.append(user_utt)

        for frame in turn['frames']:
          service = frame['service'].split('_')[0]

          if 'state' in frame:
            for slot, value in frame['state']['slot_values'].items():
              target = {'domain': service, 'slot': slot, 'value': value[0].strip(),
                    'global_id': conversation['dialogue_id'] + '_' + str(turn_count+1) }
              examples.append({'utterances': text_so_far, 'target': target})

      if len(text_so_far) > 10:
        text_so_far = text_so_far[-10:]
  
  return examples

def build_tt(args, data, ontology):
  examples = []
  for convo in progress_bar(data, total=len(data)):  
    text_so_far = []    

    for turn in convo['utterances']:
      text = turn['text']

      if turn['speaker'] == 'assistant':
        sys_utterance = f"<agent> {text}"
        text_so_far.append(sys_utterance)

      elif turn['speaker'] == 'user':
        user_utterance = f"<customer> {text}"
        context = ' '.join(text_so_far)
        text_so_far.append(user_utterance)

        if 'segments' in turn:
          labels = extract_slotvals(turn['segments'], ontology)
          for slot, value in labels.items():
            target = {'domain': 'movies', 'slot': slot, 'value': value}
            examples.append({'history': context, 'current': user_utterance, 'target': target})

      if len(text_so_far) > 10:
        text_so_far = text_so_far[-10:]
  return examples

def extract_slotvals(segments, ontology):
  labels = {}
  for segment in segments:
    slot_candidate = segment['annotations'][0]['name']
    value = segment['text']
    if slot_candidate in ontology:
      slot = ontology[slot_candidate]
      labels[slot] = value
  return labels

def get_dataloader(args, dataset, split='train'):
  sampler = RandomSampler(dataset) if dataset.shuffle else SequentialSampler(dataset)
  collate = dataset.collate_func
  dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)
  print(f"Loaded {split} data with {len(dataloader)} batches")
  return dataloader

def prepare_examples(args, data, ontology, split):
  """ Each example is a dict which should have:
    history: up to the last 10 utterances
      dialogue - defined to be the history + current_utt
    current: the current utterances, which is speaker + text
      speakers are either <agent> or <customer>
    target: a dictionary with keys global_id, domain, slot and value
  """
  if args.dataset == 'abcd':    # Action Based Conversations
    examples = build_abcd(args, data) 
  elif args.dataset == 'dstc':  # State Tracking Challenge 2
    examples = build_dstc(args, data) 
  elif args.dataset == 'gsim':    # Google Simulated Chats
    examples = build_tt(args, data, ontology) 
  elif args.dataset.startswith('mwoz'):  # MultiWoz 2.1 or 2.2
    examples = build_mwoz(args, data)
  elif args.dataset == 'sgd':   # Schema Guided Dialogue
    examples = build_sgd(args, data, ontology, split) 
  elif args.dataset == 'tt':    # TicketTalk / TaskMaster 3
    examples = build_tt(args, data, ontology) 

  return examples

def hold_out(args, datasets):
  if args.num_shots == 'zero':

    for split in ['train', 'dev', 'test']:
      original = datasets[split].data
      kept_data = []
      for example in original:
        current_domain, slot, value = example['extra']['dsv']

        # keep the chosen domain in the dev and test sets
        if current_domain == args.left_out:
          if split in ['dev', 'test']:
            kept_data.append(example)

        else:  # hold out the chosen domain from the train set
          if split == 'train':
            kept_data.append(example)

      datasets[split].data = kept_data
      new_size = len(kept_data)
      datasets[split].size = new_size
      if args.verbose:
        print(f"Previously the {split} size was {len(original)}. Now it is {new_size}.")

  elif args.num_shots == 'few':
    # separate the chosen domain from the other domains, but keep both
    # the "few" examples for in-context learning will be sampled during training
    # only keep the chosen domain in the dev and test sets
    pass
  elif args.num_shots == 'percent':
    # sample some percentage from the chosen domain
    # keep all data from the other domains for training
    # only keep the chosen domain in the dev and test sets
    percent_keep = args.threshold

  return datasets

def process_data(args, raw_data, tokenizer):
  label_set = [] # raw_data['ontology']

  cache_results, already_exist = check_cache(args)
  if already_exist:
    datasets = cache_results
  else:
    datasets = {}
    for split in ['train', 'dev', 'test']:
      examples = prepare_examples(args, raw_data[split], label_set, split)
      if args.task == 'meta_learn':
        datasets[split] = MetaLearnDataset(args, examples, tokenizer, split)
      elif args.task == 'in_context':
        datasets[split] = InContextDataset(args, examples, tokenizer, split)
      elif args.task == 'fine_tune':
        datasets[split] = FineTuneDataset(args, examples, tokenizer, split)
      print(f"Running with {len(datasets[split])} {split} examples")
    pkl.dump(datasets, open(cache_results, 'wb'))

  datasets = hold_out(args, datasets)
  return datasets, label_set
