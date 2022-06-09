import os, pdb, sys
import random
import json
import math
import pickle as pkl
import numpy as np
import re
from nltk.tokenize import sent_tokenize

from assets.static_vars import *
from utils.help import standardize_format
from components.datasets import MetaLearnDataset, InContextDataset, FineTuneDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm as progress_bar
from collections import defaultdict, Counter

def check_cache(args):
  saliency = 'filter' if args.filter else 'keepall'
  cache_file = f'{args.model}_{args.task}_{args.prompt_style}_{saliency}.pkl'
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

def normalize_length(value):
  parts = value.replace('|', ' ').split()
  if len(parts) > 5:
    if parts[0] == 'the':
      value = ' '.join(parts[1:6])
    else:
      value = ' '.join(parts[:5])
  return value

def is_salient(speaker, sentence):
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

  for phrase in ['reference', 'postcode', 'thank', 'anything else', 'phone number']:
    if phrase in current.lower():
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

  return score >= 0.5

def filter_for_saliency(utterances):
  """ Input is a list of utterances where each utterance is a speaker + text
  The output is a list of utterances that only keeps the salient sentences
  """
  filtered = []

  for utterance in utterances:
    if utterance.startswith('<agent>'):
      speaker, text = utterance[:7], utterance[8:]
    elif utterance.startswith('<customer>'):
      speaker, text = utterance[:10], utterance[11:]
    else:
      raise ValueError(f"[{utterance}] does not contain a speaker")


    texts = [sentence for sentence in sent_tokenize(text) if is_salient(speaker, sentence)]
    if len(texts) > 0:  # there is at least one salient sentence in the utterance
      joined = ' '.join(texts)
      filtered.append(f"<{speaker}> {joined}")

  return filtered

def select_utterances(args, utt_so_far, target, split):
  domain, slot, value = target['domain'], target['slot'], target['value']
  domain, slot = standardize_format(domain, slot)
  target['domain'], target['slot'] = domain, slot

  use_target = True
  lookback = -args.context_length
  utterances = utt_so_far[lookback:]
  if args.filter and args.dataset == 'mwoz':
    utterances = filter_for_saliency(utterances)

  if args.task == 'in_context' and value == '<none>':  # TODO: query result none value slot
    use_target = False
  elif split == 'train' and value == '<none>' and random.random() < 0.8:
    use_target = False
  return use_target, utterances, target

def extract_label(targets, prior_values):
  # returns a list of (domain, slot, value) tuples when the domain is an active 
  swaps = {'not mentioned': '<none>', '': '<none>'}
  valid_domains = ['train', 'taxi', 'restaurant', 'hotel', 'attraction']
  labels = []
  
  for domain, domain_data in targets.items():
    if domain not in valid_domains:
      continue
    domain_data = targets[domain]
    active_domain = False

    for slot, value in domain_data['semi'].items():
      if len(value) > 0:
        active_domain = True

    if active_domain:
      for slot, value in domain_data['book'].items():
        if not isinstance(value, list) and slot != 'ticket':
          if value in swaps:
            value = swaps[value]
          if value in GENERAL_TYPO:
            value = GENERAL_TYPO[value]
          if value == '<none>' and prior_values[f'{domain}-{slot.lower()}'] != '<none>':
            value = '<remove>'
          labels.append((domain, slot.lower(), value))

      for slot, value in domain_data['semi'].items():
        if value in swaps:
          value = swaps[value]
        if value in GENERAL_TYPO:
          value = GENERAL_TYPO[value]
        value = normalize_length(value)
        if value == '<none>' and prior_values[f'{domain}-{slot.lower()}'] != '<none>':
          value = '<remove>'
        # if value == prior_values[ds] and value not in history:
        #   value = '<none>'
        labels.append((domain, slot.lower(), value))
  return labels

def extract_sgd(frames, domain_map, ontology):
  labels = defaultdict(dict)
  for frame in frames:
    service = frame["service"]
    domain = domain_map[service]

    if domain in ALL_SPLITS and 'state' in frame:
      for pslot in frame['state']['slot_values']:
        value = frame['state']['slot_values'][pslot][0]    # by default, select the first value
        if value.lower() in GENERAL_TYPO:
          value = GENERAL_TYPO[value.lower()]
        if len(value) < 28:
          slot = ontology[service][pslot]
          labels[domain][slot] = value
  return labels

def prepare_ontology(ontology):
  from nltk.stem import WordNetLemmatizer
  lemmatizer = WordNetLemmatizer()

  domain_map = {}
  temp_ont = defaultdict(set)

  for prev_domain, slot_dict in ontology.items():
    domain, number = prev_domain.split("_")
    post_domain = lemmatizer.lemmatize(domain.lower())
    if post_domain.endswith('ing'):
      post_domain = post_domain[:-3] + 'e'
    if post_domain.endswith('cars'):
      post_domain = post_domain[:-4]
    domain_map[prev_domain] = post_domain

    if post_domain in ALL_SPLITS:
      for slot in slot_dict.values():
        temp_ont[post_domain].add(slot)

  new_ont = {domain: list(slots) for domain, slots in temp_ont.items()}
  return domain_map, new_ont

def build_sgd(args, data, ontology, split):
  examples = {}
  domain_map, new_ont = prepare_ontology(ontology)

  for conversation in progress_bar(data, total=len(data)):
    convo_id = split + "-" + conversation['dialogue_id'].replace('_','-')
    examples[convo_id] = defaultdict(list)
    text_so_far = []

    prior_values = {f'{dom}-{slot}': '<none>' for dom, slots in new_ont.items() for slot in slots}
    turn_count = 1
    for turn in conversation['turns']:
      global_id = convo_id + '_' + str(turn_count)
      text = turn['utterance']

      if turn['speaker'] == 'SYSTEM':
        sys_text = f"<agent> {text}"
        text_so_far.append(sys_text)

      elif turn['speaker'] == 'USER':
        user_utt = f"<customer> {text}"
        text_so_far.append(user_utt)
        turn_count += 1

        targets = extract_sgd(turn['frames'], domain_map, ontology)
        prev_state = {k:v for k,v in prior_values.items()}

        for domain, slot_vals in targets.items():
          active_slots = new_ont[domain]

          for slot in active_slots:
            value = slot_vals.get(slot, "<none>")
            target = {'domain': domain, 'slot': slot, 'value': value.strip(), 'global_id': global_id}
            use_target, history, target = select_utterances(args, text_so_far, target, split)
            
            if use_target:
              example = {'utterances':history, 'target':target, 'prev_state':prev_state, 'corpus':'sgd'}
              examples[convo_id][global_id].append(example)
            if value != '<none>':
              prior_values[f'{domain}-{slot}'] = value
  return examples

def build_mwoz(args, data, ontology, split):
  # written for MultiWoz v2.0, 2.1 and 2.3
  examples = {}
  speakers = ["<customer>", "<agent>"]

  for convo_id, conversation in progress_bar(data.items(), total=len(data)):
    examples[convo_id] = defaultdict(list)
    text_so_far = []
    speaker_id = 0
    turn_count = 0

    prior_values = {f'{domain}-{slot}': '<none>' for domain, slots in ontology.items() for slot in slots}

    for turn in conversation['log']:
      text = turn['text']
      speaker = speakers[speaker_id]
      utterance = f"{speaker} {text}"

      if speaker == '<agent>':
        turn_count += 1
        global_id = f'{convo_id}_{turn_count}'
        targets = extract_label(turn['metadata'], prior_values)

        prev_state = {k:v for k,v in prior_values.items()}
        for domain, slot, value in targets: 
          target = {'domain': domain, 'slot': slot, 'value': value, 'global_id': global_id}
          use_target, history, target = select_utterances(args, text_so_far, target, split)
          if use_target:
            example = {'utterances':history, 'target':target, 'prev_state':prev_state, 'corpus':'mwoz'}
            examples[convo_id][global_id].append(example)
          pval = '<none>' if value == '<remove>' else value
          prior_values[f'{domain}-{slot}'] = pval
      
      text_so_far.append(utterance)  # add agent utterance afterwards
      speaker_id = 1 - speaker_id
  
  return examples

def build_mwoz22(args, data):
  ''' Written for raw v2.2 mwoz. This follows the schema format built by SGD'''
  examples = []
  speakers = {'user': '<customer>', 'system': '<agent>'}
  allowed_domains = list(DOMAIN_SLOTS_MWOZ.keys())

  for conversation in progress_bar(data, total=len(data)):
    text_so_far = []

    for turn in conversation['turns']:
      text = turn['utterance']
      speaker = speakers[turn['speaker'].lower()]
      utterance = f"{speaker} {text}"
      text_so_far.append(utterance)
      global_id = conversation['dialogue_id'] + '_' + turn['turn_id']

      if len(turn['frames']) > 0 and speaker == '<customer>':
        act_dom = [fr['service'] for fr in turn['frames'] if fr['state']['active_intent'] != "NONE"]
        
        for frame in turn['frames']:
          domain = frame['service']
          if domain in allowed_domains:

            slotvals = frame['state']['slot_values']
            if len(slotvals) > 0:
              active_slots = [domain_slot.split('-')[1] for domain_slot, _ in slotvals.items()]
              
              for slot in DOMAIN_SLOTS_MWOZ[domain]:
                if slot in active_slots:
                  domain_slot = '-'.join([domain, slot])
                  value = slotvals[domain_slot][0]
                else:
                  value = 'none'

                target = {'domain': domain, 'slot': slot, 'value': value, 'global_id': global_id}
                use_target, history, target = select_utterances(args, text_so_far, target, split)
                if use_target:
                  examples.append({'utterances': history, 'target': target, 'corpus': 'mwoz'})      
  return examples

def create_abcd_mappings(ontology):
  # given the child subflow, return the parent flow
  intent_map = {}
  for flow, subflows in ontology['intents'].items():
    for intent in subflows:
      intent_map[intent] = flow

  enumerable_values = ontology['values']['enumerable']

  # given the action (aka. slot), return whether it requires a value
  validity_map = {}
  for action, slots in ontology['actions']['has_slotval'].items():
    validity_map[action] = True
  for action in ontology['actions']['empty_slotval']:
    validity_map[action] = False

  has_slotvals = {}
  for category, actions in ontology['actions'].items():
    for action, slotvals in actions.items():
      if len(slotvals) > 0:  # the action contains slot values
        has_slotvals[action] = slotvals

  mappings = {'intent': intent_map, 'enum': enumerable_values, 
              'valid': validity_map, 'valid_actions': has_slotvals }
  return mappings

def select_abcd_utterances(utt_so_far, target, prev_state, split):
  domain, slot, value = target['domain'], target['slot'], target['value']
  domain, slot = standardize_format(domain, slot)
  target['domain'], target['slot'] = domain, slot
  utterances = utt_so_far[-3:]
  previous = [val.lower() for slot, val in prev_state.items() if val != '<none>']
  
  use_target = False
  if value.lower() in ''.join(utterances).lower() or value in previous:
    use_target = True
  if value == '<none>' and random.random() < 0.04:
    use_target = True
  return use_target, utterances, target

def is_slotval(scenario, mappings, cand_slot, cand_val):
  if cand_slot in mappings['enumerable_values']:
    cand_values = [cv.lower() for cv in mappings['enumerable_values'][cand_slot]]
    if cand_val.lower() in cand_values:
      return True
  else:
    for category, scene in scenario.items():
      for slot, value in scene.items():
        if cand_slot == slot and cand_val == value:
          return True
  # if we find no matches, then this candidate value is not valid
  return False

def make_dialogue_state_from_scratch(intent, action, values, scene, mappings):
  targets = []
  target_domains = set()

  if mappings['valid'][action]:
    for value in values:
      cand_val = value.lower().strip() 
      candidate_slots = mappings['valid_actions'][action]
   
      domain = mappings['intent'][intent].replace('_', ' ')
      target = { 'domain': domain }
      target_domains.add(domain)
      if len(candidate_slots) == 1:
        target['slot'] = candidate_slots[0].replace('_', ' ')
        target['value'] = cand_val
        targets.append(target)
      else:
        for cand_slot in candidate_slots:
          scenario = {k: v for k, v in scene.items() if not k.endswith('flow')}
          if is_slotval(scenario, mappings, cand_slot, cand_val):
            target['slot'] = cand_slot.replace('_', ' ')
            target['value'] = cand_val
            targets.append(target)
            break

  return targets, target_domains

def make_dialogue_state(action, values, scene, mappings):
  # flow --> domain; subflow --> intent; action --> slot; slotval --> value
  slot_state = {}
  if action in mappings['valid_actions']:    #  if this is a valid action that requires a slot-val
    for value in values:
      cand_val = value.strip() 
      candidate_slots = mappings['valid_actions'][action]
   
      if len(candidate_slots) == 1:
        slot = candidate_slots[0].replace('_', ' ')
        slot_state[slot] = cand_val
      else:
        for cand_slot in candidate_slots:
          scenario = {k: v for k, v in scene.items() if not k.endswith('flow')}
          if is_slotval(scenario, mappings, cand_slot, cand_val):
            slot = cand_slot.replace('_', ' ')
            if slot == 'account id':
              cand_val = cand_val.upper()
            slot_state[slot] = cand_val
            break

  return slot_state

def build_abcd(args, data, mappings, split):
  examples = {}
  ontology = mappings['domain_slots']

  subflow_to_flow = {}
  for flow, subflows in mappings['intents'].items():
    for subflow in subflows:
      domain = flow.replace('_', ' ')
      subflow_to_flow[subflow] = domain

  for convo in progress_bar(data, total=len(data)):
    # each convo has keys: convo_id, scene, conversation
    convo_id = str(convo['convo_id'])
    examples[convo_id] = defaultdict(list)
    utt_so_far = []
    action_count = 1

    prior_values = {f'{domain}-{slot}': '<none>' for domain, slots in ontology.items() for slot in slots}
    for turn in convo['conversation']:
      # each turn has keys: speaker, text, targets, turn_count, candidates
      speaker = turn['speaker']

      if speaker == 'action':  # skip action turns
        global_id = f"{convo_id}_{action_count}"
        # each target is a 5-part list: intent, nextstep, action, value, utt_rank
        intent, nextstep, action, values, utt_rank = turn['targets']
        domain = subflow_to_flow[intent]  # intent is the subflow

        prev_state = {k:v for k,v in prior_values.items()}
        slot_state = make_dialogue_state(action, values, convo['scene'], mappings)

        if len(slot_state) == 0:  # no valid slot-vals this turn 
          continue
        action_count += 1

        for slot in ontology[domain]:
          curr_value = slot_state.get(slot, '<none>')
          prev_value = prev_state[f"{domain}-{slot}"]
          if prev_value != '<none>' and curr_value == '<none>':
            value = prev_value
          else:
            value = curr_value

          target = {'domain': domain, 'slot': slot, 'value': value, 'global_id': global_id}
          use_target, history, target = select_abcd_utterances(utt_so_far, target, prev_state, split)
          if use_target:
            example = {'utterances':history, 'target':target, 'prev_state':prev_state, 'corpus':'abcd'}
            examples[convo_id][global_id].append(example)
          if value != "<none>":
            prior_values[f'{domain}-{slot}'] = value
      else:
        text = turn['text']
        utt_so_far.append(f"<{speaker}> {text}")

  return examples

def build_dstc(args, data, ontology, split):
  examples = {}

  for convo in progress_bar(data, total=len(data)):
    convo_id = convo['guid'].replace('_', '-')
    examples[convo_id] = defaultdict(list)
    text_so_far = []

    # there is one domain in dstc
    prior_values = {f'{domain}-{slot}': '<none>' for domain, slots in ontology.items() for slot in slots}
    for turn in convo['conversation']:
      global_id = convo_id + '_' + str(turn['turn'])

      if turn['speaker'] == 'agent':
        sys_text = f"<agent> {turn['text']}"
        text_so_far.append(sys_text)
  
      elif turn['speaker'] == 'user':
        user_text = f"<customer> {turn['text']}"
        text_so_far.append(user_text)

        prev_state = {k:v for k,v in prior_values.items()}
        for slot in ontology['restaurant']:
          value = turn['inform'].get(slot, "<none>")
          if value in GENERAL_TYPO:
            value = GENERAL_TYPO[value]

          target = {'domain': 'restaurant', 'slot': slot, 'value': value, 'global_id': global_id}
          use_target, history, target = select_utterances(args, text_so_far, target, split)
          if use_target:
            example = {'utterances':history, 'target':target, 'prev_state':prev_state, 'corpus':'dstc'}
            examples[convo_id][global_id].append(example)

          if value != "<none>":
            prior_values[f'restaurant-{slot}'] = value
  
  return examples

def build_gsim(args, data, ontology, split):
  examples = {}

  for conversation in progress_bar(data, total=len(data)):
    convo_id = conversation['dialogue_id']
    examples[convo_id] = defaultdict(list)
    domain = convo_id.split('_')[0]
    text_so_far = []    

    prior_values = {f'{domain}-{slot}': '<none>' for domain, slots in ontology.items() for slot in slots}
    for turn_count, turn in enumerate(conversation['turns']):
      global_id = convo_id.replace("_","-") + '_' + str(turn_count + 1)
      if 'system_utterance' in turn:
        sys_text = turn['system_utterance']['text']
        sys_utt = f"<agent> {sys_text}"
        text_so_far.append(sys_utt)

      user_text = turn['user_utterance']['text']
      user_utt = f"<customer> {user_text}"
      text_so_far.append(user_utt)

      prev_state = {k:v for k,v in prior_values.items()}
      current_slots_tmp = {slot["slot"]:slot["value"] for slot in turn["dialogue_state"]}
      for slot in ontology[domain]:
        value = current_slots_tmp.get(slot, "<none>")
        if value in GENERAL_TYPO:
          value = GENERAL_TYPO[value]
        target = {'domain': domain, 'slot': slot, 'value': value, 'global_id': global_id}
        use_target, history, target = select_utterances(args, text_so_far, target, split)
        if use_target:
          example = {'utterances':history, 'target':target, 'prev_state':prev_state, 'corpus':'gsim'}
          examples[convo_id][global_id].append(example)
        if value != "<none>":
          prior_values[f'{domain}-{slot}'] = value

  return examples

def build_tt(args, data, ontology, split):
  examples = {}
  domain = 'movie'
  sc = Counter()
  for convo in progress_bar(data, total=len(data)):
    convo_id = convo['conversation_id'].replace('_', '-')
    examples[convo_id] = defaultdict(list)
    text_so_far = []    

    prior_values = {f'{domain}-{slot}': '<none>' for domain, slots in ontology["entities"].items() for slot in slots}
    turn_count = 1
    for turn in convo['utterances']:
      global_id = convo_id + '_' + str(turn_count)

      text = turn['text']

      if turn['speaker'] == 'assistant':
        sys_utterance = f"<agent> {text}"
        text_so_far.append(sys_utterance)

      elif turn['speaker'] == 'user':
        user_utterance = f"<customer> {text}"
        text_so_far.append(user_utterance)
        turn_count += 1

        prev_state = {k:v for k,v in prior_values.items()}
        if 'segments' in turn:
          labels = extract_slotvals(turn['segments'], ontology['slotvals'])
        else:
          continue   # no valid labels this turn
        if len(labels) == 0:
          continue

        for slot in ontology["slotvals"].values():
          value = labels.get(slot, "<none>")
          if value == '<none>':
            if split == 'train':
              if random.random() > 0.5: continue
            else:
              if random.random() > 0.1: continue

          target = {'domain': 'movies', 'slot': slot, 'value': value, 'global_id': global_id}
          use_target, history, target = select_utterances(args, text_so_far, target, split)
          if use_target:
            example = {'utterances':history, 'target':target, 'prev_state':prev_state, 'corpus':'tt'}
            examples[convo_id][global_id].append(example)
          prior_values[f'{domain}-{slot}'] = value
  
  return examples

def extract_slotvals(segments, ontology):
  labels = {}
  for segment in segments:
    slot_candidate = segment['annotations'][0]['name']
    value = segment['text'].replace('!', '').replace('.', '').replace('?', '')
    if value in GENERAL_TYPO:
      value = GENERAL_TYPO[value]
    if slot_candidate == "name.theater" and value.lower().endswith("theater"):
      value = value[:-7].strip()
    if slot_candidate in ontology and len(value) < 28:
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
    examples = build_abcd(args, data, ontology, split) 
  elif args.dataset == 'dstc':  # State Tracking Challenge 2
    examples = build_dstc(args, data, ontology, split) 
  elif args.dataset == 'gsim':    # Google Simulated Chats
    examples = build_gsim(args, data, ontology, split) 
  elif args.dataset.startswith('mwoz'):  # MultiWoz 2.1 or 2.2
    examples = build_mwoz(args, data, ontology, split)
  elif args.dataset == 'sgd':   # Schema Guided Dialogue
    examples = build_sgd(args, data, ontology, split) 
  elif args.dataset == 'tt':    # TicketTalk / TaskMaster 3
    examples = build_tt(args, data, ontology, split) 

  return examples

def process_data(args, raw_data, tokenizer):
  ontology = raw_data['ontology']

  cache_results, already_exist = check_cache(args)
  if already_exist:
    datasets = cache_results
  else:
    datasets = {}
    for split in ['train', 'dev', 'test']:
      examples = prepare_examples(args, raw_data[split], ontology, split)
      if args.task == 'meta_learn':
        datasets[split] = MetaLearnDataset(args, examples, tokenizer, split)
      elif args.task == 'in_context':
        datasets[split] = InContextDataset(args, examples, tokenizer, split)
      elif args.task == 'fine_tune':
        datasets[split] = FineTuneDataset(args, examples, tokenizer, split)
      print(f"Running with {len(datasets[split])} {split} examples")
    pkl.dump(datasets, open(cache_results, 'wb'))

  return datasets, ontology
