import os, pdb, sys
import random
import json
import math
import pickle as pkl
import numpy as np

from assets.static_vars import *
from utils.help import standardize_format
from components.datasets import MetaLearnDataset, InContextDataset, FineTuneDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm as progress_bar
from collections import defaultdict, Counter

def check_cache(args):
  cache_file = f'{args.model}_{args.task}_{args.prompt_style}_lookback{args.context_length}.pkl'
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
        if value == '<none>' and prior_values[f'{domain}-{slot.lower()}'] != '<none>':
          value = '<remove>'
        labels.append((domain, slot.lower(), value))

  return labels

def select_utterances(args, utt_so_far, target, split):
  domain, slot, value = target['domain'], target['slot'], target['value']
  domain, slot = standardize_format(domain, slot)
  target['domain'], target['slot'] = domain, slot

  use_target = True
  lookback = -args.context_length
  utterances = utt_so_far[lookback:]

  if args.task == 'in_context' and value == '<none>':  # TODO: query result none value slot
    use_target = False
  elif split == 'train' and value == '<none>' and random.random() < 0.8:
    use_target = False
  return use_target, utterances, target

  # if args.context_length < 0:
  #   return utt_so_far, True
  # if args.context_length % 2 == 0: # drop the agent utterances
  #   lookback = -args.context_length - 1
  #   utterances = [utt for utt in utt_so_far[lookback:] if utt.startswith('<customer>')]
  # history = ' '.join(utterances)
  # if value in history.lower() or value in ['<remove>', 'any']:
  #   use_target = True
  # elif num_in_history(value, history.lower()):
  #   use_target = True
  # elif value.lower() in ['yes', 'no'] and (slot in history or 'wifi' in history):
  #   use_target = True  # to handle the internet and parking use cases

def extract_label_sgd(frames, prior_values):
  labels = []
  for frame in frames:
    service = frame["service"]
    if 'state' in frame:
      for slot in frame['state']['slot_values']:
        value = frame['state']['slot_values'][slot][0]    #by default, select the first value
        if value == prior_values[f'{service}-{slot}']:
          continue
        labels.append((service, slot, value))
  return labels

def build_sgd(args, data, ontology, split):
  examples = []
  prompt = "The topic of conversation is about"

  for conversation in progress_bar(data, total=len(data)):
    text_so_far = []

    prior_values = {f'{service}-{slot}': '<none>' for service, slots in ontology.items() for slot in slots}

    for turn_count, turn in enumerate(conversation['turns']):
      text = turn['utterance']

      if turn['speaker'] == 'SYSTEM':
        sys_text = f"<agent> {text}"
        text_so_far.append(sys_text)

      elif turn['speaker'] == 'USER':
        user_utt = f"<customer> {text}"
        text_so_far.append(user_utt)

        targets = extract_label_sgd(turn['frames'], prior_values)
        prev_state = {k:v for k,v in prior_values.items()}
        for service, slot, value in targets:
          target = {'domain': service, 'slot': slot, 'value': value.strip(),
                'global_id': conversation['dialogue_id'].replace('_','-') + '_' + str(turn_count+1) }
          use_target, history, target = select_utterances(args, text_so_far, target, split)
          if use_target:
            examples.append({'utterances': history, 'target': target, 'prev_state':prev_state})
          pval = '<none>' if value == '<remove>' else value
          prior_values[f'{service}-{slot}'] = pval

  return examples

def build_mwoz(args, data, ontology, split):
  # written for MultiWoz v2.0, 2.1 and 2.3
  examples = []
  speakers = ["<customer>", "<agent>"]
  for convo_id, conversation in progress_bar(data.items(), total=len(data)):
    text_so_far = []
    speaker_id = 0
    turn_count = 0

    prior_values = {f'{domain}-{slot}': '<none>' for domain, slots in ontology.items() for slot in slots}
    for turn in conversation['log']:
      turn_count += 1
      text = turn['text']
      speaker = speakers[speaker_id]
      utterance = f"{speaker} {text}"
      
      if speaker == '<agent>':
        targets = extract_label(turn['metadata'], prior_values)
        prev_state = {k:v for k,v in prior_values.items()}
        for domain, slot, value in targets: 
          target = {'domain': domain, 'slot': slot, 'value': value,
              'global_id': f'{convo_id}_{turn_count}' }
          use_target, utterances, target = select_utterances(args, text_so_far, target, split)
          if use_target:
            examples.append({'utterances': utterances, 'target': target, 'prev_state': prev_state})
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
      
      if len(turn['frames']) > 0 and speaker == '<customer>':
        act_dom = [fr['service'] for fr in turn['frames'] if fr['state']['active_intent'] != "NONE"]
        
        for frame in turn['frames']:
          current_domain = frame['service']
          if current_domain in allowed_domains:

            slotvals = frame['state']['slot_values']
            if len(slotvals) > 0:
              active_slots = [domain_slot.split('-')[1] for domain_slot, _ in slotvals.items()]
              
              for slot in DOMAIN_SLOTS_MWOZ[current_domain]:
                if slot in active_slots:
                  domain_slot = '-'.join([current_domain, slot])
                  value = slotvals[domain_slot][0]
                else:
                  value = 'none'

                target = {'domain': current_domain, 'slot': slot, 'value': value,
                        'global_id': conversation['dialogue_id'] + '_' + turn['turn_id'] }
                use_target, history, target = select_utterances(args, text_so_far, target, split)
                if use_target:
                  examples.append({'utterances': history, 'target': target})      
  return examples

def create_abcd_mappings(ontology):
  intent_map = {}
  for flow, subflows in ontology['intents'].items():
    for intent in subflows:
      intent_map[intent] = flow

  enumerable_values = ontology['values']['enumerable']

  validity_map = {}
  for action, slots in ontology['actions']['has_slotval'].items():
    validity_map[action] = True
  for action in ontology['actions']['empty_slotval']:
    validity_map[action] = False

  mappings = {'intent': intent_map, 'enum': enumerable_values, 'valid': validity_map}
  return mappings

def is_slotval(scenario, mappings, cand_slot, cand_val):
  if cand_slot in mappings['enum']:
    cand_values = mappings['enum'][cand_slot]
    if cand_val in cand_values:
      return True
  else:
    for category, scene in scenario.items():
      for slot, value in scene.items():
        if cand_slot == slot and cand_val == value:
          return True
  # if we find no matches, then this candidate value is not valid
  return False

def make_dialogue_state(intent, action, values, scene, mappings):
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

def build_abcd(args, data, ontology, split):
  examples = []
  mappings = create_abcd_mappings(ontology)
  mappings['valid_actions'] = ontology['actions']['has_slotval']

  for convo in progress_bar(data, total=len(data)):
    # each convo has keys: convo_id, scene, conversation
    utt_so_far = []

    prior_values = {f'{domain}-{slot}': '<none>' for domain, slots in ontology.items() for slot in slots}
    for turn in convo['conversation']:
      # each turn has keys: speaker, text, targets, turn_count, candidates
      speaker = turn['speaker']

      if speaker == 'action':  # skip action turns
        intent, nextstep, action, values, utt_rank = turn['targets']
        # each target is a 5-part list: intent, nextstep, action, value, utt_rank
        targets, target_domains = make_dialogue_state(intent, action, values, convo['scene'], mappings)
  
        prev_state = {k:v for k,v in prior_values.items()}
        current_slots_tmp = {slot["domain"]+"-"+slot["slot"]:slot["value"] for slot in targets}
        for domain in target_domains:
          for slot in ontology[domain]:
            value = current_slots_tmp.get(f"{domain}-{slot}", "<none>")
            target = {'domain': 'restaurant', 'slot': slot, 'value': value,
                'global_id': str(convo['convo_id']) + '_' + str(turn['turn_count']) }
            use_target, history, target = select_utterances(args, utt_so_far, target, split)
            if use_target:
              examples.append({'utterances': history, 'target': target, 'prev_state':prev_state})
            if value != "<none>":
              prior_values[f'{domain}-{slot}'] = value

      else:
        text = turn['text']
        utt_so_far.append(f"<{speaker}> {text}")

  return examples

def build_dstc(args, data, ontology, split):
  ''' extra contains the structured label as a value '''
  examples = []

  for convo in progress_bar(data, total=len(data)):
    text_so_far = []

    # there is one domain in dstc
    prior_values = {f'{domain}-{slot}': '<none>' for domain, slots in ontology.items() for slot in slots}
    for turn in convo['conversation']:
      target = {
        'global_id': convo['guid'].replace('_', '-') + '_' + str(turn['turn']),
        'domain': 'restaurant' }

      if turn['speaker'] == 'agent':
        sys_text = f"<agent> {turn['text']}"
        text_so_far.append(sys_text)
  
      elif turn['speaker'] == 'user':
        user_text = f"<customer> {turn['text']}"
        text_so_far.append(user_text)

        prev_state = {k:v for k,v in prior_values.items()}
        for slot in ontology['restaurant']:
          value = turn['inform'].get(slot, "<none>")
          target = {'domain': 'restaurant', 'slot': slot, 'value': value,
              'global_id': convo['guid'].replace('_', '-') + '_' + str(turn['turn']) }
          use_target, history, target = select_utterances(args, text_so_far, target, split)
          if use_target:
            examples.append({'utterances': history, 'target': target, 'prev_state':prev_state})

          if value != "<none>":
            prior_values[f'restaurant-{slot}'] = value
  
  return examples

def build_gsim(args, data, ontology, split):
  examples = []

  for conversation in progress_bar(data, total=len(data)):
    dialog_id = conversation['dialogue_id']
    domain = dialog_id.split('_')[0]
    text_so_far = []    

    prior_values = {f'{domain}-{slot}': '<none>' for domain, slots in ontology.items() for slot in slots}
    for turn_count, turn in enumerate(conversation['turns']):
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
        target = {'domain': domain, 
                    'slot': slot,
                   'value': value,  
               'global_id': dialog_id + '_' + str(turn_count + 1) }
        use_target, history, target = select_utterances(args, text_so_far, target, split)
        if use_target:
          examples.append({'utterances': history, 'target': target, 'prev_state':prev_state})
        if value != "<none>":
          prior_values[f'{domain}-{slot}'] = value

  return examples

def build_tt(args, data, ontology, split):
  examples = []
  domain = 'movie'
  for convo in progress_bar(data, total=len(data)):  
    text_so_far = []    

    prior_values = {f'{domain}-{slot}': '<none>' for domain, slots in ontology.items() for slot in slots}
    for turn in convo['utterances']:
      text = turn['text']

      if turn['speaker'] == 'assistant':
        sys_utterance = f"<agent> {text}"
        text_so_far.append(sys_utterance)

      elif turn['speaker'] == 'user':
        user_utterance = f"<customer> {text}"
        text_so_far.append(user_utterance)

        prev_state = {k:v for k,v in prior_values.items()}
        if 'segments' in turn:
          labels = extract_slotvals(turn['segments'], ontology['slotvals'])
        else:
          labels = {}

        for slot in ontology['movie']:
          value = labels.get(slot, "<none>")
          target = {'domain': 'movies', 'slot': slot, 'value': value,
          'global_id': convo['conversation_id'].replace('_', '-') + '_' + str(turn['index'])}
          use_target, history, target = select_utterances(args, text_so_far, target, split)
          if use_target:
            examples.append({'utterances': history, 'target': target, 'prev_state':prev_state})
          prior_values[f'{domain}-{slot}'] = value
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

  datasets = hold_out(args, datasets)
  return datasets, ontology
