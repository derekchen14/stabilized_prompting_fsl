import os, pdb, sys
import random
import json
import math
import pickle as pkl
import numpy as np

from assets.static_vars import device, DATASETS, GENERAL_TYPO, DOMAIN_SLOTS
from utils.prompt import find_prompt
from components.datasets import MetaLearnDataset, InContextDataset, FineTuneDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm as progress_bar
from collections import defaultdict
from utils.trade_utils import prepare_data_seq

def check_cache(args):
  cache_file = f'{args.task}_{args.style}.pkl'
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

def add_speakers(conversation, first_speaker='<customer>'):
  """ takes in a list of utterances and adds the speakers in list form """
  speakers = ['<agent>', '<customer>']
  s_index = speakers.index(first_speaker)

  dialogue = []
  for turn in conversation:
    new_turn = f"{speakers[s_index]} {turn}"
    dialogue.append(new_turn)
    s_index = 1 - s_index
  return dialogue

def shift_tokens_right(targets, pad_token_id):
  # Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).  
  labels = targets.input_ids.clone()
  labels.masked_fill_(labels == pad_token_id, -100)

  output_tokens = targets.input_ids.clone()
  index_of_eos = (output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
  decoder_start_tokens = output_tokens.gather(1, index_of_eos).squeeze()
  output_tokens[:, 1:] = output_tokens[:, :-1].clone()
  output_tokens[:, 0] = decoder_start_tokens
  targets['input_ids'] = output_tokens

  return targets, labels

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

def meta_learn_mwoz(args, data, label_set):
  # written for raw v2.4 mwoz
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
          examples.append({'dialogue': context + '<label>',
                             'prompt': make_prompt(args.prompt_style, domain, slot),
                              'label': value})
          # for domain in domains:
          #   for slot in domain:
          #     example['setting'] = slot_domain
          # possible_values = '\nOptions: ' + ', '.join(label_set)  # candidate values
          # example['options'] = possible_values

      text_so_far.append(utterance)  # add agent utterance afterwards
      
      speaker_id = 1 - speaker_id
      if len(text_so_far) > args.context_len:
        text_so_far = text_so_far[-args.context_len:]

  return examples

def build_mwoz(args, data, label_set):
  """ All models are default MWOZ 2.2 which conforms to SGD format"""
  if args.task == 'meta_learn':
    return meta_learn_mwoz(args, data, label_set)
  elif args.task == 'fine_tune':
    return fine_tune_mwoz(args, data, label_set)
  elif args.task == 'in_context':
    return in_context_mwoz(args, data, label_set)
  elif args.do_interact:
    mapping = {label: idx for idx, label in enumerate(label_set)}
    return interact_mwoz(args, mapping)

def fine_tune_mwoz21(args, data, label_set):
  ''' Written for raw v2.1 mwoz.  Since evaluation is done by a library
  based on the dialog_id, we will additionally pass some extra meta data along
  with the ground truth label for evaluation, which includes dialog_id '''
  examples = []
  allowed_domains = list(DOMAIN_SLOTS.keys())

  for dial_id in progress_bar(data, total=len(data)):
    text_so_far = []

    for turn in data[dial_id]:
      # context
      context = turn['context']
      if len(context.split("<system>")) > args.context_len:
        context = "<system>".join(context.split("<system>")[-args.context_len:])

      # construct extra information, which is a structured dict
      target = { 'convo_id': dial_id.split('.')[0].lower(),  # drop the ".json"
              'turn_count': int(turn['turn_num']) }

      # building slot dict
      slot_dict = {}
      for slot in turn["slots_inf"].split(" , ")[:-1]:
        dom, slot_type, slot_value = slot.split()[0], slot.split()[1], " ".join(slot.split()[2:])
        if dom not in slot_dict:
          slot_dict[dom] = {}
        slot_dict[dom][slot_type] = slot_value

      for domain in turn["potential_domains"]:
        if domain not in allowed_domains:
          continue
        for slot_type in DOMAIN_SLOTS[domain]:
          prompt = find_prompt(args.prompt_style, domain, slot_type)
          if domain in slot_dict and slot_type in slot_dict[domain]:
            slot_value = slot_dict[domain][slot_type]
          else:
            slot_value = 'none'
          target['domain'] = domain
          target['slot'] = slot_type
          target['value'] = slot_value
          examples.append({'context': context, 'label': slot_value, 'target': target})
      
  return examples

def fine_tune_mwoz(args, data, label_set):
  ''' Written for raw v2.2 mwoz.  Since evaluation is done by a library
  based on the dialog_id, we will additionally pass some extra meta data along
  with the ground truth label for evaluation, which includes dialog_id '''
  examples = []
  speakers = {'USER': '<customer>', 'SYSTEM': '<agent>'}
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
        target = {
          'convo_id': conversation['dialogue_id'].split('.')[0].lower(),  # drop the ".json"
          'turn_count': int(turn['turn_id']) }
        
        for frame in turn['frames']:
          current_domain = frame['service']
          if current_domain in allowed_domains:

            slotvals = frame['state']['slot_values']
            if len(slotvals) > 0:
              active_slots = [domain_slot.split('-')[1] for domain_slot, _ in slotvals.items()]
              
              for slot in DOMAIN_SLOTS[current_domain]:
                # prompt = find_prompt(args.prompt_style, current_domain, slot)
                if slot in active_slots:
                  domain_slot = '-'.join([current_domain, slot])
                  value = slotvals[domain_slot][0]
                else:
                  value = 'none'

                context = ' '.join(text_so_far)
                target['domain'] = current_domain
                target['slot'] = slot
                target['value'] = value
                examples.append({'dialogue': context, 'label': value, 'target': target})
      
      if len(text_so_far) > args.context_len:
        text_so_far = text_so_far[-args.context_len:]

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
        text_so_far.append(user_text)
        context = ' '.join(text_so_far)

        for slot, value in turn['inform'].items():
          # TODO: add negatives to predict "none"
          target['slot'] = slot
          target['value'] = value
          examples.append({'dialogue': context, 'label': value, 'target': target})

      if len(text_so_far) > 10:
        text_so_far = text_so_far[-10:]

  return examples

def build_sgd(args, data, mapping, split):
  examples = []

  for conversation in progress_bar(data, total=len(data)):
    text_so_far = []    

    for turn in conversation['turns']:    
      text = turn['utterance']
      turn_count = len(text_so_far) + 1
      target = {'global_id': conversation['dialogue_id'] + '_' + str(turn_count) }

      if turn['speaker'] == 'SYSTEM':
        sys_utt = f"<agent> {text}"
        text_so_far.append(sys_utt)
  
      elif turn['speaker'] == 'USER':
        user_utt = f"<customer> {text}"
        text_so_far.append(user_utt)
        context = ' '.join(text_so_far)

        labels = extract_frame(turn)
        """labels with number of keys equal to the number of services found in that turn
        each of these will be turned into a training example
        the targets of each training example has keys: intents, requests, slots, values
        each of the four keys is a list containing the respectives items as strings """

        for service, details in labels.items():
          dialogue = context + f' <service> {service} <sep>'
          fls = details['flattened']  # labels as a long, flattened string, split by service
          sls = details['structured'] # labels as a dictionary, again split by service
          if len(sls['intents']) > 0 and len(sls['slots']) > 0:
            prompt = "The {slot} for {service} is"
            target['domain'] = service
            target['slot'] = slot
            target['value'] = fls
            examples.append({'dialogue': context, 'label': fls, 'extra': target})
      if len(text_so_far) > args.context_len:
        text_so_far = text_so_far[-args.context_len:]

  return examples

def extract_frame(turn):
  labels = {}

  for frame in turn['frames']:
    service = frame['service']
    targets = []
    labels[service] = { 'structured': defaultdict(list) }

    if 'state' in frame:
      active_intent = frame['state']['active_intent']
      labels[service]['structured']['intents'].append(active_intent)
      
      requested = frame['state']['requested_slots']
      labels[service]['structured']['requests'].extend(requested)
      for req in requested:
        req_target = f"{active_intent}(request={req})"
        targets.append(req_target)

      for slot, value in frame['state']['slot_values'].items():
        val = value[0]
        labels[service]['structured']['slots'].append(slot)
        labels[service]['structured']['values'].append(val)
        inf_target = f"{active_intent}({slot}={val})"
        targets.append(inf_target)

    labels[service]['flattened'] = ';'.join(targets)
  return labels

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
        text_so_far.append(user_utterance)
        context = ' '.join(text_so_far)

        if 'segments' in turn:
          labels = extract_slotvals(turn['segments'], ontology)
          for slot, value in labels.items():
            target = {'domain': 'movies', 'slot': slot, 'value': value}
            examples.append({'dialogue': context, 'label': value, 'target': target})

      if len(text_so_far) > args.context_len:
        text_so_far = text_so_far[-args.context_len:]

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

def create_abcd_mappings(ontology):
  intent_map = {}
  for flow, subflows in ontology['intents'].items():
    for intent in subflows:
      intent_map[intent] = flow

  enumerate_map = {}
  for slot, values in in ontology['values']['enumerable'].items():
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

def make_dialogue_state(intent, action, value, ontology, mappings):
  target = {}
  valid = False
  valid_actions = ontology['actions']['has_slotval']

  if mappings['valid'][action]:
    valid = True
    candidate_slots = valid_actions[action]

    target['domain'] = mappings['intent'][intent]
    if len(candidate_slots) == 1:
      
      target['slot'] = candidate_slots[0]
    else:
      for cand_slot in candidate_slots:
        pass

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
        target, valid = make_dialogue_state(intent, action, value, ontology, mappings)
        target['global_id'] = convo['convo_id'] + '_' + turn['turn_count']
  
        if valid:
          context = ' '.join(utt_so_far)
          example = {'dialogue': context, 'label': value, 'target': target}
          examples.append(example)  
      else:
        text = turn['text']
        utt_so_far.append(f"<{speaker}> {text}")

    if len(utt_so_far) > args.context_len:
      utt_so_far = utt_so_far[-args.context_len:]

  print("runs correctly")
  sys.exit()
  return examples

def build_gsim(args, data, mapping):
  examples = []

  for conversation in progress_bar(data, total=len(data)):
    text_so_far = []    
    convo_id = conversation['dialogue_id']
    domain = convo_id.split('_')[0]
    extra = { 'convo_id': convo_id, 'domain': domain }

    for turn in conversation['turns']:

      if 'system_utterance' in turn:
        sys_text = turn['system_utterance']['text']
        sys_utt = f"<agent> {sys_text}"
        text_so_far.append(sys_utt)

      user_text = turn['user_utterance']['text']
      user_utt = f"<customer> {user_text}"
      text_so_far.append(user_utt)
      context = ' '.join(text_so_far)

      for state in turn['dialogue_state']:
        extra['slot'] = state['slot']
        extra['value'] = state['value']
        examples.append({'dialogue': context, 'label': state['value'], 'target': extra})  

  return examples

def get_dataloader(args, dataset, split='train'):
  if args.model == 'trade':
    return dataset
  sampler = RandomSampler(dataset) if dataset.shuffle else SequentialSampler(dataset)
  collate = dataset.collate_func
  dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)
  print(f"Loaded {split} data with {len(dataloader)} batches")
  return dataloader


def prepare_examples(args, data, ontology, split):
  """ Each example is a dict which should have:
    dialogue: the context utterances as input with speakers of <agent> and <customer>
    label: the text value to predict in string format
    target: a dictionary with keys global_id, domain, slot and value
  """
  if args.dataset == 'abcd':    # Action Based Conversations
    examples = build_abcd(args, data, ontology) 
  elif args.dataset == 'dstc':  # State Tracking Challenge 2
    examples = build_dstc(args, data) 
  elif args.dataset == 'gsim':  # Google Simulated Chats
    examples = build_gsim(args, data) 
  elif args.dataset == 'mwoz':  # MultiWoz 2.2
    examples = build_mwoz(args, data, ontology)
  elif args.dataset == 'sgd':   # Schema Guided Dialogue
    examples = build_sgd(args, data, ontology, split) 
  elif args.dataset == 'tt':    # TicketTalk / TaskMaster 3
    examples = build_tt(args, data, ontology) 

  return examples

def hold_out(args, datasets):
  if args.model == "trade":
    return datasets
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
  label_set = raw_data['ontology']

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
    if args.model == 'trade':
      train, dev, test = prepare_data_seq(args, tokenizer=False)
      datasets = {
        "train": train,
        "dev"  : dev,
        "test" : test,
      }
    pkl.dump(datasets, open(cache_results, 'wb'))

  datasets = hold_out(args, datasets)
  return datasets, label_set
