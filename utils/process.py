import os, pdb, sys
import random
import json
import math
import pickle as pkl
import numpy as np

from assets.static_vars import device, DATASETS, GENERAL_TYPO
from utils.prompt import make_prompt
from components.datasets import MetaLearnDataset, InContextDataset, FineTuneDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm as progress_bar
from collections import defaultdict

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

def extract_act(user_acts, mapping):
  act_list = [ua['type'] for ua in user_acts]
  act_list.sort()
  size = len(act_list)

  if size == 0:
    act = 'NONE'
  elif size == 1:
    act = act_list[0]
  else:
    if act_list[0] == "AFFIRM":
      act = "AFFIRM"
    elif "GOOD_BYE" in act_list and "THANK_YOU" in act_list:
      act = "THANKS_GOODBYE"
    elif "NEGATE" in act_list:
      act = "NEGATE"
    else:
      act = ' '.join(act_list)
  if act == "GREETING INFORM":
    act = "GREET_INFORM"

  act_id = mapping[act]  # does implicit assertion since invalid acts won't be in the ontology
  return act_id

def abcd_retrieval(convo, examples):
  text_so_far = []
  for turn in convo['conversation']:

    if turn['targets'][1] == 'retrieve_utterance':  # otherwise is user_turn or action
      context = ' '.join(text_so_far)
      position = turn['targets'][4]  # target position of the utterance
      assert(position >= 0)
      candidates = turn['candidates']

      example = {'dialogue': context, 'position': position, 'candidates': candidates}
      examples.append(example)

    text_so_far.append(turn['text'])
  return examples

def abcd_classification(convo, mapping):
  intent = convo['conversation'][0]['targets'][0]  # 0 is the intent/subflow
  label_id = mapping[intent]

  dialogue = []
  for turn in convo['conversation']:
    speaker = turn['speaker']
    if speaker == 'action':  # skip action turns
      if len(dialogue) > 3:  # complete if at least 3 turns
        break
      else:
        continue

    text = turn['text']
    dialogue.append(f"{speaker} {text}")

  dialog_string = ' '.join(dialogue)
  return {'dialogue': dialog_string, 'label': label_id}

def build_abcd(args, data, mapping):
  examples = []
  for convo in progress_bar(data, total=len(data)):
    if args.task == 'clc':
      example = abcd_classification(convo, mapping)
      examples.append(example)  
    elif args.task == 'ir':
      examples = abcd_retrieval(convo, examples)
  return examples

def build_gsim(data, mapping):
  examples = []
  prompt = "The topic of conversation is about"

  for conversation in progress_bar(data, total=len(data)):
    text_so_far = []    
    for turn in conversation['turns']:
      if 'system_utterance' in turn:
        sys_text = turn['system_utterance']['text']
        sys_utt = f"<agent> {sys_text}"
        text_so_far.append(sys_utt)

      user_text = turn['user_utterance']['text']
      user_utt = f"<customer> {user_text}"
      text_so_far.append(user_utt)

      context = ' '.join(text_so_far)
      act_id = extract_act(turn['user_acts'], mapping)
      examples.append({'context': context, 'prompt': prompt, 'label': act_id})  

  return examples

def extract_label(targets):
  # returns a list of (domain, slot, value) tuples when the domain is an active 
  swaps = {'not mentioned': 'none', 'dontcare': 'any', '': 'none'}
  labels = []

  for domain, domain_data in targets.items():
    domain_data = targets[domain]
    active_domain = False

    for slot, value in domain_data['book'].items():
      if len(value) > 0:
        active_domain = True
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

def build_mwoz22(args, data, label_set):
  if args.task == 'meta_learn':
    return meta_learn_mwoz(args, data, label_set)
  elif args.task == 'fine_tune':
    return fine_tune_mwoz(args, data, label_set)
  elif args.task == 'in_context':
    return in_context_mwoz(args, data, label_set)
  elif args.do_interact:
    mapping = {label: idx for idx, label in enumerate(label_set)}
    return interact_mwoz(args, mapping)

def fine_tune_mwoz(args, data, label_set):
  ''' Written for raw v2.2 mwoz.  Since evaluation is done by a library, 
  based on the dialog_id, we do not actually pass the ground truth target as
  the label for evaluation.  Instead, we use dialog_id as the meta_label '''
  examples = []
  speakers = ["<customer>", "<agent>"]

  for convo_id, conversation in progress_bar(data.items(), total=len(data)):
    text_so_far = []
    speaker_id = 0

    for turn in conversation['log']:
      text = turn['text']
      speaker = speakers[speaker_id]
      utterance = f"{speaker} {text}"
      
      if speaker_id == 0:
        text_so_far.append(utterance)
      elif speaker_id == 1:
        domain, d_tracker = extract_domain(turn['metadata'], label_set, d_tracker)
        prompt = make_prompt(args.prompt_style, domain, slot)
        if len(domain) > 0:
          context = ' '.join(text_so_far)
          dialogue = context + '<label>'

          dialog_id = convo_id.split('.')[0].lower()  # drop the ".json"
          turn_count = str(len(text_so_far))
          domain_string = ';'.join(active_domains)
          meta_label = "_".join([dialog_id, turn_count, domain_string]) 

          example = {'dialogue': dialogue, 'prompt': prompt, 'label': meta_label}
          examples.append(example)
        text_so_far.append(utterance)  # add agent utterance afterwards
      
      speaker_id = 1 - speaker_id
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

def build_sgd(data, mapping, split):
  examples = []
  prompt = "The topic of conversation is about"

  for conversation in progress_bar(data, total=len(data)):
    text_so_far = []    

    for turn in conversation['turns']:    
      utt = turn['utterance']

      if turn['speaker'] == 'SYSTEM':
        sys_text = f"<agent> {utt}"
        text_so_far.append(sys_text)
  
      elif turn['speaker'] == 'USER':
        user_text = f"<customer> {utt}"
        text_so_far.append(user_text)
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
            examples.append({'dialogue': dialogue, 'flattened': fls,  'structured': sls})
      if len(text_so_far) > 14:
        text_so_far = text_so_far[-14:]

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


def build_tt(data, mapping):
  examples = []
  for conversation in progress_bar(data, total=len(data)):
    
    text_so_far = []    
    for turn in conversation:
      current_utt = turn['utterance']
      context = ' '.join(text_so_far).deepcopy()

      labels = [mapping(label) for label in turn['labels']]

      example = {'context': context, 'utterance': current_utt, 'label': labels}
      examples.append(example)
  
      text_so_far.append(current_utt)
  return examples

def get_dataloader(args, dataset, split='train'):
  sampler = RandomSampler(dataset) if dataset.shuffle else SequentialSampler(dataset)
  collate = dataset.collate_func
  dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)
  print(f"Loaded {split} data with {len(dataloader)} batches")
  return dataloader

def prepare_examples(args, data, label_set, split):
  if args.dataset == 'abcd':    # Action Based Conversations
    examples = build_abcd(args, data, mapping) 
  elif args.dataset == 'dstc':  # State Tracking Challenge 2
    examples = build_dstc(args, data, mapping) 
  elif args.dataset == 'mwoz20':  # MultiWoz 2.0
    examples = build_mwoz20(args, data, label_set)
  elif args.dataset == 'mwoz21':  # MultiWoz 2.1
    examples = build_mwoz21(args, data, label_set)
  elif args.dataset == 'mwoz22':  # MultiWoz 2.2
    examples = build_mwoz22(args, data, label_set)
  elif args.dataset == 'sgd':   # Schema Guided Dialogue
    examples = build_sgd(args, data, mapping, split) 
  elif args.dataset == 'tt':    # TicketTalk / TaskMaster 3
    examples = build_tt(args, data, mapping) 

  return examples

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
        datasets[split] = MetaLearnDataset(examples, tokenizer,  args.task, split)
      elif args.task == 'in_context':
        datasets[split] = InContextDataset(examples, tokenizer,  args.task, split)
      elif args.task == 'fine_tune':
        datasets[split] = FineTuneDataset(examples, tokenizer,  args.task, split)
      print(f"Running with {len(datasets[split])} {split} examples")
    pkl.dump(datasets, open(cache_results, 'wb'))

  return datasets, label_set
