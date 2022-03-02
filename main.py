import os, sys, pdb
import numpy as np
import random
import torch

from tqdm import tqdm as progress_bar
from components.logger import ExperienceLogger

from utils.help import *
from utils.process import process_data, get_dataloader
from utils.arguments import solicit_params
from utils.load import *
from utils.evaluate import eval_quantify, eval_qualify
from assets.static_vars import device, debug_break, STOP_TOKENS

def run_train(args, model, datasets, exp_logger):
  train_dataloader = get_dataloader(args, datasets['train'])
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs
  optimizer, scheduler = setup_optimization(args, model, total_steps)
  exp_logger.update_optimization(optimizer, scheduler)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    model.train()
      
    for step, batch in enumerate(train_dataloader):
      inputs, targets = batch
      
      if args.model == 'trade':
        inputs =  prepare_inputs(batch)
        outputs = model(inputs, use_teacher_forcing, slot_temp)
        loss = trade_loss(pred_outputs, targets)
      else:
        review_inputs(args, targets, datasets['train'].tokenizer)
        outputs = model(**inputs, labels=targets)
      exp_logger.tr_loss += outputs.loss.item()
      loss = outputs.loss / args.grad_accum_steps
      loss.backward()

      if (step + 1) % args.grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        exp_logger.optimizer.step()  # backprop to update the weights
        exp_logger.scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        exp_logger.log_train(step)
      if args.debug and step >= debug_break*args.log_interval:
        break

    eval_res = run_eval(args, model, datasets, exp_logger)
    if args.do_save and eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
      exp_logger.best_score = eval_res
      exp_logger.save_best_model(model, tokenizer, args.prune_keep)
    early_stop = exp_logger.end_epoch()
    if early_stop: break

  return model

def run_inference(args, model, dataloader, exp_logger, split, extract_text=False):
  # runs a single epoch without gradients, optionally collects meta-data along the way
  if args.task == 'in_context':
    return in_context_eval(args, model, dataloader, exp_logger, split)
  elif args.task == 'meta_learn':
    return meta_learn_eval(args, model, dataloader, exp_logger, split, extract_text)
  elif args.task == 'fine_tune':
    return fine_tune_eval(args, model, dataloader, exp_logger, split)

def fine_tune_eval(args, model, dataloader, exp_logger, split):
  ''' goes through model generation without backprop, rather than classification '''
  all_outputs, all_labels = [], []
  exp_logger.eval_step = 0

  for inputs, label_dicts in progress_bar(dataloader, total=len(dataloader)):
    all_labels.extend(label_dicts)   # notice this is "extend", not "append"

    with torch.no_grad():
      # defaults to greedy sampling, for param details see https://huggingface.co/docs/transformers/
      #        v4.15.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate 
      outputs = model.generate(**inputs, max_length=args.max_len, early_stopping=True, do_sample=False)
      output_strings = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=True)
      all_outputs.extend(output_strings)

    if split == 'dev':
      exp_logger.eval_loss = 0  # no loss, since inference only
      exp_logger.eval_step += 1
      if args.debug and exp_logger.eval_step >= debug_break: break
  return all_outputs, all_labels

def in_context_eval(args, model, datasets):
  dataset = datasets['dev']
  train_data = datasets['train']
  eos = dataset.tokenizer.eos_token
  random.shuffle(dataset.data)
  num_examples = dataset.size
  options = [{'domain': 'hotel',      'id': 8940}, 
             {'domain': 'attraction', 'id': 1078},
             {'domain': 'restaurant', 'id': 2118},
             {'domain': 'taxi',       'id': 19290},
             {'domain': 'train',      'id': 27432}]
  correct = 0
  for example in progress_bar(dataset, total=num_examples):
    input_string = example['dialogue'] + example['prompt']
    inputs = dataset.tokenizer(input_string, return_tensors='pt').to(device)
  

    included_domains = set()
    while len(inputs['input_ids'][0]) < args.max_len:
      one_shot = train_data[random.randrange(train_data.size)]
      label = one_shot['label']

      if len(included_domains) < 5 and label in included_domains:
        continue  # resample a new example

      shot_string = one_shot['dialogue'] + one_shot['prompt'] + label + eos
      input_string = shot_string + input_string
      inputs = dataset.tokenizer(input_string, return_tensors='pt').to(device)

    trimmed = {
      'input_ids': inputs['input_ids'][:, -args.max_len:],
      'attention_mask': inputs['attention_mask'][:, -args.max_len:]
    }

    with torch.no_grad():
      size = args.max_len + 4
      outputs = model.generate(**trimmed, max_length=size, early_stopping=True,
              return_dict_in_generate=True, output_scores=True)
    
    target = example['label']
    """ 
    ops = torch.concat(outputs['scores']).detach()  # seq_len, vocab_size
    preds = ops.softmax(dim=-1)
    scores = [preds[0, opt['id']].item() for opt in options]
    pred_id = scores.index(max(scores))
    answer = options[pred_id]['domain']
    """
    output_text = tokenizer.decode(outputs['sequences'][0, -4:].detach(), skip_special_tokens=False)
    answer = output_text.strip() 
    
    if args.verbose:
      print(input_string[-100:])
      print(f"---- Target: {target}, Prediction {answer} -----")
    if target in answer:
      correct += 1

  accuracy = round((float(correct) / num_examples) * 100, 1)
  print("accuracy: {}%".format(accuracy))
  return accuracy

def run_eval(args, model, datasets, exp_logger, split='dev'):
  dataloader = get_dataloader(args, datasets[split], split)
  tokenizer = datasets[split].tokenizer

  if split == 'test':        
    outputs = run_inference(args, model, dataloader, exp_logger, split)
    results = eval_quantify(args, *outputs, exp_logger, tokenizer, split)
  else:
    model.eval()
    outputs = run_inference(args, model, dataloader, exp_logger, split)
    results = eval_quantify(args, *outputs, exp_logger, tokenizer, split)

  return results

def meta_learn_eval(args, model, dataset):
  dataset = datasets['dev']
  for i in range(args.batch_size):
    sample_id = random.randrange(dataset.size)
    sample = dataset[sample_id]
    dialog = sample['context']
    if len(dialog) > 1000: continue

    print(f"---- Chat {i+1} -----")
    print(dialog)
    prompt = input("Customer: ")
    if prompt in STOP_TOKENS: sys.exit()

    input_text = dialog + " Customer: " + prompt + " Agent:"
    inputs = dataset.tokenizer(input_text, return_tensors='pt').to(device)

    # https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate 
    with torch.no_grad():
      if args.kappa > 1:
        output_embeds = model.generate(**inputs, max_length=512, early_stopping=True, do_sample=True, 
                                        num_beams=args.kappa, temperature=args.temperature, top_p=0.95)
        output_texts = tokenizer.batch_decode(output_embeds.detach(), skip_special_tokens=False)
        for i, out_text in enumerate(output_texts):
          print(f"<agent {i+1}> {out_text}")
      else: 
        output_embed = model.generate(**inputs, max_length=200, early_stopping=True, length_penalty=1.0,
                                       repetition_penalty=args.threshold, temperature=args.temperature)
        output_text = tokenizer.decode(output_embed[0].detach(), skip_special_tokens=True)
        parts = output_text.split("Agent: ")
        response = parts[-1]
        if "\n" in response:
          newline_index = response.index("\n")
          response = response[:newline_index]
        print(f"Agent: {response}")


if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  args, save_path = check_directories(args)
  set_seed(args)

  raw_data = load_data(args)
  tokenizer = load_tokenizer(args)
  datasets, ontology = process_data(args, raw_data, tokenizer)
  exp_logger = ExperienceLogger(args, ontology, save_path)

  model = load_model(args, ontology, tokenizer, save_path)
  if args.do_train:
    run_train(args, model, datasets, exp_logger)
  elif args.do_interact:
    run_interaction(args, model, datasets)
  elif args.do_eval:
    run_eval(args, model, datasets, exp_logger, split='test')
