import os, sys, pdb
import numpy as np
import random
import json

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils.arguments import solicit_params
from assets.static_vars import device, debug_break, STOP_TOKENS

def interact(args, model, dataset):
  dataset = datasets['dev']
  for i in range(args.batch_size):
    sample_id = random.randrange(dataset.size)
    sample = dataset[sample_id]
    dialog = ' '.join(sample['utterances'])
    if len(dialog) > 1000: continue

    print(f"---- Chat {i+1} -----")
    print(dialog)

    domain = sample['target']['domain']
    slot = sample['target']['slot']
    prompt = f'<sep> {domain} {slot} <label>'
    # prompt = input("Customer: ")
    if prompt in STOP_TOKENS: sys.exit()

    input_text = dialog + prompt
    inputs = dataset.tokenizer(input_text, return_tensors='pt').to(device)

    # https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate 
    with torch.no_grad():
      # if args.kappa > 1:
      #   output_embeds = model.generate(**inputs, max_length=512, early_stopping=True, do_sample=True, 
      #                                   num_beams=args.kappa, temperature=args.temperature, top_p=0.95)
      #   output_texts = tokenizer.batch_decode(output_embeds.detach(), skip_special_tokens=False)
      #   for i, out_text in enumerate(output_texts):
      #     print(f"<agent {i+1}> {out_text}")
      # else: 
      output_embed = model.generate(**inputs, max_length=512, early_stopping=True, length_penalty=1.0,
                                     repetition_penalty=args.threshold, temperature=args.temperature)

    output_text = tokenizer.decode(output_embed[0].detach(), skip_special_tokens=True)
    response = output_text.split('<label>')[1]
    print(response)
    pdb.set_trace()


def learn_in_context(args, model, datasets):
  """ does data collate, prompt concatenation and inference all-in-one"""
  dataset = datasets['dev']
  train_data = datasets['train']
  eos = dataset.tokenizer.eos_token
  # do more data prep to select examples for in_context learning
  dataset = add_context(args, dataset)

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

def load_data(args):
  cache_file = f'{args.model}_{args.task}_leftoutnone_lookback{args.context_length}.pkl'
  cache_path = os.path.join(args.input_dir, 'cache', args.dataset, cache_file)
  data = pkl.load( open( cache_path, 'rb' ) )
  return data

def load_checkpoint(args):
  ckpt_name = os.path.join(args.output_dir, args.dataset, args.task, args.checkpoint)
  print(f"Trying to load {ckpt_name}")
  model = GPT2LMHeadModel.from_pretrained(ckpt_name)
  model.eval()
  model.to(device)
  return model

if __name__ == '__main__':
  args = solicit_params()
  data = load_data(args)
  model = load_best_model(args)
  interact(args, model, data)


# python interact.py --model gpt --task fine_tune --context-length 3 --dataset mwoz --size small