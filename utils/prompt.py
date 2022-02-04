import os, pdb, sys
import numpy as np

from assets.static_vars import device

def make_prompt(style):
  # " Topic:" The customer is looking for a"
  if style == 'schema':
    prompt = "destination location of the train"
  elif style == 'question':
    prompt = "Where does the user want to go?"
  elif style == 'statement':
    prompt = "The user wants to ride the train to"
  elif style == 'token':
    prompt = "Destination:"
  elif style == 'none':
    prompt = "value is"


  prompt = " The topic of conversation is about a" 
  return prompt
