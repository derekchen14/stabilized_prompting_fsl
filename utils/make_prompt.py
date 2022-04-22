import os, pdb, sys
import numpy as np
from assets.static_vars import DOMAIN_SLOTS

def find_prompt(style, target):
  domain = target['domain'].lower()
  slot = target['slot'].lower()
  if domain.endswith('s'):
    domain = domain[:-1]

  if style == 'schema':
    return schema_descriptions[domain][slot]
  elif style == 'question':
    return question_descriptions[domain][slot]
  elif style == 'informed':
    return slot_informed(domain, slot)
  elif style == 'naive':
    return naive_style(domain, slot)
  elif style == 'human':
    return human_descriptions[domain][slot]
  elif style == 'none':
    return f"<sep> {domain} {slot} <label>"
  elif style == 'random':
    return random_style(domain, slot)

def random_style(domain, slot):
  colors = ['red', 'blue', 'green', 'indigo', 'violet', 'yellow', 'orange', 'pink', 'purple',
      'brown', 'black', 'white', 'gray', 'rose', 'cerulean', 'navy', 'magenta', 'cyan']
  animals = ['alligator','buffalo','cheetah','dog','elephant','fish','girrafe','hippo',
      'iguana','jaguar','kangaroo','lion','mammoth','newt','octopus','parrot','squirrel',
      'racoon','shark','tiger','unicorn','vulture','whale','lynx','yak','zebra']
  # slots are replaced with a random color
  # domains are replaced with a random animal
  pass

def slot_informed(domain, slot):
  desc = slot_informed_descriptions[domain][slot] + " is "
  return desc

def extract_domain(metadata, label_set, domain_tracker):
  for domain in label_set:
    domain_data = metadata[domain]

    slotvals = []
    for slot, value in domain_data['book'].items():
      if len(value) > 0 and not isinstance(value, list):
        slotvals.append(value)
    for slot, value in domain_data['semi'].items():
      if len(value) > 0:
        slotvals.append(value)

    previous = domain_tracker[domain]
    current = '_'.join(slotvals)
    if current != previous:
      domain_tracker[domain] = current
      return domain, domain_tracker  # index

  # could not find anything 
  return "", domain_tracker

def extract_domain_slot(targets):
  for domain, slots in DOMAIN_SLOTS.items():
    for slot in slots:
      pass

  return domain, slot


def naive_style(domain, slot):
  try:
    desc = naive_descriptions[domain][slot] + " is "
  except(KeyError):
    desc = f"{slot} of the {domain} is "
  return '<sep> ' + desc

def topic_prompts(style):
  if style == 'schema':
    prompt = "topic of conversation"
  elif style == 'question':
    prompt = "What is the customer looking for?"
  elif style == 'statement':
    prompt = " The topic of conversation is about a" 
  elif style == 'token':
    prompt = "Topic:"
  return prompt

def train_prompts(style):
  if style == 'schema':
    prompt = "destination location of the train"
  elif style == 'question':
    prompt = "Where does the user want to go?"
  elif style == 'statement':
    prompt = "The user wants to ride the train to"
  elif style == 'token':
    prompt = "Destination:"
  return prompt


schema_descriptions = {
  "taxi": {
    "leaveat": "what time you want the taxi to leave your departure location by",
    "destination": "what place do you want the taxi to go",
    "departure": "what place do you want to meet the taxi",
    "arriveby": "when you want the taxi to drop you off at your destination" }, 
  "restaurant": {
    "bookpeople": "number of people booking the restaurant",
    "bookday": "what day of the week to book the table at the restaurant",
    "booktime": "time of the restaurant booking",
    "food": "food type for the restaurant",
    "pricerange": "price budget for the restaurant",
    "name": "name of the restaurant",
    "area": "preferred location of restaurant"},
  "train": {
    "destination": "destination of the train",
    "day": "what day you want to take the train",
    "departure": "departure location of the train",
    "arriveby": "what time you want the train to arrive at your destination station by",
    "bookpeople": "number of people booking for train",
    "leaveat": "when you want to arrive at your destination by train"},
  "hotel": {
    "pricerange": "preferred cost of the hotel",
    "type": "type of hotel building",
    "parking": "parking facility at the hotel",
    "bookstay": "length of stay at the hotel",
    "bookday": "day of the hotel booking",
    "bookpeople": "how many people are staying at the hotel",
    "area": "rough location of the hotel",
    "stars": "rating of the hotel out of five stars",
    "internet": "whether the hotel has internet",
    "name": "which hotel are you looking for"},
  "attraction": {
    "type": "type of attraction or point of interest",
    "area": "area or place of the attraction",
    "name": "name of the attraction"}
}

slot_informed_descriptions = {
  "restaurant": {
    "area": "area of restaurant",
    "bookpeople": "number of people for the restaurant booking",
    "bookday": "day for the restaurant booking",
    "booktime": "time of booking of the restaurant",
    "food": "food of restaurant",
    "name": "name of restaurant",
    "pricerange": "price range of restaurant"},
  "taxi": {
    "arriveby": "time of arrive by of the taxi",
    "destination": "location of destination of the taxi",
    "departure": "location of departure of the taxi",
    "leaveat": "time of leave at of the taxi"}, 
  "train": {
    "arriveby": "time of arrive by of the train",
    "bookpeople": "number of people for the train booking",
    "day": "day for the train booking",
    "destination": "location of destination of the train",
    "departure": "location of departure of the train",
    "leaveat": "time of leave at of the train"},
  "hotel": {
    "area": "area of the hotel",
    "bookday": "day for the hotel booking",
    "bookstay": "number of nights for the hotel booking",
    "bookpeople": "number of people for the hotel booking",
    "internet": "whether have internet in the hotel",
    "name": "name of hotel",
    "parking": "whether have parking in the hotel",
    "pricerange": "price range of the hotel",
    "stars": "number of stars for the hotel booking",
    "type": "type of the hotel"},
  "attraction": {
    "area": "area of attraction",
    "name": "name of attraction",
    "type": "type of attraction"}
}

naive_descriptions = {
  "restaurant": {
    "area": "area of the restaurant",
    "bookpeople": "book people of the restaurant",
    "bookday": "book day of the restaurant",
    "booktime": "book time of the restaurant",
    "food": "food of the restaurant",
    "name": "name of the restaurant",
    "pricerange": "price range of the restaurant"},
  "taxi": {
    "arriveby": "arrive by of the taxi",
    "destination": "destination of the taxi",
    "departure": "departure of the taxi",
    "leaveat": "leave at of the taxi"}, 
  "train": {
    "arriveby": "arrive by of the train",
    "bookpeople": "book people of the train",
    "day": "day of the train",
    "destination": "destination of the train",
    "departure": "departure of the train",
    "leaveat": "leave at of the train"},
  "hotel": {
    "area": "area of the hotel",
    "bookday": "book day of the hotel",
    "bookstay": "book stay of the hotel",
    "bookpeople": "book people of the hotel",
    "internet": "internet of the hotel",
    "name": "name of the hotel",
    "parking": "parking of the hotel",
    "pricerange": "price range of the hotel",
    "stars": "stars of the hotel",
    "type": "type of the hotel"},
  "attraction": {
    "area": "area of the attraction",
    "name": "name of the attraction",
    "type": "type of the attraction"}
}

is_categorical = {
  "restaurant": {
    "area": True,
    "bookpeople": True,
    "bookday": True,
    "booktime": False,
    "food": True,
    "name": False,
    "pricerange": True},
  "taxi": {
    "arriveby": False,
    "destination": False,
    "departure": False,
    "leaveat": False}, 
  "train": {
    "arriveby": False,
    "bookpeople": False,
    "day": True,
    "destination": False,
    "departure": False,
    "leaveat": False},
  "hotel": {
    "area": True,
    "bookday": True,
    "bookstay": True,
    "bookpeople": True,
    "internet": True,
    "name": False,
    "parking": True,
    "pricerange": True,
    "stars": True,
    "type": True},
  "attraction": {
    "area": True,
    "name": False,
    "type": False}
}

question_descriptions = {
  "restaurant": {
    "area": "What is the area of the restaurant that the user is interested in?",
    "bookpeople": "What is the book people of the restaurant that the user is interested in?",
    "bookday": "What is the book day of the restaurant that the user is interested in?",
    "booktime": "What is the book time of the restaurant that the user is interested in?",
    "food": "What is the food of the restaurant that the user is interested in?",
    "name": "What is the name of the restaurant that the user is interested in?",
    "pricerange": "What is the price range of the restaurant that the user is interested in?"},
  "taxi": {
    "arriveby": "What is the arrive by of the taxi that the user is interested in?",
    "destination": "What is the destination of the taxi that the user is interested in?",
    "departure": "What is the departure of the taxi that the user is interested in?",
    "leaveat": "What is the leave at of the taxi that the user is interested in?"}, 
  "train": {
    "arriveby": "What is the arrive by of the train that the user is interested in?",
    "bookpeople": "What is the book people of the train that the user is interested in?",
    "day": "What is the day of the train that the user is interested in?",
    "destination": "What is the destination of the train that the user is interested in?",
    "departure": "What is the departure of the train that the user is interested in?",
    "leaveat": "What is the leave at of the train that the user is interested in?"},
  "hotel": {
    "area": "What is the area of the hotel that the user is interested in?",
    "bookday": "What is the book day of the hotel that the user is interested in?",
    "bookstay": "What is the book stay of the hotel that the user is interested in?",
    "bookpeople": "What is the book people of the hotel that the user is interested in?",
    "internet": "What is the internet of the hotel that the user is interested in?",
    "name": "What is the name of the hotel that the user is interested in?",
    "parking": "What is the parking of the hotel that the user is interested in?",
    "pricerange": "What is the price range of the hotel that the user is interested in?",
    "stars": "What is the stars of the hotel that the user is interested in?",
    "type": "What is the type of the hotel that the user is interested in?"},
  "attraction": {
    "area": "What is the area of the attraction that the user is interested in?",
    "name": "What is the name of the attraction that the user is interested in?",
    "type": "What is the type of the attraction that the user is interested in?"}
}
