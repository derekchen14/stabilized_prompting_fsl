import os, pdb, sys
import numpy as np
from assets.static_vars import DOMAIN_SLOTS

def find_prompt(style, domain, slot):
  domain = domain.lower()
  slot = slot.lower()
  if domain.endswith('s'):
    domain = domain[:-1]

  if style == 'schema':   #  taken from https://arxiv.org/abs/2109.07506
    return schema_style(domain, slot)
  elif style == 'question':
    return question_style(domain, slot)
  elif style == 'statement':
    return statement_style(domain, slot)
  elif style == 'naive':
    return naive_style(domain, slot)
  elif style == 'human':
    return human_descriptions[domain][slot]
  elif style == 'none':
    return f"<sep> {domain} {slot} <label>"
  elif style == 'random':
    return random_style(domain, slot)

def schema_style(domain, slot):
  domain_desc = schema_domains[domain]
  slot_desc = schema_slots[domain][slot]
  prompt = f"<sep> [domain] {domain_desc} [slot] {slot_desc} is"
  return prompt

def question_style(domain, slot):
  desc = question_descriptions[domain][slot]
  prompt = f"<sep> {desc}"
  return prompt

def statement_style(domain, slot):
  try:
    desc = statement_descriptions[domain][slot]
  except(KeyError):
    pdb.set_trace()
  prompt = f"<sep> {desc} is"
  return prompt

def naive_style(domain, slot):
  if slot == 'arriveby':
    slot = 'arrive by'
  elif slot == 'leaveat':
    slot = 'leave at'
  elif slot == 'pricerange':
    slot = 'price range'
  slot = slot.replace('_', ' ')

  desc = f"{slot} of the {domain}"
  prompt = f"<sep> {desc} is"
  return prompt

schema_domains = {
   "restaurant": "find places to dine and whet your appetite",
   "taxi": "rent cheap cabs to avoid traffic",
   "train": "find trains that take you to places",
   "hotel": "hotel reservations and vacation stays",
   "attraction": "find touristy stuff to do around you",
   "hospital": "making you feel better when you are ill",
   "bus": "bus service for traveling"
}

schema_slots = {
  "restaurant": {
    "area": "preferred location of restaurant",
    "day": "what day of the week to book the table at the restaurant",
    "food": "food type for the restaurant",
    "people": "number of people booking the restaurant",
    "pricerange": "price budget for the restaurant",
    "time": "time of the restaurant booking",
    "name": "name of the restaurant"},
  "taxi": {
    "leaveat": "what time you want the taxi to leave your departure location by",
    "destination": "what place do you want the taxi to go",
    "departure": "what place do you want to meet the taxi",
    "arriveby": "when you want the taxi to drop you off at your destination" }, 
  "train": {
    "destination": "destination of the train",
    "day": "what day you want to take the train",
    "departure": "departure location of the train",
    "arriveby": "what time you want the train to arrive at your destination station by",
    "people": "number of people booking for train",
    "leaveat": "when you want to arrive at your destination by train"},
  "hotel": {
    "pricerange": "preferred cost of the hotel",
    "type": "type of hotel building",
    "parking": "parking facility at the hotel",
    "stay": "length of stay at the hotel",
    "day": "day of the hotel booking",
    "people": "how many people are staying at the hotel",
    "area": "rough location of the hotel",
    "stars": "rating of the hotel out of five stars",
    "internet": "whether the hotel has internet",
    "name": "which hotel are you looking for"},
  "attraction": {
    "type": "type of attraction or point of interest",
    "area": "area or place of the attraction",
    "name": "name of the attraction"}
}

statement_descriptions = {
  "restaurant": {
    "area": "The area of the restaurant",
    "day": "The day for the restaurant booking",
    "food": "The restaurant cuisine",
    "people": "The number of people for the restaurant",
    "pricerange": "The price range of the restaurant",
    "time": "The time of the restaurant booking",
    "name": "The name of the restaurant"},
  "taxi": {
    "arriveby": "The arrival time of the taxi",
    "destination": "The destination location of the taxi",
    "departure": "The departure location of the taxi",
    "leaveat": "The leave at time of the taxi"}, 
  "train": {
    "arriveby": "The arrival time of the train",
    "people": "The number of people for the train",
    "day": "The day for the train booking",
    "destination": "The destination of the train",
    "departure": "The departure location of the train",
    "leaveat": "The leave at time of the train"},
  "hotel": {
    "area": "The area of the hotel",
    "day": "The day for the hotel booking",
    "stay": "The number of nights for the hotel booking",
    "people": "The number of people for the hotel booking",
    "internet": "The internet at the hotel",
    "name": "The name of hotel",
    "parking": "The parking at the hotel",
    "pricerange": "The price range of the hotel",
    "stars": "The number of stars for the hotel",
    "type": "The type of hotel"},
  "attraction": {
    "area": "The area of attraction",
    "name": "The name of attraction",
    "type": "The type of attraction"}
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
    "area": "Which area of the restaurant is the user interested in?",
    "people": "How many people is the restaurant booking for?",
    "day": "What day is the restaurant booking for?",
    "time": "When is time of the restaurant booking?",
    "food": "What type of food is the user interested in?",
    "name": "What is the name of the restaurant?",
    "pricerange": "What is the price range of the restaurant?"},
  "taxi": {
    "arriveby": "When is the arrive by time of the taxi?",
    "destination": "Where is the destination of the taxi?",
    "departure": "Where is the departure of the taxi?",
    "leaveat": "When is the leave at time of the taxi?"}, 
  "train": {
    "arriveby": "When is the arrive by time of the train?",
    "people": "How many people is the train booking for?",
    "day": "What day is the train booking for?",
    "destination": "Where is the destination of the train?",
    "departure": "Where is the departure of the train?",
    "leaveat": "When is the leave at time of the train?"},
  "hotel": {
    "area": "Which area of the hotel is the user interested in?",
    "day": "What is the start day of the hotel?",
    "stay": "How long is the stay at the hotel?",
    "people": "How many people is the hotel for?",
    "internet": "Does the hotel offer internet?",
    "name": "What is the name of the hotel?",
    "parking": "Does the hotel offer parking?",
    "pricerange": "What is the price range of the hotel?",
    "stars": "How many stars does the hotel have?",
    "type": "What type of hotel is the user interested in?"},
  "attraction": {
    "area": "Which area of the attraction is the user interested in?",
    "name": "What is the name of the attraction?",
    "type": "What type of attraction is the user interested in?"}
}

human_descriptions = {
  "restaurant": "that the user is interested in"
}  

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

def random_style(domain, slot):
  colors = ['red', 'blue', 'green', 'indigo', 'violet', 'yellow', 'orange', 'pink', 'purple',
      'brown', 'black', 'white', 'gray', 'rose', 'cerulean', 'navy', 'magenta', 'cyan']
  animals = ['alligator','buffalo','cheetah','dog','elephant','fish','girrafe','hippo',
      'iguana','jaguar','kangaroo','lion','mammoth','newt','octopus','parrot','squirrel',
      'racoon','shark','tiger','unicorn','vulture','whale','lynx','yak','zebra']
  # slots are replaced with a random color
  # domains are replaced with a random animal
  pass

