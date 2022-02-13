import os, pdb, sys
import numpy as np
from assets.static_vars import DOMAIN_SLOTS

schema_descriptions = {
  "taxi": {
    "leaveAt": "what time you want the taxi to leave your departure location by",
    "destination": "destination of taxi",
    "departure": "what place do you want to meet the taxi",
    "arriveBy": "when you want the taxi to drop you off at your destination" }, 
  "restaurant": {
    "book people": "number of people booking the restaurant",
    "book day": "what day of the week to book the table at the restaurant",
    "book time": "time of the restaurant booking",
    "food": "food type for the restaurant",
    "pricerange": "price budget for the restaurant",
    "name": "name of the restaurant",
    "area": "preferred location of restaurant"},
  "train": {
    "destination": "destination of the train",
    "day": "what day you want to take the train",
    "departure": "departure location of the train",
    "arriveBy": "what time you want the train to arrive at your destination station by",
    "book people": "number of people booking for train",
    "leaveAt": "when you want to arrive at your destination by train"},
  "hotel": {
    "pricerange": "preferred cost of the hotel",
    "type": "type of hotel building",
    "parking": "parking facility at the hotel",
    "book stay": "length of stay at the hotel",
    "book day": "day of the hotel booking",
    "book people": "how many people are staying at the hotel",
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
    "book people": "number of people for the restaurant booking",
    "book day": "day for the restaurant booking",
    "book time": "time of booking of the restaurant",
    "food": "food of restaurant",
    "name": "name of restaurant",
    "pricerange": "price range of restaurant"},
  "taxi": {
    "arriveBy": "time of arrive by of the taxi",
    "destination": "location of destination of the taxi",
    "departure": "location of departure of the taxi",
    "leaveAt": "time of leave at of the taxi"}, 
  "train": {
    "arriveBy": "time of arrive by of the train",
    "book people": "number of people for the train booking",
    "day": "day for the train booking",
    "destination": "location of destination of the train",
    "departure": "location of departure of the train",
    "leaveAt": "time of leave at of the train"},
  "hotel": {
    "area": "area of the hotel",
    "book day": "day for the hotel booking",
    "book stay": "number of nights for the hotel booking",
    "book people": "number of people for the hotel booking",
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
    "book people": "book people of the restaurant",
    "book day": "book day of the restaurant",
    "book time": "book time of the restaurant",
    "food": "food of the restaurant",
    "name": "name of the restaurant",
    "pricerange": "price range of the restaurant"},
  "taxi": {
    "arriveBy": "arrive by of the taxi",
    "destination": "destination of the taxi",
    "departure": "departure of the taxi",
    "leaveAt": "leave at of the taxi"}, 
  "train": {
    "arriveBy": "arrive by of the train",
    "book people": "book people of the train",
    "day": "day of the train",
    "destination": "destination of the train",
    "departure": "departure of the train",
    "leaveAt": "leave at of the train"},
  "hotel": {
    "area": "area of the hotel",
    "book day": "book day of the hotel",
    "book stay": "book stay of the hotel",
    "book people": "book people of the hotel",
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
    "book people": True,
    "book day": True,
    "book time": False,
    "food": True,
    "name": False,
    "pricerange": True},
  "taxi": {
    "arriveBy": False,
    "destination": False,
    "departure": False,
    "leaveAt": False}, 
  "train": {
    "arriveBy": False,
    "book people": False,
    "day": True,
    "destination": False,
    "departure": False,
    "leaveAt": False},
  "hotel": {
    "area": True,
    "book day": True,
    "book stay": True,
    "book people": True,
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
    "book people": "What is the book people of the restaurant that the user is interested in?",
    "book day": "What is the book day of the restaurant that the user is interested in?",
    "book time": "What is the book time of the restaurant that the user is interested in?",
    "food": "What is the food of the restaurant that the user is interested in?",
    "name": "What is the name of the restaurant that the user is interested in?",
    "pricerange": "What is the price range of the restaurant that the user is interested in?"},
  "taxi": {
    "arriveBy": "What is the arrive by of the taxi that the user is interested in?",
    "destination": "What is the destination of the taxi that the user is interested in?",
    "departure": "What is the departure of the taxi that the user is interested in?",
    "leaveAt": "What is the leave at of the taxi that the user is interested in?"}, 
  "train": {
    "arriveBy": "What is the arrive by of the train that the user is interested in?",
    "book people": "What is the book people of the train that the user is interested in?",
    "day": "What is the day of the train that the user is interested in?",
    "destination": "What is the destination of the train that the user is interested in?",
    "departure": "What is the departure of the train that the user is interested in?",
    "leaveAt": "What is the leave at of the train that the user is interested in?"},
  "hotel": {
    "area": "What is the area of the hotel that the user is interested in?",
    "book day": "What is the book day of the hotel that the user is interested in?",
    "book stay": "What is the book stay of the hotel that the user is interested in?",
    "book people": "What is the book people of the hotel that the user is interested in?",
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

def make_prompt(style, domain, slot):
  if style == 'schema':
    return schema_descriptions[domain][slot]
  elif style == 'question':
    return question_descriptions[domain][slot]
  elif style == 'informed':
    return slot_informed_descriptions[domain][slot]
  elif style == 'naive':
    return naive_descriptions[domain][slot]
  elif style == 'human':
    return human_descriptions[domain][slot]
  elif style == 'none':
    return " "
  elif style == 'random':
    return "random"

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

