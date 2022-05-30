import os, pdb, sys
import numpy as np

def find_prompt(style, domain, slot):
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
  prompt = f"<sep> {desc}?"
  return prompt

def statement_style(domain, slot):
  if domain in ['flight', 'music', 'movie', 'home', 'bus', 'medium', 'message', 'weather']:
    return naive_style(domain, slot)
  try:
    desc = statement_descriptions[domain][slot]
  except(KeyError):
    print(f"Expected domain {domain}, slot {slot}")
    desc = "blank"
    pdb.set_trace()
  prompt = f"<sep> {desc} is"
  return prompt

def naive_style(domain, slot):
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
    "price range": "price budget for the restaurant",
    "time": "time of the restaurant booking",
    "name": "name of the restaurant"},
  "taxi": {
    "leave at": "what time you want the taxi to leave your departure location by",
    "destination": "what place do you want the taxi to go",
    "departure": "what place do you want to meet the taxi",
    "arrive by": "when you want the taxi to drop you off at your destination" }, 
  "train": {
    "destination": "destination of the train",
    "day": "what day you want to take the train",
    "departure": "departure location of the train",
    "arrive by": "what time you want the train to arrive at your destination station by",
    "people": "number of people booking for train",
    "leave at": "when you want to arrive at your destination by train"},
  "hotel": {
    "price range": "preferred cost of the hotel",
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
    "location": "The location of the restaurant", 
    "area": "The area of the restaurant",
    "day": "The day for the restaurant booking",
    "date": "The date for the restaurant",
    "food": "The restaurant cuisine",
    "category": "The restaurant cuisine",
    "meal": "The meal of the restaurant",
    "num people": "The number of people for the restaurant",
    "people": "The number of people for the restaurant",
    "price range": "The price range of the restaurant",
    "time": "The time of the restaurant booking",
    "live music": "Whether the restaurant has live music",
    "has vegetarian options": "Whether the restaurant has vegetarian options",
    "address": "The address of the restaurant",
    "rating": "The rating for the restaurant",
    "outdoor seating": "Whether the restaurant has outdoor seating",
    "alcohol": "Whether the restaurant allows alcohol",
    "phone number": "The phone number of the restaurant",
    "rating": "The rating of the restaurant",
    "restaurant name": "The name of the restaurant",
    "name": "The name of the restaurant"},
  "movie": {
    "theater": "The name of the theater",
    "theatre name": "The name of the theater",
    "genre": "The genre of the movie",
    "screening": "The screening of the movie", 
    "movie": "The name of the movie",
    "show date": "The date of the movie",
    "date": "The date of the movie",
    "location": "The location of the movie",
    "num tickets": "The number of tickets for the movie",
    "preferred time": "The time of the movie",
    "show time": "The show time of the movie",
    "time": "The time of the movie"},
  "taxi": {
    "arrive by": "The arrival time of the taxi",
    "destination": "The destination location of the taxi",
    "departure": "The departure location of the taxi",
    "leave at": "The leave at time of the taxi"}, 
  "rideshare": {
    "duration": "The duration of the ride",
    "destination": "The destination location of the ride",
    "wait time": "The waiting time for the ride",
    "fare": "The rideshare fare",
    "shared ride": "Whether the rideshare is a shared ride",
    "type": "The type of car for the ride",
    "number of seats": "The number of seats for the rideshare"},
  "rental": {
    "price": "The price of the rental",
    "pickup time": "The pickup time for the rental",
    "dropoff date": "The dropoff date for the rental",
    "pickup date": "The pickup date for the rental",
    "insurance": "Whether the rental includes insurance",
    "name": "The name of the rental",
    "leave at": "The leave at time for the rental",
    "departure": "The departure time for the rental",
    "type": "The type of car for the rental",
    "pickup city": "The pickup city of the rental"},
  "train": {
    "arrive by": "The arrival time of the train",
    "people": "The number of people for the train",
    "day": "The date of the train booking",
    "destination": "The destination of the train",
    "departure": "The departure location of the train",
    "trip protection": "Whether the train offers trip protection",
    "arrival": "The arrival time of the train",
    "price": "The price of the train ticket",
    "destination": "The destination of the train ride",
    "class": "The class of the train ride",
    "date": "The date of the train booking",
    "origin": "The origin of the train",
    "departure": "The departure location of the train",
    "number of seats": "The number of seats on the train",
    "leave at": "The leave at time of the train"},
  "hotel": {
    "area": "The area of the hotel",
    "day": "The day for the hotel booking",
    "stay": "The number of days for the hotel booking",
    "people": "The number of people for the hotel booking",
    "internet": "Whether the hotel includes internet",
    "name": "The name of hotel",
    "parking": "Whether the hotel includes parking",
    "price range": "The price range of the hotel",
    "stars": "The number of stars for the hotel",
    "type": "The type of hotel",
    "rating": "The number of stars for the hotel",
    "price": "The price range of the hotel",
    "address": "The area of the hotel",
    "number of rooms": "The number of rooms for the hotel booking",
    "laundry": "Whether the hotel offers laundry service",
    "pets welcome": "Whether pets are welcome at the hotel",
    "smoking": "Whether smoking is allowed at the hotel",
    "number of days": "The number of days for the hotel booking",
    "check out": "The check out date for the hotel booking",
    "check in": "The check in date for the hotel booking",
    "location": "The location of the hotel",
    "phone number": "The phone number of the hotel",
    "street address": "The address of the hotel"},
  "attraction": {
    "area": "The area of attraction",
    "name": "The name of attraction",
    "type": "The type of attraction"},
  "event": {
    "time": "The time of the event",
    "address": "The address of the event",
    "city": "The city of the event",
    "price": "The price of the event",
    "venue": "The venue of the event",
    "category": "The type of event",
    "date": "The date of the event",
    "name": "The name of the event",
    "num people": "The number of people joining the event",
    "type": "The type of event"},
  "travel": {
    "location": "The location of travel",
    "free entry": "Whether the travel event has free entry",
    "type": "The type of travel",
    "good for kids": "Whether the travel event is good for kids",
    "name": "The name for travel",
    "phone number": "The phone number for travel"},
  "manage account": {
    "customer name": "The customer's name for the account",
    "street address": "The street address of the account",
    "zip code": "The zip code of the account",
    "product": "The product related to the account",
    "phone": "The phone number on the account",  
    "username": "The username of the account", 
    "shipping option": "The shipping method on the account",
    "account slotval": "The details of the account",
    "payment method": "The payment method of the account",
    "amount": "The amount related to the account",
    "order id": "The order id for the account",
    "membership level": "The membership level of the account",
    "order slotval": "The order related to the account",
    "reason slotval": "The reason for managing the account", 
    "company team": "The company team managing the account"},
  "storewide query": {
    "change option": "The detail being changed for the query", 
    "customer name": "The customer's name for the query",
    "account slotval": "The account of the query",
    "company team": "The company team dealing with query",
    "reason slotval": "The reason for the query",
    "order slotval": "The order related to the query",
    "shipping option": "The shipping method related to the query",
    "membership level": "The membership level of the query",
    "payment method": "The payment method for the query",
    "product": "The product of the query",
    "order slotval": "The order of the query",
    "zip code": "The zip code for the query"},
  "single item query": {
    "product": "The product of the query",
    "amount": "The amount related to the query",
    "shipping option": "The shipping method of the query",
    "company team": "The company team related to the query",
    "membership level": "The membership level related to the query",
    "customer name": "The customer's name for the query"},
  "order issue": {
    "amount": "The amount of the order",
    "account id": "The account id of the order",
    "customer name": "The customer name for the order",
    "shipping option": "The shipping method for the order",
    "order id": "The order id for the order",
    "change option": "The detail being changed for the order issue",
    "order slotval": "The value of the order",
    "account slotval": "The account for the order",
    "reason slotval": "The reason for the order issue",
    "membership level": "The membership level for the order", 
    "payment method": "The payment method for the order",
    "refund target": "The refund method for the order",
    "zip code": "The zip code for the order",
    "product": "The product name for the order", 
    "company team": "The company team dealing with the order"},
  "product defect": {
    "email": "The email for the product defect",
    "account id": "The account id for the product",
    "membership level": "The membership level for the product defect",
    "order id": "The order id for the product defect",
    "username": "The username for the product defect",
    "order slotval": "The order detail of the product", 
    "shipping option": "The shipping method for the product",
    "details slotval": "The details of the product defect", 
    "reason slotval": "The reason for the product defect",
    "product": "The name of the product",
    "amount": "The amount of the product defect", 
    "zip code": "The zip code for the product defect",
    "payment method": "The payment method for the product",
    "refund target": "The refund method for the product defect",
    "company team": "The company team dealing with the product defect",
    "customer name": "The customer's name for the product defect"},
  "purchase dispute": {
    "customer name": "The customer's name for the purchase",
    "product": "The product for the purchase",
    "account slotval": "The account of the purchase dispute",
    "amount": "The amount of the purchase dispute",
    "shipping option": "The shipping method of the purchase dispute",
    "change option": "The detail being changed for the purchase",
    "account id": "The account id for the purchase",
    "phone": "The phone number for the purchase", 
    "membership level": "The membership level for the purchase",
    "payment method": "The payment method for the purchase", 
    "username": "The username for the purchase",
    "order id": "The order id for the purchase",
    "email": "The email for the purchase",
    "zip code": "The zip code for the purchase",
    "company team": "The company team dealing with the purchase dispute", 
    "reason slotval": "The reason for the dispute",
    "order slotval": "The order detail of the purchase"},
  "account acces": {
    "email": "The email for the account",
    "details slotval": "The details of the account",
    "amount": "The amount related to the account",
    "order slotval": "The order of the account",
    "phone": "The phone number of the account",
    "pin number": "The pin number of the account",
    "shipping option": "The shipping method for the account",
    "zip code": "The zip code of the account",
    "username": "The username of the account",
    "membership level": "The membership level of the account",
    "company team": "The company team dealing with the account access",
    "customer name": "The customer's name for the account"},
  "shipping issue": {
    "customer name": "The customer's name for the shipping issue",
    "company team": "The company team dealing with the shipping issue",
    "username": "The username for the shipping issue",
    "email": "The email for the shipping issue",
    "product": "The product for the shipping issue",
    "amount": "The amount related to the shipping issue",
    "account id": "The account id for the shipping issue",
    "order id": "The order id for the shipping issue",
    "order slotval": "The order detail of the shipping issue", 
    "membership level": "The membership level for the shipping issue",
    "payment method": "The payment method for the shipping",
    "reason slotval": "The reason for the shipping issue", 
    "refund target": "The refund method for the shipping issue",
    "change option": "The detail being changed for the shipping issue",
    "full address": "The address for the shipping issue",
    "membership level": "The membership level for the shipping issue",
    "street address": "The street address for the shipping issue",
    "zip code": "The zip code for the shipping issue",
    "shipping option": "The shipping option for the shipping issue"},
  "subscription inquiry": {
    "account slotval": "The account details of the subscription",
    "account id": "The account id of the subscription",
    "order id": "The order id for the subscription",
    "refund target": "The refund method for the subscription",
    "zip code": "The zip code of the subscription",
    "amount": "The amount of the subscription", 
    "shipping option": "The shipping method of the subscription",
    "product": "The product name of the subscription",
    "details slotval": "The details of the subscription", 
    "membership level": "The membership level for the subscription",
    "company team": "The company team dealing with the subscription",
    "payment method": "The payment method for the subscription",
    "customer name": "The customer's name for the subscription"},
  "troubleshoot site": {
    "amount": "The amount related to the website issue",
    "product": "The product for the website issue",
    "email": "The email for the website issue",
    "details slotval": "The details of the website issue",
    "order slotval": "The order for the website issue",
    "membership level": "The membership level for the website issue", 
    "account slotval": "The account details related to the website issue",
    "username": "The username for the website issue",
    "company team": "The company team dealing with the website issue",
    "customer name": "The customer's name for the website issue",
    "shipping option": "The shipping method for the website issue",
  }
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
    "area": "Which area of the restaurant is the user interested in",
    "people": "How many people is the restaurant booking for",
    "day": "What day is the restaurant booking for",
    "time": "When is time of the restaurant booking",
    "food": "What type of food is the user interested in",
    "name": "What is the name of the restaurant",
    "price range": "What is the price range of the restaurant",
    "location": "Where is the restaurant",
    "date": "What date is it for the restaurant booking",
    "category": "What type of food is the user interested in",
    "meal": "What is the meal of the restaurant",
    "num people": "How many people is the restaurant booking for",
    "live music": "Does the restaurant have live music",
    "has vegetarian options": "Does the restaurant have vegetarian options",
    "address": "What is the address of the restaurant",
    "rating": "What is the rating for the restaurant",
    "outdoor seating": "Does the restaurant has outdoor seating",
    "alcohol": "Does the restaurant allow alcohol",
    "phone number": "What is the phone number of the restaurant",
    "restaurant name": "What is the name of the restaurant",
    },
  "movie": {
    "theater": "What is the name of the theater",
    "theatre name": "What is the name of the theater",
    "genre": "What genre is the movie",
    "screening": "What is the screening of the movie", 
    "movie": "What is the name of the movie",
    "show date": "What date is the movie on",
    "date": "What date is the movie on",
    "location": "Where is the movie on",
    "num tickets": "How many tickets does the user want",
    "preferred time": "What time of the movie does the user prefer",
    "show time": "When is the show time of the movie",
    "time": "When is the time of the movie",
    },
  "taxi": {
    "arrive by": "When is the arrive by time of the taxi",
    "destination": "Where is the destination of the taxi",
    "departure": "Where is the departure of the taxi",
    "leave at": "When is the leave at time of the taxi",
    }, 
  "rideshare": {
    "duration": "How long is the duration of the ride",
    "destination": "Where is the destination location of the ride",
    "wait time": "How long is the wait time for the ride",
    "fare": "How much is the rideshare fare",
    "shared ride": "Is the rideshare a shared ride",
    "type": "What type of car is it for the ride",
    "number of seats": "How many seats are there for the rideshare",
    },
  "rental": {
    "price": "How much is the rental",
    "pickup time": "When is the pickup time for the rental",
    "dropoff date": "When is the dropoff date for the rental",
    "pickup date": "When is the pickup date for the rental",
    "insurance": "Does the rental include insurance",
    "name": "What is the name of the rental",
    "leave at": "When is the leave at time for the rental",
    "departure": "When is the departure time for the rental",
    "type": "What type of car is it for the rental",
    "pickup city": "Where is the pickup city of the rental",
    },
  "train": {
    "arrive by": "When is the arrive by time of the train",
    "people": "How many people is the train booking for",
    "day": "What day is the train booking for",
    "destination": "Where is the destination of the train",
    "departure": "Where is the departure of the train",
    "leave at": "When is the leave at time of the train",
    "trip protection": "Does the train offer trip protection",
    "arrival": "When is the arrival time of the train",
    "price": "How much is the price of the train ticket",
    "class": "What class is the train ride",
    "date": "When is the date of the train booking",
    "origin": "Where is the origin of the train",
    "number of seats": "How many number of seats is there on the train",
    },
  "hotel": {
    "area": "Which area of the hotel is the user interested in",
    "day": "What is the start day of the hotel",
    "stay": "How long is the stay at the hotel",
    "people": "How many people is the hotel for",
    "internet": "Does the hotel offer internet",
    "name": "What is the name of the hotel",
    "parking": "Does the hotel offer parking",
    "price range": "What is the price range of the hotel",
    "stars": "How many stars does the hotel have",
    "type": "What type of hotel is the user interested in",
    "rating": "how many rating stars does the hotel have",
    "address": "What is the address of the hotel",
    "number of rooms": "How many rooms is there for the hotel booking",
    "laundry": "Does the hotel offer laundry service",
    "pets welcome": "Are pets welcome at the hotel",
    "smoking": "Is smoking allowed at the hotel",
    "number of days": "How many days is the hotel booking",
    "check out": "When is the check out date for the hotel booking",
    "check in": "When is the check in date for the hotel booking",
    "location": "Where is the hotel",
    "phone number": "What is the phone number of the hotel",
    "street address": "What is the address of the hotell"
    },
  "attraction": {
    "area": "Which area of the attraction is the user interested in",
    "name": "What is the name of the attraction",
    "type": "What type of attraction is the user interested in"
    },  
  "event": {
    "time": "What time is the event",
    "address": "What is the address of the event",
    "city": "Which city is the event held in",
    "price": "What is the price of the event",
    "venue": "What is the venue of the event",
    "category": "What type is the event",
    "date": "What date is the event on",
    "name": "What is the name of the event",
    "num people": "How many people are joining the event",
    "type": "What type is the event"
    },
  "travel": {
    "location": "What is the location of travel",
    "free entry": "Does the travel event have free entry",
    "type": "What type is the travel",
    "good for kids": "Is the travel event good for kids",
    "name": "What is the name for travel",
    "phone number": "What is the phone number for travel"
    },
  "manage account": {
    "customer name": "What is the customer's name for the account",
    "street address": "What is the street address of the account",
    "zip code": "What is the zip code of the account",
    "product": "What is the product related to the account",
    "phone": "What is the phone number on the account",  
    "username": "What is the username of the account", 
    "shipping option": "What is the shipping method on the account",
    "account slotval": "What is the details of the account",
    "payment method": "What is the payment method of the account",
    "amount": "What is the amount related to the account",
    "order id": "What is the order id for the account",
    "membership level": "What is the membership level of the account",
    "order slotval": "Which order is related to the account",
    "reason slotval": "What is the reason for managing the account", 
    "company team": "Which company team is managing the account"},
  "storewide query": {
    "change option": "What is the detail being changed for the query", 
    "customer name": "What is the customer's name for the query",
    "account slotval": "Which account is it for the query",
    "company team": "Which company team is dealing with query",
    "reason slotval": "What is the reason for the query",
    "order slotval": "Which order is related to the query",
    "shipping option": "What is the shipping method related to the query",
    "membership level": "What is the membership level of the query",
    "payment method": "What is the payment method for the query",
    "product": "What is the product of the query",
    "order slotval": "What is the order of the query",
    "zip code": "What is the zip code for the query",
    },
  "single item query": {
    "product": "What is the product of the query",
    "amount": "How much is the amount related to the query",
    "shipping option": "What is the shipping method of the query",
    "company team": "Which company team is related to the query",
    "membership level": "What is the membership level related to the query",
    "customer name": "What is the customer's name for the query",
    },
  "order issue": {
    "amount": "How much is the amount of the order",
    "account id": "What is the account id of the order",
    "customer name": "What is the customer name for the order",
    "shipping option": "What is the shipping method for the order",
    "order id": "What is the order id for the order",
    "change option": "What is the detail being changed for the order issue",
    "order slotval": "What is the value of the order",
    "account slotval": "What is the account for the order",
    "reason slotval": "What is the reason for the order issue",
    "membership level": "What is the membership level for the order", 
    "payment method": "What is the payment method for the order",
    "refund target": "What is the refund method for the order",
    "zip code": "What is the zip code for the order",
    "product": "What is the product name for the order", 
    "company team": "Which company team is dealing with the order",
    },
  "product defect": {
    "email": "What is the email for the product defect",
    "account id": "What is the account id for the product",
    "membership level": "What is the membership level for the product defect",
    "order id": "What is the order id for the product defect",
    "username": "What is the username for the product defect",
    "order slotval": "What is the order detail of the product", 
    "shipping option": "What is the shipping method for the product",
    "details slotval": "What is the details of the product defect", 
    "reason slotval": "What is the reason for the product defect",
    "product": "What is the name of the product",
    "amount": "How much is the amount of the product defect", 
    "zip code": "What is the zip code for the product defect",
    "payment method": "What is the payment method for the product",
    "refund target": "What is the refund method for the product defect",
    "company team": "which company team is dealing with the product defect",
    "customer name": "What is the customer's name for the product defect",
    },
  "purchase dispute": {
    "customer name": "What is the customer's name for the purchase",
    "product": "What is the product for the purchase",
    "account slotval": "What is the account of the purchase dispute",
    "amount": "How much is the purchase dispute",
    "shipping option": "What is the shipping method of the purchase dispute",
    "change option": "What is the detail being changed for the purchase",
    "account id": "What is the account id for the purchase",
    "phone": "What is the phone number for the purchase", 
    "membership level": "What is the membership level for the purchase",
    "payment method": "What is the payment method for the purchase", 
    "username": "What is the username for the purchase",
    "order id": "What is the order id for the purchase",
    "email": "What is the email for the purchase",
    "zip code": "What is the zip code for the purchase",
    "company team": "Which company team is dealing with the purchase dispute", 
    "reason slotval": "What is the reason for the dispute",
    "order slotval": "What is the order detail of the purchase",
    },
  "account acces": {
    "email": "What is the email for the account",
    "details slotval": "What is the details of the account",
    "amount": "How much is the amount related to the account",
    "order slotval": "What is the order of the account",
    "phone": "What is the phone number of the account",
    "pin number": "What is the pin number of the account",
    "shipping option": "What is the shipping method for the account",
    "zip code": "What is the zip code of the account",
    "username": "What is the username of the account",
    "membership level": "Which membership level is the account on",
    "company team": "Which company team is dealing with the account access",
    "customer name": "What is the customer's name for the account",
    },
  "shipping issue": {
    "customer name": "What is the customer's name for the shipping issue",
    "company team": "Which company team is dealing with the shipping issue",
    "username": "What is the username for the shipping issue",
    "email": "What is the email for the shipping issue",
    "product": "What is the product for the shipping issue",
    "amount": "How much is the amount related to the shipping issue",
    "account id": "What is the account id for the shipping issue",
    "order id": "What is the order id for the shipping issue",
    "order slotval": "What is the order detail of the shipping issue", 
    "membership level": "What is the membership level for the shipping issue",
    "payment method": "What is the payment method for the shipping",
    "reason slotval": "What is the reason for the shipping issue", 
    "refund target": "What is the refund method for the shipping issue",
    "change option": "What is the detail being changed for the shipping issue",
    "full address": "What is the address for the shipping issue",
    "street address": "What is the street address for the shipping issue",
    "zip code": "What is the zip code for the shipping issue",
    "shipping option": "What is the shipping option for the shipping issue",
    },
  "subscription inquiry": {
    "account slotval": "What is the account details of the subscription",
    "account id": "What is the account id of the subscription",
    "order id": "What is the order id for the subscription",
    "refund target": "What is the refund method for the subscription",
    "zip code": "What is the zip code of the subscription",
    "amount": "How much is the subscription", 
    "shipping option": "What is the shipping method of the subscription",
    "product": "What is the product name of the subscription",
    "details slotval": "What is the details of the subscription", 
    "membership level": "What is the membership level for the subscription",
    "company team": "Which company team is dealing with the subscription",
    "payment method": "What is the payment method for the subscription",
    "customer name": "What is the customer's name for the subscription"},
  "troubleshoot site": {
    "amount": "How much is the amount related to the website issue",
    "product": "What is the product for the website issue",
    "email": "What is the email for the website issue",
    "details slotval": "What is the details of the website issue",
    "order slotval": "What is the order for the website issue",
    "membership level": "What is the membership level for the website issue", 
    "account slotval": "What is the account details related to the website issue",
    "username": "What is the username for the website issue",
    "company team": "Which company team is dealing with the website issue",
    "customer name": "What is the customer's name for the website issue",
    "shipping option": "What is the shipping method for the website issue",
  }
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

