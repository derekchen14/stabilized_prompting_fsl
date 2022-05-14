import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
debug_break = 5

metric_by_task = {
    'clc': 'accuracy',
    'tlc': 'accuracy',
    'dst': 'f1_score',
    'rg': 'bow_similarity',
    'ir': 'recall@5'
}

STOP_TOKENS = ['done', 'exit', 'logout', 'finish', 'stop']

DATASETS = {
    'abcd': 'Action-Based Conversations Dataset',
    'dstc': 'Dialogue State Tracking Challenge 2',
    'gsim': 'Google Simulated Dialogue',
    'mwoz': 'MultiWoz 2.2',
    'sgd': 'Schema Guided Dialogue',
    'tt': 'TicketTalk'
}

CHECKPOINTS = {
    't5': {
        'small': 't5-small', 
        'medium': 't5-base',
        'large': 't5-11b' },
    'gpt': {
        'small': 'gpt2',
        'medium': 'gpt2-large',
        'large': 'EleutherAI/gpt-j-6B'},
    'bart': {
        'small': 'facebook/bart-base',
        'medium': 'facebook/bart-large',
        'large': 'facebook/bart-xlarge'}
}

GENERAL_TYPO = {
    # type
    "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "churches":"church",
    "mutiple sports":"multiple sports", "sports":"multiple sports", "mutliple sports":"multiple sports",
    "swimmingpool":"swimming pool", "concerthall":"concert hall", "concert":"concert hall",
    "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture",
    "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", 
    # area
    "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north",
    "cen":"centre", "east side":"east", "east area":"east", "west part of town":"west", "ce":"centre",
    "town center":"centre", "centre of cambridge":"centre", "city center":"centre", "the south":"south",
    "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north",
    "centre of town":"centre", "cb30aq": "none",
    # price
    "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate",
    # day
    "next friday":"friday", "monda": "monday",
    # names
    "catherine s": "catherines",
    # parking
    "free parking":"free",
    # internet
    "free internet":"yes", "y":"yes",
    # star
    "4 star":"4", "4 stars":"4", "0 star rarting":"<none>", "3 .":"3",
    # others
    "n":"no", "does not":"no", "does not care":"any", "dontcare": "any",
    "not men":"none", "not":"none", "art":"none", "not mendtioned":"none", "fun":"none",
}

SLOT_MAPPING = {
  "arriveby": "arrive by",
  "leaveat": "leave at",
  "pricerange": "price range",
  "date release": "release date",
  "description other": "description",
  "name character": "character name",
  "name genre": "genre",
  "name movie": "movie name",
  "name person": "person",
  "name theater": "theater",
  "time preference": "preferred time",
  "time showing": "showtime",
}

"""
<customer> i would like some soup and crackers.
<agent> what kind of soup do you want?
<customer> i want cream of broccoli.

speaker is "customer"
text is "i would like cream of broccoli."
utterance is "<customer> i would like cream of broccoli."
    utterance includes: speaker + text
    there are 3 utterances in this conversation
history is first two utterances
we do not use the term "context" anywhere
    context will always refer to support set examples
history + current_utt = dialogue
"""
