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

TASKS = {
    'classify': 'Intent Classification',
    'track': 'Dialogue State Tracking',
    'generate': 'Response Generation'
}

DATASETS = {
    'abcd': 'Action-Based Conversations Dataset',
    'sgd': 'Schema Guided Dialogue',
    'dstc': 'Dialogue State Tracking Challenge 2',
    'gsim': 'Google Simulated Dialogue',
    'mwoz': 'MultiWoz',
    'tt': 'TicketTalk'
}

CHECKPOINTS = {
    'roberta': 'roberta-large',
    'gpt': 'gpt2-medium',
    'bart': 'facebook/bart-large'
}
