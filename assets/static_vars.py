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
    'roberta': {
        'small': 'roberta-base', 
        'medium': 'roberta-large',
        'large': 'roberta-xlarge' },
    'gpt': {
        'small': 'gpt2',
        'medium': 'gpt2-medium',
        'large': 'EleutherAI/gpt-j-6B'},
    'bart': {
        'small': 'facebook/bart-base',
        'medium': 'facebook/bart-large',
        'large': 'facebook/bart-xlarge'}
}
