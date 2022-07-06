import argparse
import os


def solicit_params():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input-dir", default='assets', type=str, 
                help="The input training data file (a text file).")
    parser.add_argument("--output-dir", default='results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--dataset", default='mwoz', type=str, 
                choices=['abcd', 'dstc', 'gsim', 'mwoz', 'sgd', 'tt'],
                help="which dataset to choose from out of all possible options")
    parser.add_argument("--task", default='fine_tune', type=str,
                choices=['in_context', 'meta_learn', 'fine_tune', 'reptile'],
                help="in context has no gradients, meta learn uses inner loop gradients to \
                improve outer loop performance, fine tune performs outer loop training only")
    parser.add_argument("--model", default='gpt', type=str, choices=['t5', 'bart', 'gpt', 'trade'],
                help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--size", default='small', type=str, choices=['small', 'medium', 'large'],
                help="Size of the model, use small for debugging, but report results on large")
    parser.add_argument("--checkpoint", default='', type=str,
                help="Enter the filename of a checkpoint for manual override")
    parser.add_argument("--seed", default=42, type=int)

    # Custom paper parameters
    parser.add_argument("--num-shots", default="full", type=str,
                choices=["point", "one", "five", "ten", "full"], help="point allows only 0.1 of data, \
                while full is used with all training data, 1/5/10 is percent of training data")
    parser.add_argument("--threshold", default=1.1, type=float,
                help="Used as the repetition penalty during inference of generation")
    parser.add_argument("--temperature", default=1.0, type=float,
                help="Temperature for increasing diversity when decoding, mainly for paraphrase")
    parser.add_argument("--style", default='dataset', type=str, choices=['domain', 'dataset','prompt'],
                help="Subset of data held out for testing. For example, if domain is the chosen style, \
                then we meta learn on [taxi, hotel, restaurant, train] and test on [attraction].")
    parser.add_argument("--left-out", default='', type=str,   # see args.style
                help="Name of the domain or dataset left out of training and used for testing only")
    parser.add_argument("--prompt-style", default="none", type=str, help='type of prompt', 
                choices=["schema", "question", "statement", "naive", "human", "none", "random"])
    parser.add_argument("--maximum-length", default=1024, type=int,
                help="Maximum length of sequences for model input")
    parser.add_argument("--context-length", default=2, type=int,
                help="Number of turns to look back into dialogue context, eats into token length")

    # SBERT Retriever params
    parser.add_argument("--search", default="oracle", type=str, help="find similar examples for context",  
                choices=["oracle", "cosine", "euclidean", "mahalanobis"])
    parser.add_argument("--loss-function", default="cosine", type=str,
                choices=["contrast", "cosine", "custom", "default", "zero_one"], help="loss function for fine-tuning")
    parser.add_argument("--kappa", default=1000, type=int, 
                help="Number of examples to use as negatives during constrastive training")
    parser.add_argument("--filter", action="store_true",
                help="Filter out non-salient utterances as predicted by logistic regression")
    parser.add_argument("--use-tuned", action="store_true",
                help="Use the fine tuned SBERT model rather than the default one")

    # Key settings
    parser.add_argument("--ignore-cache", action="store_true",
                help="Whether to ignore cache and create a new input data")
    parser.add_argument("--debug", action="store_true",
                help="Whether to run in debug mode which is exponentially faster")
    parser.add_argument("--verbose", action="store_true",
                help="Whether to run with extra prints to help debug")
    parser.add_argument("--do-train", action="store_true",
                help="Whether to run training.")
    parser.add_argument("--do-eval", action="store_true",
                help="Whether to run eval on the test set.")
    parser.add_argument("--do-save", action="store_true",
                help="Whether to save models, which override previous checkpoints")
    parser.add_argument("--do-leave", action='store_true',
                help="Run the leftout dataset as extra evaluation")
    parser.add_argument("--log-interval", type=int, default=500,
                help="Log every X updates steps.")
    parser.add_argument("--eval-interval", default='whole', type=str, 
                choices=['tenth', 'quarter', 'half', 'whole'],
                help="Ratio of dev data to process before printing out a score")
    parser.add_argument("--checkpoint-interval", default=-1, type=int, 
                help="The number of update steps to save a checkpoint for meta-learning")
    parser.add_argument("--chunk-ratio", default=-1, type=float, 
                help="chunk_ratio * step_per_epoch = checkpoint_interval")
    parser.add_argument("--qualify", action='store_true',
                help="Whether to include joint accuracy scores during evaluation")
    parser.add_argument("--quantify", action='store_true',
                help="Whether to include inform/success/BLEU scores during evaluation")
    parser.add_argument("--prune-keep", default=-1, type=int,
                help="Number of models to keep around after pruning, by default does not prune")
    parser.add_argument("--parallel", action="store_true",
                help="Whether to run in parallel")
    parser.add_argument("--ensemble", default=-1, type=int,
                help="setting up the ensembling size")
    parser.add_argument("--filter-threshold", default=0.4, type=float,
                help="Used as the threshold for filtering irrelevant sentence in saliency model")
    parser.add_argument("--train-percent", default=-1.0, type=float,
                help="percentage of training data for fine-tuning")

    # Hyper-parameters for tuning
    parser.add_argument("--batch-size", default=12, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument('--grad-accum-steps', default=1, type=int,
                help='Number of steps for gradient accumulation')
    parser.add_argument("--learning-rate", default=3e-4, type=float,
                help="Model learning rate starting point.")
    parser.add_argument("--drop-rate", default=0.1, type=float,
                help="Dropout rate with default of 10%")
    parser.add_argument("--glove-dim", default=0, type=int,
                help="Hidden dimension for glove, either 100 or 300 when used")
    parser.add_argument("--hidden-dim", default=300, type=int,
                help="Hidden dimension for intermediate projection of ranking model")
    parser.add_argument("--embed-dim", default=1024, type=int,
                help="The embedding dimension of pretrained LM.")
    parser.add_argument("--weight-decay", default=0.0, type=float,
                help="Weight decay if we apply some.")
    parser.add_argument("--n-epochs", default=3, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument("--warmup-steps", default=0, type=int, 
                help="Linear warmup over warmup-steps.")
    parser.add_argument("--percent", default=1.0, type=float, 
                help="Amount of train and dev data to use during training.")

    args = parser.parse_args()
    return args
