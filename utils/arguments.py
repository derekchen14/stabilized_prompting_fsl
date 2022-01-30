import argparse
import os

def solicit_params():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input-dir", default='assets', type=str, 
                help="The input training data file (a text file).")
    parser.add_argument("--output-dir", default='results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--dataset", default='gsim', type=str, 
                choices=['gsim', 'abcd', 'mwoz', 'tt', 'sgd', 'dstc'],
                help="which dataset to choose from out of the ten options")
    parser.add_argument("--task", default='classify', type=str,
                choices=['classify', 'track', 'generate'],
                help="Classify corresponds to intent classification, Track refers to dialogue \
                    state tracking and Generate refers to response generation")
    parser.add_argument("--model", default='roberta', type=str, choices=['roberta', 'bart', 'gpt'],
                help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--size", default='small', type=str, choices=['small', 'medium', 'large'],
                help="Size of the model, use small for debugging, but report results on large")
    parser.add_argument("--style", default='domains', type=str,
                help="specific style name, needed to pull the correct label list from the ontology")
    parser.add_argument("--checkpoint", default='', type=str,
                help="Enter the filename of a checkpoint for manual override")
    parser.add_argument("--seed", default=42, type=int)

    # Custom paper parameters
    parser.add_argument("--prune-keep", default=-1, type=int,
                help="Number of models to keep around after pruning, by default does not prune")
    parser.add_argument("--num-shots", default="zero", type=str,
                choices=["zero", "few", "percent"], help="zero-shot and few-shot learning load \
                    an untrained model, percent loads a model fine-tuned on X% of data")
    parser.add_argument("--threshold", default=0.25, type=float,
                help="Determines the amount of data used for pre-training the model; See num-shots")
    parser.add_argument("--temperature", default=1.4, type=float,
                help="Temperature for increasing diversity when decoding, mainly for paraphrase")
    parser.add_argument("-k", "--kappa", default=1, type=int,
                help="Integer param: could be num clusters, dimensions of NT matrix or other")
    parser.add_argument("--max-len", default=1024, type=int,
                help="Maximum length of sequences for model input")

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
    parser.add_argument("--do-interact", action="store_true",
                help="When turned on, you dynamically feed in a prompt for inference")
    parser.add_argument("--log-interval", type=int, default=500,
                help="Log every X updates steps.")
    parser.add_argument("--qualify", action='store_true',
                help="Whether to include joint accuracy scores during evaluation")
    parser.add_argument("--quantify", action='store_true',
                help="Whether to include inform/success/BLEU scores during evaluation")
    
    # Hyper-parameters for tuning
    parser.add_argument("--batch-size", default=12, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument('--grad-accum-steps', default=1, type=int,
                help='Number of steps for gradient accumulation')
    parser.add_argument("--learning-rate", default=3e-5, type=float,
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
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=3, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument("--warmup-steps", default=0, type=int, 
                help="Linear warmup over warmup-steps.")

    args = parser.parse_args()
    return args
