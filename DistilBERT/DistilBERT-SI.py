# Python file in order to test subspace inference techniques with DistilBERT
import torch
import os, sys
import numpy as np
import argparse
import torch.nn.functional as F
import datasets
import transformers

from accelerate.logging import get_logger
from transformers import PretrainedConfig, AutoConfig
from datasets import load_dataset
from ../posteriors import 
from .. import data, losses, utils

logger = get_logger(__name__)

# Using relevant arguments for argparsers
parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.01, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--swag', action='store_true')
parser.add_argument('--swag_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swag_lr', type=float, default=0.02, metavar='LR', help='SWA LR (default: 0.02)')
parser.add_argument('--swag_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--cov_mat', action='store_true', help='save sample covariance')
parser.add_argument('--max_num_models', type=int, default=20, help='maximum number of SWAG models to save')
parser.add_argument('--swag_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')
parser.add_argument('--loss', type=str, default='CE', help='loss to use for training model (default: Cross-entropy)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--no_schedule', action='store_true', help='store schedule')
parser.add_argument('--save_iterates', action='store_true', help='save all iterates in the SWA(G) stage (default: off)')
parser.add_argument('--inference', choices=['low_rank_gaussian', 'projected_sgd'], default='low_rank_gaussian')
parser.add_argument('--subspace', choices=['covariance', 'pca', 'freq_dir'], default='covariance')
args = parser.parse_args()

args.device = None
if torch.cude.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


print('Using fine-tuned DistilBERT model')

# Loading in DistilBERT model parameters
dBERT = torch.load('DistilBERT-Results/pytorch_model.bin')

# Using GLUE dataset
glue_dataset = load_dataset("glue", 'cola')
label_list = glue_dataset["train"].features["label"].names
num_labels = len(label_list)

# Defined config for dataset preprocessing
config = AutoConfig.from_pretrained('distilbert-base-uncased', num_labels=num_labels, finetuning_task = 'cola')

# Changing set order of labels in model
label_to_id = None
if (
    dBERT.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in dBERT.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
if label_to_id is not None:
        dBERT.config.label2id = label_to_id
        dBERT.config.id2label = {id: label for label, id in config.label2id.items()}


dBERT.to(args.device)

if args.cov_mat:
    args.no_cov_mat = False
else:
    args.no_cov_mat = True

if args.swag:
    print('SWAG training')
    swag_model = SWAG(dBERT.base, 
                    subspace_type=args.subspace, subspace_kwargs={'max_rank': args.max_num_models},
                    *dBERT.args, num_classes=num_classes, **dBERT.kwargs)
    swag_model.to(args.device)
else:
    print('SGD training')
