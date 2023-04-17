# Python file in order to test subspace inference techniques with DistilBERT
import torch
import argparse
import os, sys
import numpy as np
import time
import tabulate
import logging
import argparse
import torch.nn.functional as F
import datasets
import transformers

from accelerate.logging import get_logger
from accelerate import Accelerator
from transformers import (PretrainedConfig, 
                          AutoConfig, 
                          AutoTokenizer, 
                          AutoModelForSequenceClassification)
from datasets import load_dataset

from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
import dswag

import data, losses, utils
sys.path.remove(os.path.abspath('../si_local_components'))

logger = get_logger(__name__)

# Using relevant arguments for argparsers
parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
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
parser.add_argument("--task_name", type=str, default=None, help="The name of the glue task to train on.")
parser.add_argument("--with_tracking", action="store_true", help="Whether to enable experiment trackers for logging.",)
parser.add_argument("--report_to", type=str, default="all", help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
parser.add_argument("--use_slow_tokenizer", action="store_true",  help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",)
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
args = parser.parse_args()


send_example_telemetry("run_glue_no_trainer", args)

# Initializing the accelerator
accelerator = (Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator())

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

# Handle the repository creation
args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')
accelerator.wait_for_everyone()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Using datasets coming from GLUE
# This (and subsequent code) may not work for regression task (stsb)
raw_datasets = load_dataset("glue", args.task_name)
label_list = raw_datasets["train"].features["label"].names
num_labels = len(label_list)

shuffle_train = True
loaders = \
        {
            'train': torch.utils.data.DataLoader(
                raw_datasets["train"],
                batch_size=args.batch_size,
                shuffle=True and shuffle_train,
                num_workers=args.num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                raw_datasets["test"],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            ),
        }, \
    


# Defined config for dataset preprocessing
dBERT = torch.load('DistilBERT-Results/pytorch_model.bin')


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
                    *dBERT.args, **dBERT.kwargs)
    swag_model.to(args.device)
else:
    print('SGD training')

def schedule(epoch):
    t = (epoch) / (args.swag_start if args.swag else args.epochs)
    lr_ratio = args.swag_lr / args.lr_init if args.swag else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor

# use a slightly modified loss function that allows input of model 
if args.loss == 'CE':
    criterion = losses.cross_entropy
    #criterion = F.cross_entropy
elif args.loss == 'adv_CE':
    criterion = losses.adversarial_cross_entropy



# ADAM as opposed to standard SGD for DistilBERT

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in dBERT.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in dBERT.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]


optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr_init)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    dBERT.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if args.swag and args.swag_resume is not None:
    checkpoint = torch.load(args.swag_resume)
    swag_model.subspace.rank = torch.tensor(0)
    swag_model.load_state_dict(checkpoint['state_dict'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', 'mem_usage']
if args.swag:
    columns = columns[:-2] + ['swa_te_loss', 'swa_te_acc'] + columns[-2:]
    swag_res = {'loss': None, 'accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=dBERT.state_dict(),
    optimizer=optimizer.state_dict()
)

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0.

import ipdb
ipdb.set_trace()

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    if not args.no_schedule:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init
    
    if (args.swag and (epoch + 1) > args.swag_start) and args.cov_mat:
        train_res = utils.train_epoch(loaders['train'], dBERT, criterion, optimizer, cuda = False)
    else:
        train_res = utils.train_epoch(loaders['train'], dBERT, criterion, optimizer, cuda = False)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], dBERT, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None}

    if args.swag and (epoch + 1) > args.swag_start and (epoch + 1 - args.swag_start) % args.swag_c_epochs == 0:
        #sgd_preds, sgd_targets = utils.predictions(loaders["test"], model)
        sgd_res = utils.predict(loaders["test"], dBERT)
        sgd_preds = sgd_res["predictions"]
        sgd_targets = sgd_res["targets"]
        # print("updating sgd_ens")
        if sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            #TODO: rewrite in a numerically stable way
            sgd_ens_preds = sgd_ens_preds * n_ensembled / (n_ensembled + 1) + sgd_preds/ (n_ensembled + 1)
        n_ensembled += 1
        swag_model.collect_model(dBERT)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            swag_model.set_swa()
            utils.bn_update(loaders['train'], swag_model)
            swag_res = utils.eval(loaders['test'], swag_model, criterion)
        else:
            swag_res = {'loss': None, 'accuracy': None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=dBERT.state_dict(),
            optimizer=optimizer.state_dict()
        )
        if args.swag:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                name='swag',
                state_dict=swag_model.state_dict(),
            )
            
    elif args.save_iterates:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=dBERT.state_dict(),
            optimizer=optimizer.state_dict()
        )

    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated()/(1024.0 ** 3)
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep, memory_usage]
    if args.swag:
        values = values[:-2] + [swag_res['loss'], swag_res['accuracy']] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=dBERT.state_dict(),
        optimizer=optimizer.state_dict()
    )
    if args.swag and args.epochs > args.swag_start:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            name='swag',
            state_dict=swag_model.state_dict(),
        )

        utils.set_weights(dBERT, swag_model.mean)
        utils.bn_update(loaders['train'], dBERT)
        print("SWA solution")
        print(utils.eval(loaders['test'], dBERT, losses.cross_entropy))

        utils.save_checkpoint(
            args.dir,
            name='swa',
            state_dict=dBERT.state_dict(),
        )

if args.swag:
    np.savez(os.path.join(args.dir, "sgd_ens_preds.npz"), predictions=sgd_ens_preds, targets=sgd_targets)
