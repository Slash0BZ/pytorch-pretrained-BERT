# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSingleTokenClassification, BertForSingleTokenClassificationFollowTemporal, TemporalModelJoint, TemporalModelJointWithLikelihood, BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertForSingleTokenClassificationWithPooler, TemporalModelArguments
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, label_a, label_b, target_idx_a,
                 target_idx_b):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label_a = label_a
        self.label_b = label_b
        self.target_idx_a = target_idx_a
        self.target_idx_b = target_idx_b


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, lm_labels, target_idx_a, target_idx_b, soft_labels_a, soft_labels_b):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_labels = lm_labels
        self.target_idx_a = target_idx_a
        self.target_idx_b = target_idx_b
        self.soft_target_a = soft_labels_a
        self.soft_target_b = soft_labels_b


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class TemporalVerbProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        f = open(os.path.join(data_dir, "train.formatted.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        examples = self._create_examples(lines, "train")
        return examples

    def get_dev_examples(self, data_dir):
        f = open(os.path.join(data_dir, "test.formatted.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        for i in range(0, len(lines) - 1, 2):
            groups_a = lines[i].split("\t")
            groups_b = lines[i+1].split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = groups_a[0]
            if len(groups_a) > 1:
                target_idx_a = int(groups_a[1])
                label_a = text_a.split()[target_idx_a].lower()
            else:
                target_idx_a = 0
                label_a = "seconds"
            text_b = groups_b[0]
            if len(groups_b) > 1:
                target_idx_b = int(groups_b[1])
                label_b = text_b.split()[target_idx_b].lower()
            else:
                target_idx_b = 0
                label_b = "seconds"

            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label_a=label_a, label_b=label_b,
                    target_idx_a=target_idx_a, target_idx_b=target_idx_b
                )
            )

        return examples


def randomize_likelihood_labels(l):
    ret = [-1] * l
    for i in range(1, l - 1):
        prob = random.random()
        if prob < 0.1:
            ret[i] = 0
    return ret


def random_word(tokens, target_id, tokenizer):
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                """CHANGED: NEVER PREDICT [UNK]"""
                # output_label.append(tokenizer.vocab["[UNK]"])
                output_label.append(-1)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    tmp_prob = random.random()
    # Sum with 0.15 to 80
    # Yet never compute loss until feedback
    output_label[target_id] = -1
    if tmp_prob < 0.65:
        tokens[target_id] = "[MASK]"

    return tokens, output_label


def assign_round_soft_label(length, target):
    label_vector = [0.157, 0.001, 0.0]
    label_vector_rev = [0.0, 0.001, 0.157]
    ret_vec = [0.0] * length
    ret_vec[target] = 0.683
    for i in range(target + 1, target + 4):
        cur_target = i
        if i >= length:
            cur_target -= length
        ret_vec[cur_target] = label_vector[i - target - 1]
    for i in range(target - 1, target - 4, -1):
        cur_target = i
        ret_vec[cur_target] = max(ret_vec[cur_target], label_vector_rev[i - target + 3])
    return ret_vec


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_soft_labels(orig_label, vocab_size=30522):
    keywords = {
        "second": [0, 0],
        "seconds": [0, 0],
        "minute": [0, 1],
        "minutes": [0, 1],
        "hour": [0, 2],
        "hours": [0, 2],
        "day": [0, 3],
        "days": [0, 3],
        "week": [0, 4],
        "weeks": [0, 4],
        "month": [0, 5],
        "months": [0, 5],
        "year": [0, 6],
        "years": [0, 6],
        "decade": [0, 7],
        "decades": [0, 7],
        "century": [0, 8],
        "centuries": [0, 8],
        "dawns": [1, 0],
        "mornings": [1, 1],
        "noons": [1, 2],
        "afternoons": [1, 3],
        "evenings": [1, 4],
        "dusks": [1, 5],
        "nights": [1, 6],
        "midnights": [1, 7],
        "dawn": [1, 0],
        "morning": [1, 1],
        "noon": [1, 2],
        "afternoon": [1, 3],
        "evening": [1, 4],
        "dusk": [1, 5],
        "night": [1, 6],
        "midnight": [1, 7],
        "monday": [2, 0],
        "tuesday": [2, 1],
        "wednesday": [2, 2],
        "thursday": [2, 3],
        "friday": [2, 4],
        "saturday": [2, 5],
        "sunday": [2, 6],
        "mondays": [2, 0],
        "tuesdays": [2, 1],
        "wednesdays": [2, 2],
        "thursdays": [2, 3],
        "fridays": [2, 4],
        "saturdays": [2, 5],
        "sundays": [2, 6],
        "january": [3, 0],
        "february": [3, 1],
        "march": [3, 2],
        "april": [3, 3],
        "may": [3, 4],
        "june": [3, 5],
        "july": [3, 6],
        "august": [3, 7],
        "september": [3, 8],
        "october": [3, 9],
        "november": [3, 10],
        "december": [3, 11],
        "januarys": [3, 0],
        "januaries": [3, 0],
        "februarys": [3, 1],
        "februaries": [3, 1],
        "marches": [3, 2],
        "marchs": [3, 2],
        "aprils": [3, 3],
        "mays": [3, 4],
        "junes": [3, 5],
        "julys": [3, 6],
        "julies": [3, 6],
        "augusts": [3, 7],
        "septembers": [3, 8],
        "octobers": [3, 9],
        "novembers": [3, 10],
        "decembers": [3, 11],
        "springs": [4, 0],
        "summers": [4, 1],
        "autumns": [4, 2],
        "winters": [4, 3],
        "spring": [4, 0],
        "summer": [4, 1],
        "autumn": [4, 2],
        "winter": [4, 3],
    }
    vocab_indices = {
        0: [3823, 2781, 2847, 2420, 3134, 2706, 2086, 5109, 4693],
        1: [6440, 2851, 11501, 5027, 3944, 18406, 2305, 7090],
        2: [6928, 9857, 9317, 9432, 5958, 5095, 4465],
        3: [2254, 2337, 2233, 2258, 2089, 2238, 2251, 2257, 2244, 2255, 2281, 2285],
        4: [3500, 2621, 7114, 3467],
    }
    if orig_label not in keywords:
        return None
    label_group = keywords[orig_label][0]
    label_id = keywords[orig_label][1]
    all_soft_labels = [0.0] * vocab_size
    soft_labels = [0.0] * len(vocab_indices[label_group])
    if label_group == 0:
        label_vector = [0.383, 0.242, 0.06, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(label_id, len(soft_labels)):
            soft_labels[i] = label_vector[i - label_id]
        for i in range(label_id - 1, -1, -1):
            soft_labels[i] = label_vector[label_id - i]
    else:
        soft_labels = assign_round_soft_label(len(soft_labels), label_id)

    for idx, sl in enumerate(soft_labels):
        all_soft_labels[vocab_indices[label_group][idx]] = soft_labels[idx]

    return all_soft_labels


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a, lm_labels_a = random_word(example.text_a.lower().split(), example.target_idx_a, tokenizer)
        tokens_b, lm_labels_b = random_word(example.text_b.lower().split(), example.target_idx_b, tokenizer)
        """NOTE BYPASSING MASKING"""
        # tokens_a = example.text_a.lower().split()
        # tokens_a[example.target_idx_a] = "[MASK]"
        # lm_labels_a = [-1] * len(tokens_a)
        # tokens_b = example.text_b.lower().split()
        # tokens_b[example.target_idx_b] = "[MASK]"
        # lm_labels_b = [-1] * len(tokens_b)

        if len(tokens_a) + len(tokens_b) + 3 > max_seq_length:
            # Never delete any token
            continue

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        lm_labels = [-1] + lm_labels_a + [-1] + lm_labels_b + [-1]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        padding = [0] * (max_seq_length - len(input_ids))
        lm_padding = [-1] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        lm_labels += lm_padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_labels) == max_seq_length

        target_idx_a = example.target_idx_a + 1
        target_idx_b = example.target_idx_b + len(tokens_a) + 2

        soft_labels_a = get_soft_labels(example.label_a.lower())
        soft_labels_b = get_soft_labels(example.label_b.lower())

        if soft_labels_a is None or soft_labels_b is None:
            # FIX HERE
            continue

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in example.text_a.split()]))
            logger.info("label: %s" % example.label_a)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                lm_labels=lm_labels,
                target_idx_a=target_idx_a,
                target_idx_b=target_idx_b,
                soft_labels_a=soft_labels_a,
                soft_labels_b=soft_labels_b,
            )
        )
    return features


def simple_accuracy(preds, labels, tolerances=None):
    correct = 0.0
    if tolerances is not None:
        for i, v in enumerate(preds):
            if preds[i] == labels[i]:
            # if labels[i] - tolerances[i] <= preds[i] <= labels[i] + tolerances[i]:
                correct += 1.0
        return correct / float(len(preds))
    else:
        return 0.0


def compute_f1(p, r):
    return 2 * p * r / (p + r)


def compute_metrics(task_name, preds, labels, additional=None):
    if task_name == "tempoalverb":
        return simple_accuracy(preds, labels)
    else:
        raise KeyError(task_name)


def soft_cross_entropy_loss(logits_a, logits_b, soft_target_a, soft_target_b, lm_loss):
    loss_a = -soft_target_a * torch.log(nn.functional.softmax(logits_a, -1))
    loss_a = torch.sum(loss_a, -1)

    loss_b = -soft_target_b * torch.log(nn.functional.softmax(logits_b, -1))
    loss_b = torch.sum(loss_b, -1)

    mean_loss = loss_a.mean() + loss_b.mean()

    if lm_loss is not None:
        return mean_loss + lm_loss, mean_loss.item()
    else:
        return mean_loss.item()


def compute_distance(logits, target):
    vocab_indices = {
        0: [3823, 2781, 2847, 2420, 3134, 2706, 2086, 5109, 4693],
        1: [6440, 2851, 11501, 5027, 3944, 18406, 2305, 7090],
        2: [6928, 9857, 9317, 9432, 5958, 5095, 4465],
        3: [2254, 2337, 2233, 2258, 2089, 2238, 2251, 2257, 2244, 2255, 2281, 2285],
        4: [3500, 2621, 7114, 3467],
    }
    reverse_map = {}
    index_map = {}
    avg_dist = 0.0
    for i in vocab_indices:
        for j, idx in enumerate(vocab_indices[i]):
            reverse_map[idx] = vocab_indices[i]
            index_map[idx] = j
    for i in range(0, logits.shape[0]):
        label_id = int(np.argmax(target[i]))
        group_indices = reverse_map[label_id]
        scores_in_order = []
        for gi in group_indices:
            scores_in_order.append(logits[i][gi])
        predicted_relative_label_id = int(np.argmax(np.array(scores_in_order)))
        avg_dist += abs(index_map[label_id] - predicted_relative_label_id)

    return float(avg_dist) / float(logits.shape[0])


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=48,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "temporalverb": TemporalVerbProcessor,
    }

    output_modes = {
        "temporalverb": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        # n_gpu = 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = TemporalModelArguments.from_pretrained(args.bert_model, cache_dir=cache_dir)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps,
                             e=1e-4)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_lm_labels = torch.tensor([f.lm_labels for f in train_features], dtype=torch.long)
        all_target_idxs_a = torch.tensor([f.target_idx_a for f in train_features], dtype=torch.long)
        all_soft_targets_a = torch.tensor([f.soft_target_a for f in train_features], dtype=torch.float)
        all_target_idxs_b = torch.tensor([f.target_idx_b for f in train_features], dtype=torch.long)
        all_soft_targets_b = torch.tensor([f.soft_target_b for f in train_features], dtype=torch.float)

        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_lm_labels,
            all_target_idxs_a, all_soft_targets_a, all_target_idxs_b, all_soft_targets_b,
        )
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        output_loss_file = os.path.join(args.output_dir, "loss_log.txt")
        f_loss = open(output_loss_file, "a")
        epoch_loss = 0.0
        epoch_label_loss = 0.0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            middle_loss = 0.0
            middle_label_loss = 0.0
            middle_rel_loss = 0.0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                if step == 0:
                    epoch_loss = 0.0
                    epoch_label_loss = 0.0

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_labels, target_ids_a, soft_targets_a, target_ids_b, soft_targets_b = batch

                lm_loss, cls_a, cls_b = model(
                    input_ids, segment_ids, input_mask, lm_labels, target_ids_a, target_ids_b
                )

                loss, non_lm_loss = soft_cross_entropy_loss(
                    cls_a.view(-1, 30522), cls_b.view(-1, 30522),
                    soft_targets_a, soft_targets_b,
                    lm_loss
                )

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

                tr_loss += loss.item()
                middle_loss += loss.item()
                middle_label_loss += non_lm_loss
                epoch_loss += loss.item()
                epoch_label_loss += non_lm_loss

                if step % 100 == 0:
                    f_loss.write(("Total Loss: " + str(middle_loss)) + "\n")
                    f_loss.write(("Label Loss: " + str(middle_label_loss)) + "\n")
                    f_loss.write(("Rel Loss: " + str(middle_rel_loss)) + "\n")
                    f_loss.flush()
                    middle_loss = 0.0
                    middle_label_loss = 0.0
                    middle_rel_loss = 0.0
                nb_tr_examples += input_ids.size(0) * 2
                nb_tr_steps += 1
                if step % 1000 == 0:
                    actual_dir = args.output_dir + "_epoch_" + str(_)
                    if not os.path.exists(actual_dir):
                        os.makedirs(actual_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(actual_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_config_file = os.path.join(actual_dir, CONFIG_NAME)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = TemporalModelArguments(config)
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = TemporalModelArguments.from_pretrained(args.bert_model)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_lm_labels = torch.tensor([f.lm_labels for f in eval_features], dtype=torch.long)
        all_target_idxs_a = torch.tensor([f.target_idx_a for f in eval_features], dtype=torch.long)
        all_target_idxs_b = torch.tensor([f.target_idx_b for f in eval_features], dtype=torch.long)
        all_soft_targets_a = torch.tensor([f.soft_target_a for f in eval_features], dtype=torch.float)
        all_soft_targets_b = torch.tensor([f.soft_target_b for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_lm_labels, all_target_idxs_a, all_soft_targets_a, all_target_idxs_b, all_soft_targets_b,
        )

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()

        output_file = os.path.join(args.output_dir, "bert_logits.txt")
        f_out = open(output_file, "w")
        total_loss = []
        lm_total_loss = []
        prediction_distance = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_labels, target_ids_a, soft_targets_a, target_ids_b, soft_targets_b = batch

            with torch.no_grad():
                lm_loss, cls_a, cls_b = model(
                    input_ids, segment_ids, input_mask, lm_labels, target_ids_a, target_ids_b
                )
                cls_a = cls_a.view(-1, 30522)
                cls_b = cls_b.view(-1, 30522)
                avg_loss = soft_cross_entropy_loss(cls_a, cls_b, soft_targets_a, soft_targets_b, None)
            prediction_distance.append((compute_distance(cls_a.cpu().numpy(), soft_targets_a.cpu().numpy()) + compute_distance(cls_b.cpu().numpy(), soft_targets_b.cpu().numpy())) / 2.0)
            lm_total_loss.append(lm_loss.item())
            total_loss.append(avg_loss)

        f_out.write("Temporal Loss\n")
        f_out.write(str(np.mean(np.array(total_loss))) + "\n")
        f_out.write("LM Loss\n")
        f_out.write(str(np.mean(np.array(lm_total_loss))) + "\n")
        f_out.write("Label Distance\n")
        f_out.write(str(np.mean(np.array(prediction_distance))) + "\n")


if __name__ == "__main__":
    main()
