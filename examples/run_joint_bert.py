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
from pytorch_pretrained_bert.modeling import BertForSingleTokenClassification, BertForSingleTokenClassificationFollowTemporal, TemporalModelJoint, BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertForSingleTokenClassificationWithPooler
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, label_a, label_b, target_idx_a,
                 target_idx_b, rel_label, inst_type):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label_a = label_a
        self.label_b = label_b
        self.target_idx_a = target_idx_a
        self.target_idx_b = target_idx_b
        self.rel_label = rel_label
        self.inst_type = inst_type


class InputFeatures(object):

    def __init__(self, input_ids_a, input_ids_b, input_mask_a, input_mask_b, segment_ids_a, segment_ids_b,
                 label_id_a, label_id_b, target_idx_a, target_idx_b, adjustment_a, adjustment_b,
                 soft_target_a, soft_target_b, lm_labels_a, lm_labels_b, rel_soft_target):
        self.input_ids_a = input_ids_a
        self.input_ids_b = input_ids_b
        self.input_mask_a = input_mask_a
        self.input_mask_b = input_mask_b
        self.segment_ids_a = segment_ids_a
        self.segment_ids_b = segment_ids_b
        self.label_id_a = label_id_a
        self.label_id_b = label_id_b
        self.target_idx_a = target_idx_a
        self.target_idx_b = target_idx_b
        self.adjustment_a = adjustment_a
        self.adjustment_b = adjustment_b
        self.soft_target_a = soft_target_a
        self.soft_target_b = soft_target_b
        self.lm_labels_a = lm_labels_a
        self.lm_labels_b = lm_labels_b
        self.rel_soft_target = rel_soft_target


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
        count_map = {}
        for e in examples:
            if e.label_a not in count_map:
                count_map[e.label_a] = 0
            if e.label_b not in count_map:
                count_map[e.label_b] = 0
            count_map[e.label_a] += 1
            count_map[e.label_b] += 1
        print(count_map)
        return examples

    def get_dev_examples(self, data_dir):
        f = open(os.path.join(data_dir, "test.formatted.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

    def get_labels(self):
        return [
            "seconds",
            "minutes",
            "hours",
            "days",
            "weeks",
            "months",
            "years",
            "decades",
            "centuries",
        ]

    def get_typical_labels(self):
        return [
            # time of the dar
            "dawn", "morning", "noon", "afternoon", "evening", "dusk", "night", "midnight",
            # time of the week
            "monday", "tuesday", "wednesday", 'thursday', "friday", "saturday", "sunday",
            # time of the year
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
            # time of the year
            "spring", "summer", "autumn", "winter",
        ]

    def plural_units(self, u):
        m = {
            "second": "seconds",
            "minute": "minutes",
            "hour": "hours",
            "day": "days",
            "week": "weeks",
            "month": "months",
            "year": "years",
            "decade": "decades",
            "century": "centuries",
        }
        if u in m:
            return m[u]
        return u

    def normalize_timex(self, v_input, u):
        if u in ["instantaneous", "forever"]:
            return u, str(1)

        convert_map = {
            "seconds": 1.0,
            "minutes": 60.0,
            "hours": 60.0 * 60.0,
            "days": 24.0 * 60.0 * 60.0,
            "weeks": 7.0 * 24.0 * 60.0 * 60.0,
            "months": 30.0 * 24.0 * 60.0 * 60.0,
            "years": 365.0 * 24.0 * 60.0 * 60.0,
            "decades": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
        }
        seconds = convert_map[self.plural_units(u)] * float(v_input)
        prev_unit = "seconds"
        for i, v in enumerate(convert_map):
            if seconds / convert_map[v] < 0.5:
                break
            prev_unit = v
        if prev_unit == "seconds" and seconds > 60.0:
            prev_unit = "centuries"
        new_val = int(seconds / convert_map[prev_unit])

        return prev_unit, str(new_val)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        for (i, line) in enumerate(lines):
            groups = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            if len(groups) == 7:
                text_a = groups[0]
                target_idx_a = int(groups[1])
                label_a = groups[2]
                text_b = groups[3]
                target_idx_b = int(groups[4])
                label_b = groups[5]
                rel_label = groups[6]
                if len(text_a.split()) > 128 or len(text_b.split()) > 128:
                    continue

                if label_a != "NONE" and groups[-1] in ["DUR", "FREQ"]:
                    label_a_num = float(label_a.split()[0])
                    label_a, _ = self.normalize_timex(label_a_num, label_a.split()[1].lower())

                if label_b != "NONE" and groups[-1] in ["DUR", "FREQ"]:
                    label_b_num = float(label_b.split()[0])
                    label_b, _ = self.normalize_timex(label_b_num, label_b.split()[1].lower())

                examples.append(
                    InputExample(
                        guid=guid, text_a=text_a, text_b=text_b, label_a=label_a, label_b=label_b,
                        target_idx_a=target_idx_a, target_idx_b=target_idx_b, rel_label=rel_label, inst_type=groups[-1]
                    )
                )
            else:
                continue

        return examples


def randomize_tokens(tokens, tokenizer):
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
                output_label.append(tokenizer.vocab["[UNK]"])
                # logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_single_pair(text, target_idx, label, tokenizer, max_seq_length, label_map, typical_label_map, inst_type):
    tokens = text.lower().split()
    """
    MASKING TOKENS FOR LM LOSSES
    """
    tokens, masking_labels = randomize_tokens(tokens, tokenizer)
    masking_labels = [-1] + masking_labels + [-1]
    # masking_labels = [-1] * (len(tokens) + 2)
    """"""
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))
    masking_labels += [-1] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if inst_type in ["FREQ", "DUR"]:
        if label == "NONE":
            label_id = -1
            soft_labels = None
        else:
            label_id = label_map[label]
            label_size = len(label_map)
            soft_labels = [0.0] * label_size
            label_vector = [0.383, 0.242, 0.06, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for i in range(label_id, len(soft_labels)):
                soft_labels[i] = label_vector[i - label_id]
            for i in range(label_id - 1, -1, -1):
                soft_labels[i] = label_vector[label_id - i]
    elif inst_type in ["TYPICAL"]:
        label_id = typical_label_map[label]
        label_vector = [0.383, 0.242, 0.06, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        soft_labels = [0.0] * len(typical_label_map)
        if 0 <= label_id <= 7:
            for i in range(label_id, len(soft_labels)):
                if i > 7:
                    break
                soft_labels[i] = label_vector[i - label_id]
            for i in range(label_id - 1, -1, -1):
                if i < 0:
                    break
                soft_labels[i] = label_vector[label_id - i]
        if 8 <= label_id <= 14:
            for i in range(label_id, len(soft_labels)):
                if i > 14:
                    break
                soft_labels[i] = label_vector[i - label_id]
            for i in range(label_id - 1, -1, -1):
                if i < 8:
                    break
                soft_labels[i] = label_vector[label_id - i]
        if 15 <= label_id <= 26:
            for i in range(label_id, len(soft_labels)):
                if i > 26:
                    break
                soft_labels[i] = label_vector[i - label_id]
            for i in range(label_id - 1, -1, -1):
                if i < 15:
                    break
                soft_labels[i] = label_vector[label_id - i]
        if 27 <= label_id <= 30:
            for i in range(label_id, len(soft_labels)):
                if i > 30:
                    break
                soft_labels[i] = label_vector[i - label_id]
            for i in range(label_id - 1, -1, -1):
                if i < 27:
                    break
                soft_labels[i] = label_vector[label_id - i]
    else:
        label_id = -1
        soft_labels = None

    if inst_type == "FREQ":
        soft_labels = soft_labels + [0.0] * len(label_map) + [0.0] * len(typical_label_map)
    if inst_type == "DUR":
        soft_labels = [0.0] * len(label_map) + soft_labels + [0.0] * len(typical_label_map)
    if inst_type == "TYPICAL":
        soft_labels = [0.0] * len(label_map) + [0.0] * len(label_map) + soft_labels

    weight_map_duration = {
        0: 11.66,
        1: 2.04,
        2: 2.08,
        3: 1.71,
        4: 1.33,
        5: 0.78,
        6: 0.44,
        7: 0.35,
        8: 2.74
    }

    adjustment = 1.0
    if inst_type == "DUR":
        if label_id > -1:
            adjustment = weight_map_duration[label_id]

    return input_ids, input_mask, segment_ids, label_id, target_idx + 1, adjustment, soft_labels, masking_labels


def convert_examples_to_features(examples, label_list, typical_label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    typical_label_map = {label: i for i, label in enumerate(typical_label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        input_ids_a, input_mask_a, segment_ids_a, label_id_a, target_idx_a, adjustment_a, soft_labels_a, lm_labels_a = convert_single_pair(
            example.text_a, example.target_idx_a, example.label_a, tokenizer, max_seq_length, label_map, typical_label_map, example.inst_type
        )

        if example.text_b is not None:
            input_ids_b, input_mask_b, segment_ids_b, label_id_b, target_idx_b, adjustment_b, soft_labels_b, lm_labels_b = convert_single_pair(
                example.text_b, example.target_idx_b, example.label_b, tokenizer, max_seq_length, label_map, typical_label_map, example.inst_type
            )
        else:
            print("ERROR!")
            input_ids_b, input_mask_b, segment_ids_b, label_id_b, target_idx_b, adjustment_b, soft_labels_b, lm_labels_b = [None] * len(label_list)

        rel_soft_labels = [0.0] * 2
        if example.rel_label == "LESS":
            rel_soft_labels[0] = 1.0
        elif example.rel_label == "MORE":
            rel_soft_labels[1] = 1.0

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in example.text_a.split()]))
            logger.info("label: %s" % example.label_a)
            logger.info("soft label: %s" % " ".join(str(x) for x in soft_labels_a))
            logger.info("input ids: %s" % " ".join(str(x) for x in input_ids_a))
            logger.info("LM label: %s" % " ".join(str(x) for x in lm_labels_a))
            logger.info("Classification label: %s" % label_id_a)

        features.append(
            InputFeatures(
                input_ids_a=input_ids_a,
                input_ids_b=input_ids_b,
                input_mask_a=input_mask_a,
                input_mask_b=input_mask_b,
                segment_ids_a=segment_ids_a,
                segment_ids_b=segment_ids_b,
                label_id_a=label_id_a,
                label_id_b=label_id_b,
                target_idx_a=target_idx_a,
                target_idx_b=target_idx_b,
                adjustment_a=adjustment_a,
                adjustment_b=adjustment_b,
                soft_target_a=soft_labels_a,
                soft_target_b=soft_labels_b,
                lm_labels_a=lm_labels_a,
                lm_labels_b=lm_labels_b,
                rel_soft_target=rel_soft_labels
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


def soft_cross_entropy_loss(logits_a, logits_b, soft_target_a, soft_target_b, adjustment_a, adjustment_b, lm_loss):
    loss_a = -soft_target_a * torch.log(logits_a)
    loss_a = torch.sum(loss_a, -1) * adjustment_a

    loss_b = -soft_target_b * torch.log(logits_b)
    loss_b = torch.sum(loss_b, -1) * adjustment_b

    mean_loss = loss_a.mean() + loss_b.mean()

    if lm_loss is not None:
        return mean_loss + lm_loss, mean_loss.item()
    else:
        return mean_loss, mean_loss.item()


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

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    typical_label_list = processor.get_typical_labels()
    num_labels = len(label_list)
    num_typical_labels = len(typical_label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # tokenizer = BasicTokenizer()

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
    model = TemporalModelJoint.from_pretrained(args.bert_model,
              cache_dir=cache_dir,
              num_labels=num_labels,
              num_typical_labels=num_typical_labels)
    # for p in model.bert.parameters():
    #     p.requires_grad = False
    # for p in model.dur_classifier.parameters():
    #     p.requires_grad = False
    # for p in model.freq_classifier.parameters():
    #     p.requires_grad = False
    # for p in model.typical_classifier.parameters():
    #     p.requires_grad = False

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
            train_examples, label_list, typical_label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids_a = torch.tensor([f.input_ids_a for f in train_features], dtype=torch.long)
        all_input_mask_a = torch.tensor([f.input_mask_a for f in train_features], dtype=torch.long)
        all_segment_ids_a = torch.tensor([f.segment_ids_a for f in train_features], dtype=torch.long)
        all_target_idxs_a = torch.tensor([f.target_idx_a for f in train_features], dtype=torch.long)
        all_soft_targets_a = torch.tensor([f.soft_target_a for f in train_features], dtype=torch.float)
        all_adjustments_a = torch.tensor([f.adjustment_a for f in train_features], dtype=torch.float)
        all_label_ids_a = torch.tensor([f.label_id_a for f in train_features], dtype=torch.long)
        all_lm_labels_a = torch.tensor([f.lm_labels_a for f in train_features], dtype=torch.long)

        all_input_ids_b = torch.tensor([f.input_ids_b for f in train_features], dtype=torch.long)
        all_input_mask_b = torch.tensor([f.input_mask_b for f in train_features], dtype=torch.long)
        all_segment_ids_b = torch.tensor([f.segment_ids_b for f in train_features], dtype=torch.long)
        all_target_idxs_b = torch.tensor([f.target_idx_b for f in train_features], dtype=torch.long)
        all_soft_targets_b = torch.tensor([f.soft_target_b for f in train_features], dtype=torch.float)
        all_adjustments_b = torch.tensor([f.adjustment_b for f in train_features], dtype=torch.float)
        all_label_ids_b = torch.tensor([f.label_id_b for f in train_features], dtype=torch.long)
        all_lm_labels_b = torch.tensor([f.lm_labels_b for f in train_features], dtype=torch.long)

        all_rel_labels = torch.tensor([f.rel_soft_target for f in train_features], dtype=torch.float)

        train_data = TensorDataset(
            all_input_ids_a, all_input_mask_a, all_segment_ids_a, all_target_idxs_a, all_soft_targets_a, all_adjustments_a,
            all_input_ids_b, all_input_mask_b, all_segment_ids_b, all_target_idxs_b, all_soft_targets_b, all_adjustments_b,
            all_rel_labels, all_label_ids_a, all_label_ids_b, all_lm_labels_a, all_lm_labels_b
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
        epoch_rel_loss = 0.0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            middle_loss = 0.0
            middle_label_loss = 0.0
            middle_rel_loss = 0.0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                if step == 0:
                    # f_loss.write(("Total Loss: " + str(epoch_loss)) + "\n")
                    # f_loss.write(("Label Loss: " + str(epoch_label_loss)) + "\n")
                    # f_loss.write(("Rel Loss: " + str(epoch_rel_loss)) + "\n")
                    # f_loss.flush()
                    epoch_loss = 0.0
                    epoch_label_loss = 0.0
                    epoch_rel_loss = 0.0

                batch = tuple(t.to(device) for t in batch)
                input_ids_a, input_mask_a, segment_ids_a, target_ids_a, soft_targets_a, adjustments_a, \
                input_ids_b, input_mask_b, segment_ids_b, target_ids_b, soft_targets_b, adjustments_b, rel_labels, \
                label_ids_a, label_ids_b, lm_labels_a, lm_labels_b = batch

                # define a new function to compute loss values for both output_modes
                logits_a, logits_b, lm_loss = model(
                    input_ids_a, segment_ids_a, input_mask_a, target_ids_a,
                    input_ids_b, segment_ids_b, input_mask_b, target_ids_b,
                    lm_labels_a, lm_labels_b
                )

                loss, non_lm_loss = soft_cross_entropy_loss(
                    logits_a.view(-1, num_labels * 2 + num_typical_labels), logits_b.view(-1, num_labels * 2 + num_typical_labels),
                    soft_targets_a, soft_targets_b,
                    adjustments_a, adjustments_b,
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
                nb_tr_examples += input_ids_a.size(0) * 2
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
        model = TemporalModelJoint(config, num_labels=num_labels, num_typical_labels=num_typical_labels)
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = TemporalModelJoint.from_pretrained(args.bert_model, num_labels=num_labels, num_typical_labels=num_typical_labels)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, typical_label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids_a = torch.tensor([f.input_ids_a for f in eval_features], dtype=torch.long)
        all_input_mask_a = torch.tensor([f.input_mask_a for f in eval_features], dtype=torch.long)
        all_segment_ids_a = torch.tensor([f.segment_ids_a for f in eval_features], dtype=torch.long)
        all_target_idxs_a = torch.tensor([f.target_idx_a for f in eval_features], dtype=torch.long)
        all_soft_targets_a = torch.tensor([f.soft_target_a for f in eval_features], dtype=torch.float)
        all_adjustments_a = torch.tensor([f.adjustment_a for f in eval_features], dtype=torch.float)

        all_input_ids_b = torch.tensor([f.input_ids_b for f in eval_features], dtype=torch.long)
        all_input_mask_b = torch.tensor([f.input_mask_b for f in eval_features], dtype=torch.long)
        all_segment_ids_b = torch.tensor([f.segment_ids_b for f in eval_features], dtype=torch.long)
        all_target_idxs_b = torch.tensor([f.target_idx_b for f in eval_features], dtype=torch.long)
        all_soft_targets_b = torch.tensor([f.soft_target_b for f in eval_features], dtype=torch.float)
        all_adjustments_b = torch.tensor([f.adjustment_b for f in eval_features], dtype=torch.float)

        all_rel_labels = torch.tensor([f.rel_soft_target for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(
            all_input_ids_a, all_input_mask_a, all_segment_ids_a, all_target_idxs_a, all_soft_targets_a, all_adjustments_a,
            all_input_ids_b, all_input_mask_b, all_segment_ids_b, all_target_idxs_b, all_soft_targets_b, all_adjustments_b,
            all_rel_labels
        )

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        output_file = os.path.join(args.output_dir, "bert_logits.txt")
        f_out = open(output_file, "w")
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids_a, input_mask_a, segment_ids_a, target_ids_a, soft_targets_a, adjustments_a, \
            input_ids_b, input_mask_b, segment_ids_b, target_ids_b, soft_targets_b, adjustments_b, rel_labels = batch

            with torch.no_grad():
                logits_a, logits_b, _ = model(
                    input_ids_a, segment_ids_a, input_mask_a, target_ids_a,
                    input_ids_b, segment_ids_b, input_mask_b, target_ids_b, None, None
                )
                logits = torch.cat((logits_a, logits_b), -1)

            w_ready_logits = logits.detach().cpu().numpy()
            for l in w_ready_logits:
                # ll = l[0]
                ll = l
                s = ""
                for lll in ll:
                    s += str(lll) + "\t"
                f_out.write(s + "\n")
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        loss = tr_loss/nb_tr_steps if args.do_train else None


if __name__ == "__main__":
    main()
