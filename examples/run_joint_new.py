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
from pytorch_pretrained_bert.modeling import TemporalModelJointNew, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label, target_idx, dimension):
        self.guid = guid
        self.text = text
        self.target_idx = target_idx
        self.label = label
        self.dimension = dimension


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, lm_labels, target_idx, soft_labels, adjustment):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_labels = lm_labels
        self.target_idx = target_idx
        self.soft_labels = soft_labels
        self.adjustment = adjustment


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
        seconds = convert_map[u] * float(v_input)
        prev_unit = "seconds"
        for i, v in enumerate(convert_map):
            if seconds / convert_map[v] < 0.5:
                break
            prev_unit = v
        if prev_unit == "seconds" and seconds > 60.0:
            prev_unit = "centuries"
        new_val = int(seconds / convert_map[prev_unit])

        return prev_unit, str(new_val)

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
        for i in range(0, len(lines)):
            guid = "%s-%s" % (set_type, i)

            groups = lines[i].split("\t")
            text = groups[0]
            target_idx = int(groups[1])
            label = groups[2]
            if " " in label:
                label, _ = self.normalize_timex(float(label.split()[0]), label.split()[1])
            dimension = groups[3]

            examples.append(
                InputExample(
                    guid=guid, text=text, target_idx=target_idx, label=label, dimension=dimension,
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


def random_word(tokens, tokenizer):
    output_label = []

    for i, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP"]:
            output_label.append(-1)
            continue
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

    assert len(tokens) == len(output_label)
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


def get_soft_labels(orig_label):
    keywords = {
        "seconds": [0, 0],
        "minutes": [0, 1],
        "hours": [0, 2],
        "days": [0, 3],
        "weeks": [0, 4],
        "months": [0, 5],
        "years": [0, 6],
        "decades": [0, 7],
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
        "falls": [4, 2],
        "winters": [4, 3],
        "spring": [4, 0],
        "summer": [4, 1],
        "autumn": [4, 2],
        "fall": [4, 2],
        "winter": [4, 3],
        "yes": [5, 0],
        "no": [5, 1]
    }
    group_sizes = {
        0: 9, 1: 8, 2: 7, 3: 12, 4: 4, 5: 2,
    }
    if orig_label not in keywords:
        print(orig_label)
        print("Error: should never happen.")
        return None, -1, -1
    label_group = keywords[orig_label][0]
    label_id = keywords[orig_label][1]
    if label_group == 0:
        label_vector_map = [
            [0.7412392700826488, 0.24912843928090708, 0.009458406390501939, 0.00016606286478555588, 7.30646267660146e-06, 5.120999881870982e-07, 2.807187666675083e-09, 1.1281434169352893e-11, 2.2746998844715235e-14],
            [0.1966797321503831, 0.5851870686462161, 0.1966797321503831, 0.018764436502991182, 0.0023274596334087274, 0.0003541235975250493, 7.345990219746334e-06, 1.0063714849200163e-07, 6.917246071508814e-10],
            [0.006037846064281208, 0.15903304473396812, 0.47317575760473235, 0.2453125506936087, 0.08577848758081653, 0.028331933195186416, 0.002224080407310708, 0.00010386603946867244, 2.4336806272483765e-06],
            [7.670003601399247e-05, 0.010977975780067789, 0.17749198395637333, 0.3423587734906385, 0.26762063340149095, 0.1613272650199883, 0.03558053856215351, 0.004304815288057253, 0.00026131446521643803],
            [2.9988887071575075e-06, 0.0012100380728822476, 0.055152791268338504, 0.23782074256001023, 0.3042366976664677, 0.26508603808222186, 0.11004905411557955, 0.02384861332046269, 0.0025930260253300397],
            [2.0746396147545137e-07, 0.00018172156454292432, 0.017980428741889258, 0.14150527501625504, 0.2616505043598367, 0.3002937686386626, 0.20006984338208827, 0.06704560384872892, 0.011272646984034582],
            [1.2229783553502831e-09, 4.053791134884565e-06, 0.0015178670785093325, 0.03356114730820127, 0.11681011704848474, 0.21514985378380605, 0.32292803014499954, 0.22873846705624448, 0.08129046256564142],
            [5.572730351111348e-12, 6.296884648723474e-08, 8.037356297882278e-05, 0.004603998728773229, 0.02870209972984776, 0.08174969138345407, 0.25935558127072855, 0.3661526110790698, 0.25935558127072855],
            [1.5291249697362676e-14, 5.890006839530834e-10, 2.5628217033951704e-06, 0.0003803288616050351, 0.004246905247620703, 0.018704964177227536, 0.12543280829498968, 0.35294802591125274, 0.4982844040965849],
        ]
        soft_labels = label_vector_map[label_id]
    elif label_group == 5:
        soft_labels = [0.0] * 2
        soft_labels[label_id] = 1.0
    elif label_group == 1:
        soft_labels = assign_round_soft_label(group_sizes[label_group], label_id) + [0.0] * 23
    elif label_group == 2:
        soft_labels = [0.0] * 8 + assign_round_soft_label(group_sizes[label_group], label_id) + [0.0] * 16
    elif label_group == 3:
        soft_labels = [0.0] * 15 + assign_round_soft_label(group_sizes[label_group], label_id) + [0.0] * 4
    elif label_group == 4:
        soft_labels = [0.0] * 27 + assign_round_soft_label(group_sizes[label_group], label_id)
    else:
        print("Error: should never happen")
        soft_labels = []

    real_label_id = np.argmax(np.array(soft_labels))

    return soft_labels, label_group, real_label_id


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens, lm_labels = random_word(example.text.split(), tokenizer)
        first_sent_length = 0
        for i, t in enumerate(tokens):
            if t == "[SEP]":
                first_sent_length = i + 1

        if len(tokens) > max_seq_length:
            # Never delete any token
            continue

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * first_sent_length + [1] * (len(tokens) - first_sent_length)
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

        target_idx = example.target_idx
        soft_label_length = 51
        dimension_index_map = {
            "DUR": (0, 9),
            "FREQ": (9, 18),
            "TYP": (18, 49),
            "ORD": (49, 51),
        }
        """COMBINED WEIGHT"""
        adjustment_map = {
            "DUR": [16.540970625798213, 3.8448873689335374, 1.8235395159973191, 1.7189426938584824, 1.3133984780633436, 0.6898745167472576, 0.3434554135229082, 0.4791514984698619, 2.9429924423175677],
            "FREQ": [13.185669572993516, 6.121698064248274, 5.332203236597052, 0.33904740552954327, 1.198149333170816, 1.2243396128496549, 0.26337423751216854, 6.271079213184476, 62.116377040547654],
            "TYP": [15.011903251186517, 2.6310161379285804, 16.128595969787035, 4.748663273334511, 8.70443842712503,
             126.35847041230507, 1.8558631487460426, 10.990857737475844, 0.3117006033401086, 0.29835544451624,
             0.29643179679113296, 0.307263298263331, 0.3252442559705793, 0.4985271348537628, 0.4670663883933824,
             1.4400097382836274, 1.9593478160051412, 1.1165805847529298, 1.2998248540267092, 1.3422531855264204,
             1.2093635692627283, 1.2316912421572568, 1.921463145242504, 1.61436069314308, 1.7979767275026446,
             1.7378749460113, 1.6234262768874386, 4.658234627016129, 2.3844200240262516, 4.628852264012146,
             7.839001245792138],
            "ORD": [1.0, 1.0]
        }
        soft_labels = [0.0] * soft_label_length
        per_group_soft_label, _, relative_label_id = get_soft_labels(example.label)
        for i in range(dimension_index_map[example.dimension][0], dimension_index_map[example.dimension][1]):
            soft_labels[i] = per_group_soft_label[i - dimension_index_map[example.dimension][0]]

        adjustment = adjustment_map[example.dimension][relative_label_id]

        assert len(soft_labels) == soft_label_length

        if ex_index < 100:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("LM label: %s" % " ".join(str(x) for x in lm_labels))
            logger.info("soft_labels: %s" % " ".join(str(x) for x in soft_labels))
            logger.info("adjustments: %s" % str(adjustment))
            logger.info("target index: %s" % str(target_idx))
            logger.info("label: %s" % example.label)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                lm_labels=lm_labels,
                target_idx=target_idx,
                soft_labels=soft_labels,
                adjustment=adjustment,
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


def soft_cross_entropy_loss(logits, soft_labels, lm_loss, adjustments=None):

    logits_softmaxed = torch.cat((
        nn.functional.log_softmax(logits.narrow(1, 0, 9), -1),
        nn.functional.log_softmax(logits.narrow(1, 9, 9), -1),
        nn.functional.log_softmax(logits.narrow(1, 18, 31), -1),
        nn.functional.log_softmax(logits.narrow(1, 49, 2), -1),
    ), -1)

    loss = -soft_labels * logits_softmaxed
    loss = torch.sum(loss, -1)
    if adjustments is not None:
        loss = loss * adjustments
    loss = loss.mean()

    if lm_loss is not None:
        return loss + lm_loss, loss.item()
    else:
        return loss.item()


def combine_map(map_big, map_small):
    for key in map_small:
        if key not in map_big:
            map_big[key] = 0.0
        map_big[key] += map_small[key]
    return map_big


def compute_distance(logits, target):
    logits_range_map = {
        "duration": [0, 9],
        "frequency": [9, 18],
        "time_of_day": [18, 26],
        "time_of_week": [26, 33],
        "month": [33, 45],
        "season": [45, 49],
        "ordering": [49, 51]
    }
    log_distance = [0.0, 1.7781512503836434, 3.556302500767287, 4.936513742478892, 5.781611782493149, 6.413634997198555, 7.498806606935368, 8.498806606935366, 9.498806606935368]
    reverse_map = {}
    for i in range(0, 51):
        for key in logits_range_map:
            if logits_range_map[key][1] > i >= logits_range_map[key][0]:
                reverse_map[i] = key
    result_map = {}
    count_map = {}
    per_label_map = {}
    per_label_count_map = {}
    for key in logits_range_map:
        result_map[key] = 0.0
        count_map[key] = 0.0
    result_map['duration-log'] = 0.0
    result_map['frequency-log'] = 0.0
    count_map['duration-log'] = 0.0
    count_map['frequency-log'] = 0.0

    for i in range(0, logits.shape[0]):
        cur_logits = logits[i]
        cur_target = target[i]
        target_label_id = np.argmax(cur_target)
        label_group = reverse_map[target_label_id]
        target_label_id -= logits_range_map[label_group][0]
        logits_vec = cur_logits[logits_range_map[label_group][0]:logits_range_map[label_group][1]]
        predicted_label_id = np.argmax(logits_vec)
        result_map[label_group] += float(abs(target_label_id - predicted_label_id))
        if label_group in ["duration", "frequency"]:
            result_map[label_group + "-log"] += float(abs(log_distance[int(target_label_id)] - log_distance[int(predicted_label_id)]))
            count_map[label_group + "-log"] += 1.0
        count_map[label_group] += 1.0

        label_key = label_group + "-" + str(target_label_id)
        if label_key not in per_label_map:
            per_label_map[label_key] = 0.0
            per_label_count_map[label_key] = 0.0
        if label_group in ["duration", "frequency"]:
            per_label_map[label_key] += float(abs(log_distance[int(target_label_id)] - log_distance[int(predicted_label_id)]))
        else:
            per_label_map[label_key] += float(abs(target_label_id - predicted_label_id))

        per_label_count_map[label_key] += 1.0

    return result_map, count_map, per_label_map, per_label_count_map


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
    model = TemporalModelJointNew.from_pretrained(args.bert_model, cache_dir=cache_dir)

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
        all_target_idxs = torch.tensor([f.target_idx for f in train_features], dtype=torch.long)
        all_soft_labels = torch.tensor([f.soft_labels for f in train_features], dtype=torch.float)
        all_adjustments = torch.tensor([f.adjustment for f in train_features], dtype=torch.float)

        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_lm_labels, all_target_idxs, all_soft_labels, all_adjustments
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
                input_ids, input_mask, segment_ids, lm_labels, target_ids, soft_labels, adjustments = batch

                logits, lm_loss = model(
                    input_ids, segment_ids, input_mask, target_ids, lm_labels
                )

                loss, non_lm_loss = soft_cross_entropy_loss(
                    logits.view(-1, 51), soft_labels.view(-1, 51), lm_loss, adjustments
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
        model = TemporalModelJointNew(config)
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = TemporalModelJointNew.from_pretrained(args.bert_model)
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
        all_target_idxs = torch.tensor([f.target_idx for f in eval_features], dtype=torch.long)
        all_soft_labels = torch.tensor([f.soft_labels for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_lm_labels, all_target_idxs, all_soft_labels
        )

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()

        output_file = os.path.join(args.output_dir, "bert_logits.txt")
        f_out = open(output_file, "w")
        total_loss = []
        lm_total_loss = []
        prediction_distance_map = {}
        prediction_count_map = {}
        per_label_distance_map = {}
        per_label_count_map = {}
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_labels, target_ids, soft_labels = batch

            with torch.no_grad():
                logits, lm_loss = model(
                    input_ids, segment_ids, input_mask, target_ids, lm_labels
                )
                loss = soft_cross_entropy_loss(
                    logits.view(-1, 51), soft_labels.view(-1, 51), None
                )
            prediction_distance_map = combine_map(prediction_distance_map, compute_distance(logits.view(-1, 51).cpu().numpy(), soft_labels.cpu().numpy())[0])
            prediction_count_map = combine_map(prediction_count_map, compute_distance(logits.view(-1, 51).cpu().numpy(), soft_labels.cpu().numpy())[1])
            per_label_distance_map = combine_map(per_label_distance_map, compute_distance(logits.view(-1, 51).cpu().numpy(), soft_labels.cpu().numpy())[2])
            per_label_count_map = combine_map(per_label_count_map, compute_distance(logits.view(-1, 51).cpu().numpy(), soft_labels.cpu().numpy())[3])
            lm_total_loss.append(lm_loss.item())
            total_loss.append(loss)

        f_out.write("Temporal Loss\n")
        f_out.write(str(np.mean(np.array(total_loss))) + "\n")
        f_out.write("LM Loss\n")
        f_out.write(str(np.mean(np.array(lm_total_loss))) + "\n")
        f_out.write("Label Distance\n")
        for key in prediction_distance_map:
            f_out.write(key + "\n")
            f_out.write(str(prediction_distance_map[key] / prediction_count_map[key]) + "\n")

        merge_map = {}
        merge_count_map = {}
        for key in per_label_count_map:
            group = key.split("-")[0]
            if group not in merge_map:
                merge_map[group] = 0.0
                merge_count_map[group] = 0.0
            merge_map[group] += float(per_label_distance_map[key] / per_label_count_map[key])
            merge_count_map[group] += 1.0
        f_out.write("Macro Log Distance\n")
        for key in merge_count_map:
            f_out.write(key + "\n")
            f_out.write(str(merge_map[key] / merge_count_map[key]) + "\n")


if __name__ == "__main__":
    main()
