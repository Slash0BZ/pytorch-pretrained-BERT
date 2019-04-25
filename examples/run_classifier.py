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
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from math import floor

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForTemporalClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, target_idx=0, tolerance=1, subj_mask=None, obj_mask=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.target_idx = target_idx
        self.tolerance = tolerance
        self.subj_mask = subj_mask
        self.obj_mask = obj_mask


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, target_idx=0, tolerance=3, subj_mask=None, obj_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.target_idx = target_idx
        self.tolerance = tolerance
        self.subj_mask = subj_mask
        self.obj_mask = obj_mask


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


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class AnimalProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        f = open(os.path.join(data_dir, "train.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        f = open(os.path.join(data_dir, "eval.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

    def get_labels(self):
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            groups = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = groups[0]
            label = groups[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class TemporalNominalProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        f = open(os.path.join(data_dir, "train.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        f = open(os.path.join(data_dir, "test.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

    def get_labels(self):
        return [str(x) for x in range(40)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        unit_map = {
            "second": 0.0,
            "seconds": 0.0,
            "minute": 1.0,
            "minutes": 1.0,
            "hour": 2.0,
            "hours": 2.0,
            "day": 3.0,
            "days": 3.0,
            "week": 4.0,
            "weeks": 4.0,
            "month": 5.0,
            "months": 5.0,
            "year": 6.0,
            "years": 6.0,
            "century": 7.0,
            "centuries": 7.0,
        }
        advance_map = {
            "second": 60.0,
            "seconds": 60.0,
            "minute": 60.0,
            "minutes": 60.0,
            "hour": 24.0,
            "hours": 24.0,
            "day": 7.0,
            "days": 7.0,
            "week": 4.0,
            "weeks": 4.0,
            "month": 12.0,
            "months": 12.0,
            "year": 100.0,
            "years": 100.0,
            "century": 2.0,
            "centuries": 2.0,
        }
        for (i, line) in enumerate(lines):
            groups = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = groups[0]
            if len(text_a.split()) > 120:
                continue
            target_idx = int(groups[1])
            label_raw = groups[2]
            label_num = float(label_raw.split(" ")[0])
            if label_num < 1.0:
                continue
            label_unit = label_raw.split(" ")[1].lower()
            label_idx = int(floor(label_num * 5.0 / advance_map[label_unit]))
            if label_idx > 4:
                label_idx = 4
            if label_idx < 0:
                label_idx = 0
            label_idx = unit_map[label_unit] * 5.0 + label_idx
            assert 40 > label_idx >= 0
            subj_mask = [0] * len(text_a.split())
            obj_mask = [0] * len(text_a.split())

            if int(groups[3]) > -1:
                for j in range(int(groups[3]), int(groups[4])):
                    subj_mask[j] = 1
            if int(groups[5]) > -1:
                for j in range(int(groups[5]), int(groups[6])):
                    obj_mask[j] = 1

            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, label=str(int(label_idx)), target_idx=target_idx, subj_mask=subj_mask, obj_mask=obj_mask
                ))
        return examples


class TemporalVerbProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        f = open(os.path.join(data_dir, "train.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        f = open(os.path.join(data_dir, "test.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

    def get_labels(self):
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        value_map = {
            "second": 1.0,
            "seconds": 1.0,
            "minute": 60.0,
            "minutes": 60.0,
            "hour": 60.0 * 60.0,
            "hours": 60.0 * 60.0,
            "day": 24.0 * 60.0 * 60.0,
            "days": 24.0 * 60.0 * 60.0,
            "week": 7.0 * 24.0 * 60.0 * 60.0,
            "weeks": 7.0 * 24.0 * 60.0 * 60.0,
            "month": 28.0 * 24.0 * 60.0 * 60.0,
            "months": 28.0 * 24.0 * 60.0 * 60.0,
            "year": 336.0 * 24.0 * 60.0 * 60.0,
            "years": 336.0 * 24.0 * 60.0 * 60.0,
            "century": 100.0 * 336.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 336.0 * 24.0 * 60.0 * 60.0,
        }
        for (i, line) in enumerate(lines):
            groups = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = groups[0]
            if len(text_a.split()) > 120:
                continue
            target_idx = int(groups[1])
            label_raw = groups[2]
            label_num = float(label_raw.split(" ")[0])
            if label_num < 1.0:
                continue
            label_unit = label_raw.split(" ")[1].lower()
            # label_idx = label_num * value_map[label_unit] / 2592000.0
            label_idx = math.log(label_num * value_map[label_unit]) / 22.0
            if label_idx > 1.0:
                continue
            subj_mask = [0] * len(text_a.split())
            obj_mask = [0] * len(text_a.split())

            if int(groups[3]) > -1:
                for j in range(int(groups[3]), int(groups[4])):
                    subj_mask[j] = 1
            if int(groups[5]) > -1:
                for j in range(int(groups[5]), int(groups[6])):
                    obj_mask[j] = 1

            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, label=label_idx, target_idx=target_idx, subj_mask=subj_mask, obj_mask=obj_mask
                ))
        return examples


class TimebankProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        f = open(os.path.join(data_dir, "train.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        f = open(os.path.join(data_dir, "test.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

    def get_labels(self):
        # return [str(x) for x in range(40)]
        return [0, 1]

    def get_label_from_expression(self, exp):
        if exp == "NULL":
            return None
        unit = ""
        day_label = 0
        unit_map = {
            "second": 0.0,
            "seconds": 0.0,
            "minute": 1.0,
            "minutes": 1.0,
            "hour": 2.0,
            "hours": 2.0,
            "day": 3.0,
            "days": 3.0,
            "week": 4.0,
            "weeks": 4.0,
            "month": 5.0,
            "months": 5.0,
            "year": 6.0,
            "years": 6.0,
            "century": 7.0,
            "centuries": 7.0,
        }
        advance_map = {
            "second": 60.0,
            "seconds": 60.0,
            "minute": 60.0,
            "minutes": 60.0,
            "hour": 24.0,
            "hours": 24.0,
            "day": 7.0,
            "days": 7.0,
            "week": 4.0,
            "weeks": 4.0,
            "month": 12.0,
            "months": 12.0,
            "year": 100.0,
            "years": 100.0,
            "century": 2.0,
            "centuries": 2.0,
        }
        value_map = {
            "second": 1.0,
            "seconds": 1.0,
            "minute": 60.0,
            "minutes": 60.0,
            "hour": 60.0 * 60.0,
            "hours": 60.0 * 60.0,
            "day": 24.0 * 60.0 * 60.0,
            "days": 24.0 * 60.0 * 60.0,
            "week": 7.0 * 24.0 * 60.0 * 60.0,
            "weeks": 7.0 * 24.0 * 60.0 * 60.0,
            "month": 28.0 * 24.0 * 60.0 * 60.0,
            "months": 28.0 * 24.0 * 60.0 * 60.0,
            "year": 336.0 * 24.0 * 60.0 * 60.0,
            "years": 336.0 * 24.0 * 60.0 * 60.0,
            "century": 100.0 * 336.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 336.0 * 24.0 * 60.0 * 60.0,
        }
        if exp.startswith("PT") and exp[-1] == "M":
            unit = "minute"
        if exp[-1] == "M" and exp[1] != "T":
            unit = "month"
        if exp[-1] == "Y":
            unit = "year"
        if exp[-1] == "D":
            unit = "day"
        if exp[-1] == "W":
            unit = "week"
        if exp[-1] == "H":
            unit = "hour"
        if exp[-1] == "S":
            unit = "second"

        if exp.startswith("PT"):
            exp = exp[2:]
        else:
            day_label = 1
            exp = exp[1:]

        exp = exp[:-1]
        label_num = float(exp)
        if unit == "":
            return None
        label_unit = unit

        # Hasn't changed to correct ones
        label_idx = int(floor(label_num * 5.0 / advance_map[label_unit]))
        if label_idx > 4:
            label_idx = 4
        if label_idx < 0:
            label_idx = 0
        label_idx = unit_map[label_unit] * 5.0 + label_idx

        return label_idx, day_label, label_num * value_map[unit]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            groups = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = groups[0]
            if len(text_a.split()) > 120:
                continue
            target_idx = int(groups[1])
            label_lower, day_label, lower_val = self.get_label_from_expression(groups[2])
            label_upper, _, upper_val = self.get_label_from_expression(groups[3])

            if label_lower is None or label_upper is None or label_lower > label_upper:
                continue

            subj_mask = [0] * len(text_a.split())
            obj_mask = [0] * len(text_a.split())

            if int(groups[4]) > -1:
                for j in range(int(groups[4]), int(groups[5])):
                    subj_mask[j] = 1
            if int(groups[6]) > -1:
                for j in range(int(groups[6]), int(groups[7])):
                    obj_mask[j] = 1

            label_num = int(float(label_upper + label_lower) / 2.0)

            lower_e = math.log(lower_val)
            upper_e = math.log(upper_val)

            if (lower_e + upper_e) / 2.0 >= 11.367:
                true_label = 1
            else:
                true_label = 0

            # examples.append(
            #     InputExample(guid=guid, text_a=text_a, label=str(label_num), target_idx=target_idx, tolerance=max(1, max(
            #         abs(label_lower - label_num), abs(label_upper - label_num)
            #     )), subj_mask=subj_mask, obj_mask=obj_mask)
            # )
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=true_label, target_idx=target_idx, tolerance=max(1, max(
                    abs(label_lower - label_num), abs(label_upper - label_num)
                )), subj_mask=subj_mask, obj_mask=obj_mask)
            )
        return examples



class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # tokens_a = tokenizer.tokenize(example.text_a)
        tokens_a = example.text_a.lower().split()

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        subj_mask = example.subj_mask
        obj_mask = example.obj_mask

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        subj_mask = [0] + subj_mask + [0] + padding
        obj_mask = [0] + obj_mask + [0] + padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(subj_mask) == max_seq_length
        assert len(obj_mask) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("subj_mask: %s" % " ".join([str(x) for x in subj_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              target_idx=example.target_idx + 1,
                              tolerance=example.tolerance,
                              subj_mask=subj_mask,
                              obj_mask=obj_mask))
    return features


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


def transform_label_to_timex(label_id):
    unit_map = {
        "second": 0.0,
        "seconds": 0.0,
        "minute": 1.0,
        "minutes": 1.0,
        "hour": 2.0,
        "hours": 2.0,
        "day": 3.0,
        "days": 3.0,
        "week": 4.0,
        "weeks": 4.0,
        "month": 5.0,
        "months": 5.0,
        "year": 6.0,
        "years": 6.0,
        "century": 7.0,
        "centuries": 7.0,
    }
    advance_map = {
        "second": 60.0,
        "seconds": 60.0,
        "minute": 60.0,
        "minutes": 60.0,
        "hour": 24.0,
        "hours": 24.0,
        "day": 7.0,
        "days": 7.0,
        "week": 4.0,
        "weeks": 4.0,
        "month": 12.0,
        "months": 12.0,
        "year": 100.0,
        "years": 100.0,
        "century": 2.0,
        "centuries": 2.0,
    }
    unit_reverse_map = {}
    for k in unit_map:
        unit_reverse_map[unit_map[k]] = k

    group = floor(label_id / 5.0)
    unit = unit_reverse_map[group]
    reminder = label_id - group * 5.0
    label_num = advance_map[unit] * reminder / 5.0
    label_num += 1.0
    return str(label_num) + " " + unit


def compute_f1(p, r):
    return 2 * p * r / (p + r)

def day_classification(preds, labels):
    small_correct = 0.0
    small_total = 0.0
    small_predicted = 0.0
    big_correct = 0.0
    big_total = 0.0
    big_predicted = 0.0

    for i, l in enumerate(labels):
        if l == 0:
            small_total += 1.0
            if preds[i] == 0:
                small_correct += 1.0
        else:
            big_total += 1.0
            if preds[i] == 1:
                big_correct += 1.0

    for p in preds:
        if p == 0:
            small_predicted += 1.0
        else:
            big_predicted += 1.0

    r_small = small_correct / small_total
    p_small = small_correct / small_predicted

    r_big = big_correct / big_total
    p_big = big_correct / big_predicted

    print("Less than a day: " + str(p_small) + ", " + str(r_small) + ", " + str(compute_f1(p_small, r_small)))
    print("More than a day: " + str(p_big) + ", " + str(r_big) + ", " + str(compute_f1(p_big, r_big)))


def acc_and_f1(preds, labels, additional=None):
    acc = simple_accuracy(preds, labels, additional)
    f_out = open("./results.txt", "w")
    for p in preds:
        f_out.write(transform_label_to_timex(p) + "\n")
    # f1 = f1_score(y_true=labels, y_pred=preds)
    day_classification(preds, labels)
    return {
        "acc": acc,
        # "f1": f1,
        # "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def avg_hit_loss(preds, labels):
    hit_10 = 0.0
    hit_20 = 0.0
    hit_30 = 0.0
    for (i, v) in enumerate(preds):
        if v * 0.9 <= labels[i] <= v * 1.1:
            hit_10 += 1.0
        if v * 0.8 <= labels[i] <= v * 1.2:
            hit_20 += 1.0
        if v * 0.7 <= labels[i] <= v * 1.3:
            hit_30 += 1.0

    return {
        "10% acc: ": hit_10 / float(len(preds)),
        "20% acc: ": hit_20 / float(len(preds)),
        "30% acc: ": hit_30 / float(len(preds)),
    }


def compute_metrics(task_name, preds, labels, additional=None):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "animal":
        return avg_hit_loss(preds, labels)
    elif task_name == "temporalnom":
        return acc_and_f1(preds, labels, additional)
    elif task_name == "timebank":
        return acc_and_f1(preds, labels, additional)
    elif task_name == "tempoalverb":
        return avg_hit_loss(preds, labels)
    else:
        raise KeyError(task_name)


oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalization factor for Gaussians


def gaussian_distribution(y, mu, sigma):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI


def mdn_loss_fn(pi, sigma, mu, y):
    result = gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    # print(result)

    m = torch.distributions.Normal(loc=mu, scale=sigma)
    zero_cdf = torch.sum(m.cdf(0.0) * pi, dim=1)
    result += zero_cdf * 10.0
    # print(zero_cdf)
    # result += torch.log(zero_cdf)
    # print(result)

    return torch.mean(result)


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
                        default=8,
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
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,
        "animal": AnimalProcessor,
        "temporalnom": TemporalNominalProcessor,
        "timebank": TimebankProcessor,
        "temporalverb": TemporalVerbProcessor,
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification",
        "animal": "regression",
        "temporalnom": "classification",
        "timebank": "regression",
        "temporalverb": "regression",
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
    num_labels = len(label_list)

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
    model = BertForTemporalClassification.from_pretrained(args.bert_model,
              cache_dir=cache_dir,
              num_labels=num_labels)
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
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_target_idxs = torch.tensor([f.target_idx for f in train_features], dtype=torch.long)
        all_subj_masks = torch.tensor([f.subj_mask for f in train_features], dtype=torch.long)
        all_obj_masks = torch.tensor([f.obj_mask for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_target_idxs, all_subj_masks, all_obj_masks
        )
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            middle_loss = 0.0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, target_idxs, subj_masks, obj_masks = batch

                # define a new function to compute loss values for both output_modes
                pi, mu, sigma = model(input_ids, segment_ids, input_mask, labels=None, target_idx=target_idxs, subj_mask=subj_masks, obj_mask=obj_masks)

                if output_mode == "classification":
                    loss = None
                elif output_mode == "regression":
                    label_ids = label_ids.unsqueeze(1)
                    loss = mdn_loss_fn(pi, sigma, mu, label_ids)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                middle_loss += loss.item()
                if step % 100 == 0:
                    print("Loss: " + str(middle_loss))
                    middle_loss = 0.0
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if step % 1000 == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
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
        model = BertForTemporalClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = BertForTemporalClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_target_idxs = torch.tensor([f.target_idx for f in eval_features], dtype=torch.long)
        all_tolerances = torch.tensor([f.tolerance for f in eval_features], dtype=torch.long)
        all_subj_masks = torch.tensor([f.subj_mask for f in eval_features], dtype=torch.long)
        all_obj_masks = torch.tensor([f.obj_mask for f in eval_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_target_idxs, all_tolerances, all_subj_masks, all_obj_masks)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        f_logits_out = open("./result_logits.txt", "w")
        for input_ids, input_mask, segment_ids, label_ids, target_idxs, tolerances, subj_masks, obj_masks in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            target_idxs = target_idxs.to(device)
            subj_masks = subj_masks.to(device)
            obj_masks = obj_masks.to(device)

            with torch.no_grad():
                pi, mu, sigma = model(input_ids, segment_ids, input_mask, labels=None, target_idx=target_idxs, subj_mask=subj_masks, obj_mask=obj_masks)

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss = None
                # loss_fct = CrossEntropyLoss(weight=loss_weight)
                # tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == "regression":
                m = torch.distributions.Normal(loc=mu, scale=sigma)
                # divident = 1451520000.0
                divident = 2592000.0
                day_val = 24.0 * 3600.0 / divident
                day_cdf = torch.sum(m.cdf(day_val) * pi, dim=1).detach().cpu().numpy()
                rest_cdf = torch.sum(m.cdf(1.0) * pi, dim=1).detach().cpu().numpy()
                zero_cdf = torch.sum(m.cdf(0.0) * pi, dim=1).detach().cpu().numpy()
                for i, v in enumerate(day_cdf):
                    if v >= 0.5 * rest_cdf[i]:
                        preds.append(0)
                    else:
                        preds.append(1)

                # prev = None
                # tmp_map = {}
                # for i in range(-180, 720):
                #     # val = float(math.exp(float(i))) / divident
                #     val = float(i) * 600 / divident
                #     cdf = torch.sum(m.cdf(val) * pi, dim=1).detach().cpu().numpy()
                #     cdf_copy = torch.sum(m.cdf(val) * pi, dim=1).detach().cpu().numpy()
                #     if prev is not None:
                #         cdf = np.subtract(cdf, prev)
                #     for j, c in enumerate(cdf):
                #         if j not in tmp_map:
                #             tmp_map[j] = []
                #         tmp_map[j].append(c)
                #     prev = cdf_copy
                # for i in range(0, 8):
                #     if i not in tmp_map:
                #         break
                #     vals = tmp_map[i]
                #     concat = ""
                #     for v in vals:
                #         concat += str(v) + "\t"
                #     concat += "\n"
                #     f_logits_out.write(concat)

                prev = None
                tmp_map = {}
                for i in range(0, 600):
                    # val = float(math.exp(float(i))) / divident
                    val = float(i) * 4320.0 / divident
                    cdf = torch.sum(m.cdf(val) * pi, dim=1).detach().cpu().numpy()
                    cdf_copy = torch.sum(m.cdf(val) * pi, dim=1).detach().cpu().numpy()
                    if prev is not None:
                        cdf = np.subtract(cdf, prev)
                    for j, c in enumerate(cdf):
                        if j not in tmp_map:
                            tmp_map[j] = []
                        tmp_map[j].append(c)
                    prev = cdf_copy
                for i in range(0, 8):
                    if i not in tmp_map:
                        break
                    vals = tmp_map[i]
                    concat = ""
                    for v in vals:
                        concat += str(v) + "\t"
                    concat += "\n"
                    f_logits_out.write(concat)

            nb_eval_steps += 1
            # if len(preds) == 0:
            #     preds.append(logits.detach().cpu().numpy())
            # else:
            #     preds[0] = np.append(
            #         preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        # preds = preds[0]
        if output_mode == "classification":
            pass
        elif output_mode == "regression":
            pass
        result = compute_metrics(task_name, preds, all_label_ids.numpy(), all_tolerances.numpy())
        loss = tr_loss/nb_tr_steps if args.do_train else None
        #
        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss
        #
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # hack for MNLI-MM
        if task_name == "mnli":
            task_name = "mnli-mm"
            processor = processors[task_name]()

            if os.path.exists(args.output_dir + '-MM') and os.listdir(args.output_dir + '-MM') and args.do_train:
                raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
            if not os.path.exists(args.output_dir + '-MM'):
                os.makedirs(args.output_dir + '-MM')

            eval_examples = processor.get_dev_examples(args.data_dir)
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)
            
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]
            preds = np.argmax(preds, axis=1)
            result = compute_metrics(task_name, preds.view(-1).numpy(), all_label_ids.numpy())
            loss = tr_loss/nb_tr_steps if args.do_train else None

            result['eval_loss'] = eval_loss
            result['global_step'] = global_step
            result['loss'] = loss

            output_eval_file = os.path.join(args.output_dir + '-MM', "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()
