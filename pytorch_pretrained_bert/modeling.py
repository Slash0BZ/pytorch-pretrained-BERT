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
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
import numpy as np
# import satnet
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from .file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'

def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertCustomEncoder(nn.Module):
    def __init__(self, config, num_layer=4):
        super(BertCustomEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states


class BertEncoderPredicate(nn.Module):
    def __init__(self, config):
        super(BertEncoderPredicate, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(4)])

    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
                        from_tf=False, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu' if not torch.cuda.is_available() else None)
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        """ Load vanilla BERT weights """
        # vanilla_archive_file = PRETRAINED_MODEL_ARCHIVE_MAP["bert-base-uncased"]
        # resolved_vanilla_file = cached_path(vanilla_archive_file, cache_dir=cache_dir)
        # tempdir = tempfile.mkdtemp()
        # logger.info("extracting archive file {} to temp dir {}".format(
        #     resolved_vanilla_file, tempdir))
        # with tarfile.open(resolved_vanilla_file, 'r:gz') as archive:
        #     archive.extractall(tempdir)
        # vanilla_weights_path = os.path.join(tempdir, WEIGHTS_NAME)
        # vanilla_state_dict = torch.load(vanilla_weights_path, map_location='cpu' if not torch.cuda.is_available() else None)
        #
        # old_keys = []
        # new_keys = []
        # for key in vanilla_state_dict.keys():
        #     new_key = None
        #     if 'gamma' in key:
        #         new_key = key.replace('gamma', 'weight')
        #     if 'beta' in key:
        #         new_key = key.replace('beta', 'bias')
        #     if new_key:
        #         old_keys.append(key)
        #         new_keys.append(new_key)
        # for old_key, new_key in zip(old_keys, new_keys):
        #     vanilla_state_dict[new_key] = vanilla_state_dict.pop(old_key)
        #
        # old_keys = []
        # new_keys = []
        # for key in state_dict.keys():
        #     if key.startswith("bert"):
        #         lookup = key.replace("bert", "bert_vanilla")
        #         old_keys.append(key)
        #         new_keys.append(lookup)
        # for o, n in zip(old_keys, new_keys):
        #     state_dict[n] = copy.deepcopy(vanilla_state_dict[o])
        # del vanilla_state_dict
        """
        LOAD VANILLA WEIGHTS END
        """

        # copy_keys = []
        # orig_keys = []
        # for key in state_dict.keys():
        #     if key.startswith("bert"):
        #         new_key = key.replace("bert", "bert_temporal")
        #         copy_keys.append(new_key)
        #         orig_keys.append(key)
        # for o, n in zip(orig_keys, copy_keys):
        #     state_dict[n] = copy.deepcopy(state_dict[o])
        #     # pass

        # copy_keys = []
        # orig_keys = []
        # for key in state_dict.keys():
        #     if key.startswith("bert_temporal"):
        #         new_key = key.replace("bert_temporal", "bert.bert_temporal")
        #         copy_keys.append(new_key)
        #         orig_keys.append(key)
        #     if key.startswith("pi_classifier"):
        #         new_key = key.replace("pi_classifier", "bert.pi_classifier")
        #         copy_keys.append(new_key)
        #         orig_keys.append(key)
        #     if key.startswith("mu_classifier"):
        #         new_key = key.replace("mu_classifier", "bert.mu_classifier")
        #         copy_keys.append(new_key)
        #         orig_keys.append(key)
        #     if key.startswith("sigma_classifier"):
        #         new_key = key.replace("sigma_classifier", "bert.sigma_classifier")
        #         copy_keys.append(new_key)
        #         orig_keys.append(key)
        # for o, n in zip(orig_keys, copy_keys):
        #     state_dict[n] = state_dict.pop(o)

        # copy_keys = []
        # orig_keys = []
        # for key in state_dict.keys():
        #     if key.startswith("bert.bert_temporal"):
        #         new_key = key.replace("bert.bert_temporal", "bert_temporal")
        #         copy_keys.append(new_key)
        #         orig_keys.append(key)
        # for o, n in zip(orig_keys, copy_keys):
        #      state_dict[n] = state_dict.pop(o)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls( pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForSingleTokenClassificationFollowTemporal(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSingleTokenClassificationFollowTemporal, self).__init__(config)
        self.num_labels = num_labels
        self.bert_temporal = BertModel(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)
        # self.log_softmax = nn.LogSoftmax(-1)
        # self.softmax = nn.Softmax(-1)

        self.pi_classifier = nn.Linear(config.hidden_size * 1, 4)
        self.mu_classifier = nn.Linear(config.hidden_size * 1, 4)
        self.sigma_classifier = nn.Linear(config.hidden_size * 1, 4)

        # self.logit_classifier_1 = nn.Linear(7, 7)
        # self.logit_classifier_2 = nn.Linear(config.hidden_size, 7)
        # self.logit_classifier_final = nn.Linear(14, 7)

        # self.mlp_layers = nn.Sequential(
        #     nn.Linear(8, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 7),
        # )

        self.rel_classifier = nn.Linear(num_labels * 2, 2)

        self.logit_classifier = nn.Linear(11, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.final_mlp_layers = nn.Sequential(
            nn.ReLU(),
            # nn.Linear(num_labels * 2, num_labels),
            nn.Linear(num_labels * 1, num_labels),
            nn.ReLU(),
            nn.Linear(num_labels, num_labels),
        )

    def get_single_inference(self, input_ids, token_type_ids, attention_mask, target_ids):
        seq_output, _ = self.bert_temporal(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        target_all_output = seq_output.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, seq_output.size(2)))
        target_all_output = self.dropout(target_all_output)

        orig_seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        orig_target_all_output = orig_seq_output.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, orig_seq_output.size(2)))
        orig_target_all_output = self.dropout(orig_target_all_output)

        pi = nn.functional.softmax(self.pi_classifier(target_all_output), -1).view(-1, 4)
        mu = self.mu_classifier(target_all_output).view(-1, 4)
        sigma = torch.exp(self.sigma_classifier(target_all_output)).view(-1, 4)

        # logits = torch.cat((pi * mu, pi * sigma), -1)
        # logits = self.mlp_layers(logits).unsqueeze(1)

        # logits = self.logit_classifier(target_all_output)
        logits_orig = self.classifier(orig_target_all_output)

        # logits = self.final_mlp_layers(torch.cat((logits, logits_orig), 2))
        # logits = self.final_mlp_layers(logits + logits_orig)
        # logits = self.final_mlp_layers(logits_orig)

        m = torch.distributions.Normal(loc=mu, scale=sigma)
        #
        logits_1 = torch.sum(m.log_prob(0) * pi, -1).unsqueeze(1)
        logits_2 = torch.sum(m.log_prob(2.7) * pi, -1).unsqueeze(1)
        logits_3 = torch.sum(m.log_prob(6.8) * pi, -1).unsqueeze(1)
        logits_4 = torch.sum(m.log_prob(10.0) * pi, -1).unsqueeze(1)
        logits_5 = torch.sum(m.log_prob(11.9) * pi, -1).unsqueeze(1)
        logits_6 = torch.sum(m.log_prob(13.3) * pi, -1).unsqueeze(1)
        logits_7 = torch.sum(m.log_prob(15.9) * pi, -1).unsqueeze(1)
        logits_8 = torch.sum(m.log_prob(18.2) * pi, -1).unsqueeze(1)
        logits_9 = torch.sum(m.log_prob(20.5) * pi, -1).unsqueeze(1)
        logits_10 = torch.sum(m.log_prob(22.0) * pi, -1).unsqueeze(1)
        logits_11 = torch.sum(m.log_prob(23.0) * pi, -1).unsqueeze(1)
        #
        logits = torch.cat(
            (
                logits_1,
                logits_2,
                logits_3,
                logits_4,
                logits_5,
                logits_6,
                logits_7,
                logits_8,
                logits_9,
                logits_10,
                logits_11,
             ), -1).unsqueeze(1)
        # logits = self.logit_classifier(logits)
        logits = self.logit_classifier(nn.functional.log_softmax(logits, -1))
        logits = self.final_mlp_layers(logits)
        # logits = self.final_mlp_layers(torch.cat((logits, logits_orig), 2))
        # logits = self.logit_classifier_1(logits)

        # orig_logits = self.logit_classifier_2(orig_target_all_output)
        # final_logits = self.logit_classifier_final(torch.cat())

        return logits

    def forward(self, input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a,
                input_ids_b, token_type_ids_b, attention_mask_b, target_ids_b):
        logits_a = self.get_single_inference(input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a)
        logits_b = self.get_single_inference(input_ids_b, token_type_ids_b, attention_mask_b, target_ids_b)

        logits_a_b = torch.cat((logits_a, logits_b), 2)
        logits_a_b = nn.functional.relu(logits_a_b)
        rel_logits = self.rel_classifier(logits_a_b)

        return torch.cat((logits_a_b, rel_logits), 2)


class BertForSingleTokenClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSingleTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.log_softmax = nn.LogSoftmax(-1)
        self.softmax = nn.Softmax(-1)

        self.comparison_classifier = nn.Linear(config.hidden_size, 1)
        self.label_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.rel_classifier = nn.Linear(2, 2)

        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

        self.lm_loss_fn = CrossEntropyLoss(ignore_index=-1)

    def get_single_inference(self, input_ids, token_type_ids, attention_mask, target_ids):
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        target_all_output = seq_output.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, seq_output.size(2)))

        pooled_output = self.dropout(target_all_output)
        logits = self.label_classifier(pooled_output)
        comparison_logits = self.comparison_classifier(pooled_output)

        # cls_logits = self.cls(seq_output)

        """CHANGE"""
        return logits, comparison_logits, None
        # return target_all_output, comparison_logits, cls_logits

    def compute_lm_loss(self, cls_output, labels):
        if labels is None:
            return None
        lm_loss = self.lm_loss_fn(cls_output.view(-1, 30522), labels.view(-1))
        return lm_loss

    def forward(self, input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a,
                input_ids_b, token_type_ids_b, attention_mask_b, target_ids_b, lm_labels_a, lm_labels_b):
        logits_a, comparison_logits_a, cls_a = self.get_single_inference(input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a)
        logits_b, comparison_logits_b, cls_b = self.get_single_inference(input_ids_b, token_type_ids_b, attention_mask_b, target_ids_b)

        logits_a_b = torch.cat((logits_a, logits_b), 2)
        logits_rel = self.rel_classifier(torch.cat((comparison_logits_a, comparison_logits_b), 2))

        """CHANGE"""
        # return torch.cat((logits_a_b, logits_rel), 2), self.compute_lm_loss(cls_a, lm_labels_a) + self.compute_lm_loss(cls_b, lm_labels_b)
        return torch.cat((logits_a_b, logits_rel), 2), None
        # return logits_a_b, None


class TemporalModelJoint(BertPreTrainedModel):
    def __init__(self, config, num_labels, num_typical_labels):
        super(TemporalModelJoint, self).__init__(config)
        self.num_labels = num_labels
        self.num_typical_labels = num_typical_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.log_softmax = nn.LogSoftmax(-1)
        self.softmax = nn.Softmax(-1)

        self.dur_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.freq_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.typical_classifier = nn.Linear(config.hidden_size, self.num_typical_labels)

        self.concat_size = self.num_labels * 2 + self.num_typical_labels

        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

        self.lm_loss_fn = CrossEntropyLoss(ignore_index=-1)

    def get_single_inference(self, input_ids, token_type_ids, attention_mask, target_ids):
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        target_all_output = seq_output.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, seq_output.size(2)))

        pooled_output = self.dropout(target_all_output)

        dur_logits = self.softmax(self.dur_classifier(pooled_output)).view(-1, self.num_labels)
        freq_logits = self.softmax(self.freq_classifier(pooled_output)).view(-1, self.num_labels)
        typical_logits = self.softmax(self.typical_classifier(pooled_output)).view(-1, self.num_typical_labels)

        cls_logits = self.cls(seq_output)

        return freq_logits, dur_logits, typical_logits, cls_logits

    def compute_lm_loss(self, cls_output, labels):
        if labels is None:
            return None
        lm_loss = self.lm_loss_fn(cls_output.view(-1, 30522), labels.view(-1))
        return lm_loss

    def forward(self, input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a,
                input_ids_b, token_type_ids_b, attention_mask_b, target_ids_b, lm_labels_a, lm_labels_b):
        freq_a, dur_a, typ_a, cls_a = self.get_single_inference(input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a)
        freq_b, dur_b, typ_b, cls_b = self.get_single_inference(input_ids_b, token_type_ids_b, attention_mask_b, target_ids_b)

        lm_loss_a = None
        lm_loss_a = self.compute_lm_loss(cls_a, lm_labels_a)
        lm_loss_b = self.compute_lm_loss(cls_b, lm_labels_b)
        if lm_loss_a is not None:
            return torch.cat((freq_a, dur_a, typ_a), -1), torch.cat((freq_b, dur_b, typ_b), -1), lm_loss_a + lm_loss_b
        else:
            return torch.cat((freq_a, dur_a, typ_a), -1), torch.cat((freq_b, dur_b, typ_b), -1), None


class TemporalModelJointNew(BertPreTrainedModel):
    def __init__(self, config, num_labels=51):
        super(TemporalModelJointNew, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.likelihood_classifier = nn.Linear(config.hidden_size, 4)

        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

        self.lm_loss_fn = CrossEntropyLoss(ignore_index=-1)

    def compute_lm_loss(self, cls_output, labels):
        if labels is None:
            return None
        lm_loss = self.lm_loss_fn(cls_output.view(-1, 30522), labels.view(-1))
        return lm_loss

    def forward(self, input_ids, token_type_ids, attention_mask, target_ids, lm_labels=None, likelihood_labels=None):
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        target_all_output = seq_output.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, seq_output.size(2)))

        pooled_output = self.dense(target_all_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        # pooled_output = self.dropout(target_all_output)
        logits = self.classifier(pooled_output)

        if lm_labels is None:
            return logits

        cls_logits = self.cls(seq_output)
        lm_loss = self.compute_lm_loss(cls_logits, lm_labels)

        if likelihood_labels is None:
            return logits, lm_loss

        likelihood_logits = self.likelihood_classifier(seq_output)
        likelihood_loss = self.lm_loss_fn(likelihood_logits.view(-1, 4), likelihood_labels.view(-1))

        return logits, lm_loss + likelihood_loss, likelihood_logits.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, 4))


class TemporalModelJointNewPureTransformer(BertPreTrainedModel):
    def __init__(self, config, num_labels=51):
        super(TemporalModelJointNewPureTransformer, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        for p in self.bert.parameters():
            p.requires_grad = False

        self.transformer = BertCustomEncoder(config, num_layer=6)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.likelihood_classifier = nn.Linear(config.hidden_size, 4)

        self.apply(self.init_bert_weights)

        self.label_loss_fn = CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, token_type_ids, attention_mask, target_ids, lm_labels=None, likelihood_labels=None):
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        seq_output = self.transformer(seq_output, attention_mask)
        target_all_output = seq_output.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, seq_output.size(2)))

        pooled_output = self.dense(target_all_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        likelihood_logits = self.likelihood_classifier(seq_output)
        likelihood_loss = self.label_loss_fn(likelihood_logits.view(-1, 4), likelihood_labels.view(-1))

        return logits, likelihood_loss, likelihood_logits.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, 4))


class TemporalModelArguments(BertPreTrainedModel):
    def __init__(self, config):
        super(TemporalModelArguments, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

        self.lm_loss_fn = CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(self, input_ids, token_type_ids, attention_mask, lm_labels, target_ids):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)
        if lm_labels is not None:
            masked_lm_loss = self.lm_loss_fn(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
        else:
            masked_lm_loss = None
        ret_cls = prediction_scores.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, prediction_scores.size(2)))

        return masked_lm_loss, ret_cls


class TargetLMPrediction(BertPreTrainedModel):
    def __init__(self, config):
        super(TargetLMPrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, target_ids):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)
        ret_cls = prediction_scores.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, prediction_scores.size(2)))

        return ret_cls


class TemporalModelJointWithLikelihood(BertPreTrainedModel):

    def __init__(self, config, num_labels, num_typical_labels):
        super(TemporalModelJointWithLikelihood, self).__init__(config)
        self.num_labels = num_labels
        self.num_typical_labels = num_typical_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.log_softmax = nn.LogSoftmax(-1)
        self.softmax = nn.Softmax(-1)

        self.dur_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.freq_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.typical_classifier = nn.Linear(config.hidden_size, self.num_typical_labels)

        self.likelihood_classifier = nn.Linear(config.hidden_size, 2)

        self.concat_size = self.num_labels * 2 + self.num_typical_labels

        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

        self.lm_loss_fn = CrossEntropyLoss(ignore_index=-1)

    def get_single_inference(self, input_ids, token_type_ids, attention_mask, target_ids):
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        target_all_output = seq_output.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, seq_output.size(2)))

        pooled_output = self.dropout(target_all_output)

        dur_logits = self.softmax(self.dur_classifier(pooled_output)).view(-1, self.num_labels)
        freq_logits = self.softmax(self.freq_classifier(pooled_output)).view(-1, self.num_labels)
        typical_logits = self.softmax(self.typical_classifier(pooled_output)).view(-1, self.num_typical_labels)

        likelihood_logits = self.likelihood_classifier(seq_output)
        cls_logits = self.cls(seq_output)

        return freq_logits, dur_logits, typical_logits, cls_logits, likelihood_logits

    def compute_aux_loss(self, cls_output, likelihood_output, lm_labels, likelihood_labels):
        lm_loss = None
        if lm_labels is not None:
            lm_loss = self.lm_loss_fn(cls_output.view(-1, 30522), lm_labels.view(-1))
        likelihood_loss = None
        if likelihood_labels is not None:
            likelihood_loss = self.lm_loss_fn(likelihood_output.view(-1, 2), likelihood_labels.view(-1))
        if likelihood_loss is not None:
            if lm_loss is not None:
                return likelihood_loss + lm_loss
            else:
                return lm_loss
        else:
            if lm_loss is not None:
                return lm_loss
            else:
                return None

    def forward(self, input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a,
                input_ids_b, token_type_ids_b, attention_mask_b, target_ids_b, lm_labels_a, lm_labels_b, lh_labels_a, lh_labels_b):

        freq_a, dur_a, typ_a, cls_a, lh_a = self.get_single_inference(input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a)
        aux_loss_a = self.compute_aux_loss(cls_a, lh_a, lm_labels_a, lh_labels_a)
        freq_b, dur_b, typ_b, cls_b, lh_b = self.get_single_inference(input_ids_b, token_type_ids_b, attention_mask_b, target_ids_b)
        aux_loss_b = self.compute_aux_loss(cls_b, lh_b, lm_labels_b, lh_labels_b)

        if aux_loss_a is not None:
            return torch.cat((freq_a, dur_a, typ_a), -1), torch.cat((freq_b, dur_b, typ_b), -1), aux_loss_a + aux_loss_b
        else:
            return torch.cat((freq_a, dur_a, typ_a), -1), torch.cat((freq_b, dur_b, typ_b), -1), None


class HieveModel(BertPreTrainedModel):
    def __init__(self, config, num_labels=4):
        super(HieveModel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        # for p in self.bert.parameters():
        #     p.requires_grad = False
        # self.transformer = BertCustomEncoder(config, num_layer=6)

        self.pooler_dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.pooler_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 51)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

        self.label_loss_fn = CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, token_type_ids, attention_mask, target_ids_a, target_ids_b, labels=None):
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # seq_output = self.transformer(seq_output, attention_mask)

        output_a = seq_output.gather(1, target_ids_a.view(-1, 1).unsqueeze(2).repeat(1, 1, seq_output.size(2)))
        output_b = seq_output.gather(1, target_ids_b.view(-1, 1).unsqueeze(2).repeat(1, 1, seq_output.size(2)))

        # output_a = self.dense(output_a)
        # output_a = self.activation(output_a)
        # output_a = self.dropout(output_a)
        # output_a = self.classifier(output_a)
        # output_b = self.dense(output_b)
        # output_b = self.activation(output_b)
        # output_b = self.dropout(output_b)
        # output_b = self.classifier(output_b)

        pooled_output = self.pooler_dense(torch.cat((output_a, output_b), -1))
        pooled_output = self.pooler_activation(pooled_output)
        logits = self.pooler_classifier(pooled_output)

        if labels is None:
            return logits
        else:
            return self.label_loss_fn(logits.view(-1, 4), labels.view(-1))


class BertForSingleTokenClassificationWithPooler(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSingleTokenClassificationWithPooler, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.log_softmax = nn.LogSoftmax(-1)
        self.softmax = nn.Softmax(-1)

        self.comparison_classifier = nn.Linear(config.hidden_size, 1)
        self.label_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.rel_classifier = nn.Linear(2, 2)

        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

        self.lm_loss_fn = CrossEntropyLoss(ignore_index=-1)

    def get_single_inference(self, input_ids, token_type_ids, attention_mask, target_ids):
        seq_output, target_all_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        pooled_output = self.dropout(target_all_output)
        logits = self.label_classifier(pooled_output)
        comparison_logits = self.comparison_classifier(pooled_output)

        cls_logits = self.cls(seq_output)

        return logits, comparison_logits, cls_logits

    def compute_lm_loss(self, cls_output, labels):
        if labels is None:
            return None
        lm_loss = self.lm_loss_fn(cls_output.view(-1, 30522), labels.view(-1))
        return lm_loss

    def forward(self, input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a,
                input_ids_b, token_type_ids_b, attention_mask_b, target_ids_b, lm_labels_a, lm_labels_b):
        logits_a, comparison_logits_a, cls_a = self.get_single_inference(input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a)
        logits_b, comparison_logits_b, cls_b = self.get_single_inference(input_ids_b, token_type_ids_b, attention_mask_b, target_ids_b)

        logits_a_b = torch.cat((logits_a, logits_b), -1)
        logits_rel = self.rel_classifier(torch.cat((comparison_logits_a, comparison_logits_b), -1))

        # return torch.cat((logits_a_b, logits_rel), -1), self.compute_lm_loss(cls_a, lm_labels_a) + self.compute_lm_loss(cls_b, lm_labels_b)
        return torch.cat((logits_a_b, logits_rel), -1), None


class BertForSingleTokenClassificationWithVanilla(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSingleTokenClassificationWithVanilla, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.bert_vanilla = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)
        self.log_softmax = nn.LogSoftmax(-1)
        self.softmax = nn.Softmax(-1)

        self.comparison_classifier = nn.Linear(config.hidden_size, 1)
        self.label_classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.label_classifier_vanilla = nn.Linear(config.hidden_size, self.num_labels)

        # self.label_shift_classifier = nn.Linear(self.num_labels, self.num_labels)
        # self.label_shift_classifier.weight.data.copy_(torch.eye(self.num_labels))

        self.rel_classifier = nn.Linear(2, 2)

    def get_single_inference(self, input_ids, token_type_ids, attention_mask, target_ids):
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        target_all_output = seq_output.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, seq_output.size(2)))
        pooled_output = self.dropout(target_all_output)

        logits = self.label_classifier(pooled_output)
        # final_logits = self.label_shift_classifier(logits)

        seq_output_vanilla, _ = self.bert_vanilla(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        target_all_output = seq_output_vanilla.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, seq_output.size(2)))
        pooled_output_vanilla = self.dropout(target_all_output)
        logits_vanilla = self.label_classifier(pooled_output_vanilla)

        final_logits = logits + logits_vanilla
        # final_logits = logits

        comparison_logits = self.comparison_classifier(pooled_output)

        return final_logits, comparison_logits

    def forward(self, input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a,
                input_ids_b, token_type_ids_b, attention_mask_b, target_ids_b):
        logits_a, comparison_logits_a = self.get_single_inference(input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a)
        logits_b, comparison_logits_b = self.get_single_inference(input_ids_b, token_type_ids_b, attention_mask_b, target_ids_b)

        logits_a_b = torch.cat((logits_a, logits_b), 2)
        logits_rel = self.rel_classifier(torch.cat((comparison_logits_a, comparison_logits_b), 2))

        return torch.cat((logits_a_b, logits_rel), 2)


class BertForTemporalClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForTemporalClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert_temporal = BertModel(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)

        # self.subj_attention = BertEncoderPredicate(config)
        # self.obj_attention = BertEncoderPredicate(config)
        # self.arg3_attention = BertEncoderPredicate(config)
        # self.all_attention = BertEncoderPredicate(config)

        self.n_gussians = 4

        self.pi_classifier = nn.Linear(config.hidden_size * 1, self.n_gussians)
        self.mu_classifier = nn.Linear(config.hidden_size * 1, self.n_gussians)
        self.sigma_classifier = nn.Linear(config.hidden_size * 1, self.n_gussians)

        # self.main_range = 290304000.0
        # self.mu_weight = torch.tensor([
        #     math.exp(5.0) / self.main_range,
        #     (math.exp(10.0) - math.exp(5.0)) / self.main_range,
        #     (math.exp(15.0) - math.exp(10.0)) / self.main_range,
        #     (math.exp(20.0) - math.exp(15.0)) / self.main_range,
        # ], dtype=torch.float).cuda()
        # self.mu_bias = torch.tensor([
        #     0,
        #     math.exp(5.0) / self.main_range,
        #     math.exp(10.0) / self.main_range,
        #     math.exp(15.0) / self.main_range,
        # ], dtype=torch.float).cuda()

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, target_idx=None, subj_mask=None, obj_mask=None, arg3_mask=None):
        sequence_output, _ = self.bert_temporal(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        # subj_output = self.subj_attention(sequence_output, subj_mask)
        # target_subj_output = subj_output.gather(1, target_idx.view(-1, 1).unsqueeze(2).repeat(1, 1, subj_output.size(2)))
        #
        # obj_output = self.obj_attention(sequence_output, obj_mask)
        # target_obj_output = obj_output.gather(1, target_idx.view(-1, 1).unsqueeze(2).repeat(1, 1, obj_output.size(2)))
        #
        # arg3_output = self.arg3_attention(sequence_output, arg3_mask)
        # target_arg3_output = arg3_output.gather(1, target_idx.view(-1, 1).unsqueeze(2).repeat(1, 1, arg3_output.size(2)))

        # all_output = self.all_attention(sequence_output, attention_mask)
        # target_all_output = all_output.gather(1, target_idx.view(-1, 1).unsqueeze(2).repeat(1, 1, all_output.size(2)))
        target_all_output = sequence_output.gather(1, target_idx.view(-1, 1).unsqueeze(2).repeat(1, 1, sequence_output.size(2)))
        states = target_all_output

        # states = torch.cat((target_subj_output, target_obj_output), 2)
        # states = torch.cat((target_subj_output, target_obj_output, target_arg3_output), 2)
        # states = target_subj_output + target_obj_output + target_arg3_output
        states = self.dropout(states)

        pi = nn.functional.softmax(self.pi_classifier(states), -1)
        mu = self.mu_classifier(states).view(-1, self.n_gussians)
        sigma = torch.exp(self.sigma_classifier(states))
        # mu = nn.functional.sigmoid(self.mu_classifier(states)).view(-1, self.n_gussians)
        # sigma = nn.functional.sigmoid(self.sigma_classifier(states))

        # mu = mu * self.mu_weight.repeat(mu.size()[0], 1)
        # mu = mu + self.mu_bias.repeat(mu.size()[0], 1)
        # sigma = torch.mul(nn.functional.sigmoid(self.sigma_classifier(states)).view(-1, self.n_gussians), 0.5) * mu

        if labels is not None:
            """NOT IN USE"""
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # return loss
            return None
        else:
            return pi.view(-1, self.n_gussians), mu.view(-1, self.n_gussians), sigma.view(-1, self.n_gussians)


class BertForMultipleChoice(BertPreTrainedModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_choices):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class BertForTokenClassification(BertPreTrainedModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


class BertBoundary(BertPreTrainedModel):
    def __init__(self, config, num_labels=4):
        super(BertBoundary, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.apply(self.init_bert_weights)
        self.classifier = nn.Linear(config.hidden_size, 4)

        self.lm_loss_fn = CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, token_type_ids, attention_mask, lm_labels=None):
        seq_output, target_all_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(seq_output)
        logits = self.classifier(pooled_output)

        if lm_labels is None:
            return logits
        else:
            loss = self.lm_loss_fn(logits.view(-1, 4), lm_labels.view(-1))
            return loss
