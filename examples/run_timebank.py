from __future__ import absolute_import, division, print_function, unicode_literals

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from tqdm import tqdm, trange

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import gensim
import numpy as np
import math
import random


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class TimeBankClassification(nn.Module):
    def __init__(self, num_labels, input_size):
        super(TimeBankClassification, self).__init__()
        self.num_labels = num_labels
        self.hidden_size = 768

        self.intermediate_1 = nn.Linear(input_size, self.hidden_size)
        self.intermediate_2 = nn.Linear(self.hidden_size, 16)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(input_size, num_labels)
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, inputs, labels=None):
        logits = self.classifier(inputs)

        if labels is not None:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class Runner:

    class Instance:

        def __init__(self, input, label, num_label, input_size):
            self.input = input
            self.label = label
            self.num_label = num_label
            self.input_size = input_size

    def __init__(self, data_file, embedding_file):
        self.num_label = 2
        self.input_size = 100

        self.embedding = [x.strip().split("\t") for x in open(embedding_file).readlines()]
        # self.glove_model = gensim.models.KeyedVectors.load_word2vec_format("data/glove_model.txt", binary=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.convert_map = {
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
            "month": 30.0 * 24.0 * 60.0 * 60.0,
            "months": 30.0 * 24.0 * 60.0 * 60.0,
            "year": 365.0 * 24.0 * 60.0 * 60.0,
            "years": 365.0 * 24.0 * 60.0 * 60.0,
            "century": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
        }
        self.all_data = self.load_data_file(data_file)
        random.shuffle(self.all_data)

        self.train_data = self.all_data

        new_train_data = []
        test_data = []
        label_count = [0, 0]
        for d in self.train_data:
            if label_count[d.label] >= 100:
                test_data.append(d)
                continue
            label_count[d.label] += 1
            new_train_data.append(d)
        self.train_data = new_train_data
        self.eval_data = test_data

        self.model = TimeBankClassification(self.num_label, self.input_size).to(self.device)

    def get_seconds(self, exp):

        unit = ""
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
            exp = exp[1:]
        if unit == "" or exp == "NULL":
            return None
        exp = exp[:-1]
        label_num = float(exp)

        return label_num * self.convert_map[unit]

    def load_data_file(self, data_file):
        instances = []
        for i, line in enumerate([x.strip() for x in open(data_file).readlines()]):
            group = line.split("\t")
            lower = self.get_seconds(group[2])
            upper = self.get_seconds(group[3])
            lower_e = math.log(lower)
            upper_e = math.log(upper)

            if (lower_e + upper_e) / 2.0 >= 11.367:
                label = 1
            else:
                label = 0

            input = [float(x) for x in self.embedding[i]]
            # input_1 = input[:24]
            # input = [float(sum(input_1))]
            # input_2 = input[5:10]
            # input_3 = input[200:205]
            # input_4 = input[300:305]
            # input = [float(sum(input_1)), float(sum(input_2)), float(sum(input_3)), float(sum(input_4))]
            # input_24 = input[:24]
            # input_rest = input[24:]
            # input = [float(sum(input_24)), float(sum(input_rest))]
            # input = [float(random.random()) for x in self.embedding[i]]
            # input = [0] * 100
            # verb = group[0].split()[int(group[1])]
            # if verb in self.glove_model.vocab:
            #     input = list(self.glove_model.get_vector(verb))

            instance = self.Instance(
                input, label, self.num_label, self.input_size
            )
            instances.append(instance)

        return instances

    def adjust_learning_rate(self, optimizer, epoch, orig_lr):
        lr = orig_lr * (0.1 ** (epoch // 200))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, data, epoch=5, batch_size=32, learning_rate=0.2):

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        all_train_input = torch.tensor([f.input for f in data], dtype=torch.float)
        all_train_labels = torch.tensor([f.label for f in data], dtype=torch.long)

        train_data_tensor = TensorDataset(
            all_train_input, all_train_labels
        )
        train_sampler = RandomSampler(train_data_tensor)
        train_dataloader = DataLoader(train_data_tensor, sampler=train_sampler, batch_size=batch_size)

        self.model.train()
        tmp_loss = 0
        global_step = 0
        for _ in trange(epoch, desc="Epoch"):
            epoch_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                inputs, labels = batch

                logits = self.model(inputs, None)

                loss = loss_fn(logits.view(-1, self.num_label), labels.view(-1))
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                self.adjust_learning_rate(optimizer, _, learning_rate)

            tmp_loss += epoch_loss
            if _ % 10 == 0:
                for param_group in optimizer.param_groups:
                    print(param_group['lr'])
                print("Avg Epoch Loss: " + str(tmp_loss / 10.0))
                tmp_loss = 0

    def eval(self, data, batch_size=8):
        all_test_inputs = torch.tensor([f.input for f in data], dtype=torch.float)
        all_test_labels = torch.tensor([f.label for f in data], dtype=torch.long)

        eval_data = TensorDataset(
            all_test_inputs, all_test_labels
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        self.model.eval()
        all_preds = []
        all_labels = []
        for inputs, labels in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).detach().cpu().numpy()

            with torch.no_grad():
                logits = self.model(inputs, None).detach().cpu().numpy()
                # logits[:, 0] += 0.9
            preds = np.argmax(logits, axis=1)

            all_preds += list(preds)
            all_labels += list(labels)

        assert len(all_preds) == len(all_labels)
        correct = 0.0
        s_labeled = 0.0
        s_predicted = 0.0
        s_correct = 0.0
        l_labeled = 0.0
        l_predicted = 0.0
        l_correct = 0.0
        for i, p in enumerate(all_preds):
            if all_labels[i] == p:
                correct += 1.0
                if p == 0.0:
                    s_correct += 1.0
                else:
                    l_correct += 1.0
        for p in all_preds:
            if p == 1.0:
                l_predicted += 1.0
            else:
                s_predicted += 1.0
        for l in all_labels:
            if l == 1.0:
                l_labeled += 1.0
            else:
                s_labeled += 1.0

        print("Acc.: " + str(correct / float(len(all_preds))))
        print("Less than a day: " + str(s_correct / s_predicted) + ", " + str(s_correct / s_labeled))
        print("Longer than a day: " + str(l_correct / l_predicted) + ", " + str(l_correct / l_labeled))


if __name__ == "__main__":
    runner = Runner(
        data_file="samples/duration/timebank_svo_verbonly.txt",
        embedding_file="./result_logits.txt"
    )
    print(len(runner.train_data))
    print(len(runner.eval_data))
    runner.train(runner.train_data[:200], epoch=1000, batch_size=1)
    runner.eval(runner.eval_data)
