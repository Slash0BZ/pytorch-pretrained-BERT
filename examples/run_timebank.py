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
from torch.optim.lr_scheduler import LambdaLR

random.seed(9001)

import nltk
# nltk.download('wordnet', download_dir='/scratch/xzhou45/bert/venv/nltk_data')
from nltk.stem import WordNetLemmatizer



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
    def __init__(self, num_labels, duration_input_size=2, verb_input_size=1, vocab_size=100):
        super(TimeBankClassification, self).__init__()
        print(vocab_size)
        self.num_labels = num_labels
        self.duration_input_size = duration_input_size
        self.verb_input_size = verb_input_size
        self.embed = nn.Embedding(vocab_size, 1)
        self.duration_classifier = nn.Sequential(
            nn.Linear(self.duration_input_size, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )
        self.verb_classifier = nn.Linear(1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, self.num_labels)
        )

    def forward(self, inputs_duration, inputs_verb, labels=None):
        logits = self.duration_classifier(nn.functional.softmax(inputs_duration, -1))
        logits_verb = self.verb_classifier(self.embed(inputs_verb)).view(-1, 1)
        logits = self.classifier(torch.cat((logits, logits_verb), -1))
        # logits = self.classifier(torch.cat((logits, self.embed(inputs_verb).view(-1, 2)), -1))

        if labels is not None:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class Runner:

    class Instance:

        def __init__(self, input_duration, input_verb, label, num_label, input_size_duration, input_size_verb):
            self.input_duration = input_duration
            self.input_verb = input_verb
            self.label = label
            self.num_label = num_label
            self.input_size_duration = input_size_duration
            self.input_size_verb = input_size_verb

    def __init__(self, data_file, embedding_file):
        self.num_label = 2
        self.vocab = {}
        self.lemmatizer = WordNetLemmatizer()
        self.embedding = [x.strip().split("\t") for x in open(embedding_file).readlines()]
        # self.glove_model = gensim.models.KeyedVectors.load_word2vec_format("data/glove_model.txt", binary=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.all_data = self.load_data_file(data_file)
        random.shuffle(self.all_data)

        self.train_data = self.all_data

        new_train_data = []
        test_data = []
        label_count = [0, 0]
        for d in self.train_data:
            if label_count[d.label] >= 500:
                test_data.append(d)
                continue
            label_count[d.label] += 1
            new_train_data.append(d)
        self.train_data = new_train_data
        self.eval_data = test_data

        self.model = TimeBankClassification(self.num_label, vocab_size=len(self.vocab)).to(self.device)


    def custom_softmax(self, l):
        ret_list = []
        s = 0.0
        for num in l:
            s += math.exp(num)
        for num in l:
            ret_list.append(math.exp(num) / s)
        return ret_list

    def load_data_file(self, data_file):
        instances = []
        for i, line in enumerate([x.strip() for x in open(data_file).readlines()]):
            group = line.split("\t")
            label_a = 0
            if group[2] == "1 years":
                label_a = 1
            label_b = 0
            if group[5] == "1 years":
                label_b = 1

            input = [float(x) for x in self.embedding[i]]

            verb_a = self.lemmatizer.lemmatize(group[0].split()[int(group[1])].lower(), 'v')
            verb_b = self.lemmatizer.lemmatize(group[3].split()[int(group[4])].lower(), 'v')

            if verb_a not in self.vocab:
                self.vocab[verb_a] = len(self.vocab)
            if verb_b not in self.vocab:
                self.vocab[verb_b] = len(self.vocab)

            # glove_a = [0] * 100
            # glove_b = [0] * 100
            #
            # if verb_a in self.glove_model.vocab:
            #     glove_a = list(self.glove_model.get_vector(verb_a))
            # if verb_b in self.glove_model.vocab:
            #     glove_b = list(self.glove_model.get_vector(verb_b))

            scores_a = self.custom_softmax(input[0:9])
            instance = self.Instance(
                # input[0:9], glove_a, label_a, self.num_label, 9, 100
                # input[0:9], [self.vocab[verb_a]], label_a, self.num_label, 9, 1
                # [sum(scores_a[0:3]), sum(scores_a[3:9])], [self.vocab[verb_a]], label_a, self.num_label, 2, 1
                [0.0, 0.0], [self.vocab[verb_a]], label_a, self.num_label, 2, 1
                # [0] * 9, glove_a, label_a, self.num_label, 9, 100
            )
            instances.append(instance)

            scores_b = self.custom_softmax(input[9:18])
            instance = self.Instance(
                # input[9:18], glove_b, label_b, self.num_label, 9, 100
                # input[9:18], [self.vocab[verb_b]], label_b, self.num_label, 9, 1
                # [sum(scores_b[0:3]), sum(scores_b[3:9])], [self.vocab[verb_b]], label_b, self.num_label, 2, 1
                [0.0, 0.0], [self.vocab[verb_b]], label_b, self.num_label, 2, 1
                # [0] * 9, glove_b, label_b, self.num_label, 9, 100
            )
            instances.append(instance)

        return instances

    def adjust_learning_rate(self, optimizer, epoch, orig_lr):
        lr = orig_lr * (0.1 ** (epoch // 200))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, data, epoch=5, batch_size=32, learning_rate=0.02):

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=3200,
                                         t_total=32000)
        loss_fn = nn.CrossEntropyLoss()

        all_train_input_duration = torch.tensor([f.input_duration for f in data], dtype=torch.float)
        all_train_input_verb = torch.tensor([f.input_verb for f in data], dtype=torch.long)
        all_train_labels = torch.tensor([f.label for f in data], dtype=torch.long)

        train_data_tensor = TensorDataset(
            all_train_input_duration, all_train_input_verb, all_train_labels
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
                inputs_duration, inputs_verb, labels = batch

                logits = self.model(inputs_duration, inputs_verb, None)

                loss = loss_fn(logits.view(-1, self.num_label), labels.view(-1))
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                global_step += 1

                # self.adjust_learning_rate(optimizer, _, learning_rate)

            tmp_loss += epoch_loss
            if _ % 100 == 0:
                for param_group in optimizer.param_groups:
                    print(param_group['lr'])
                print("Avg Epoch Loss: " + str(tmp_loss / 10.0))
                tmp_loss = 0

    def eval(self, data, out_file, batch_size=8):
        all_test_inputs_duration = torch.tensor([f.input_duration for f in data], dtype=torch.float)
        all_test_inputs_verb = torch.tensor([f.input_verb for f in data], dtype=torch.long)
        all_test_labels = torch.tensor([f.label for f in data], dtype=torch.long)

        eval_data = TensorDataset(
            all_test_inputs_duration, all_test_inputs_verb, all_test_labels
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        self.model.eval()
        all_preds = []
        all_labels = []
        for inputs_duration, inputs_verb, labels in tqdm(eval_dataloader, desc="Evaluating"):
            inputs_duration = inputs_duration.to(self.device)
            inputs_verb = inputs_verb.to(self.device)
            labels = labels.to(self.device).detach().cpu().numpy()

            with torch.no_grad():
                logits = self.model(inputs_duration, inputs_verb, None).detach().cpu().numpy()

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

        if out_file is not None:
            with open(out_file, "a") as f_out:
                f_out.write("Acc.: " + str(correct / float(len(all_preds))))
                f_out.write("\n")
                f_out.write("Less than a day: " + str(s_correct / s_predicted) + ", " + str(s_correct / s_labeled))
                f_out.write("\n")
                f_out.write("Longer than a day: " + str(l_correct / l_predicted) + ", " + str(l_correct / l_labeled))
                f_out.write("\n")
        else:
            print("Acc.: " + str(correct / float(len(all_preds))))
            print("Less than a day: " + str(s_correct / s_predicted) + ", " + str(s_correct / s_labeled))
            print("Longer than a day: " + str(l_correct / l_predicted) + ", " + str(l_correct / l_labeled))



if __name__ == "__main__":
    runner = Runner(
        data_file="samples/timebank/test.formatted.txt",
        embedding_file="bert_timebank_eval/bert_logits.txt"
    )
    print(len(runner.train_data))
    print(len(runner.eval_data))
    runner.train(runner.train_data, epoch=1000, batch_size=32)
    runner.eval(runner.train_data, None)
    print(runner.train_data[-1].input_verb)
    runner.eval(runner.eval_data, "timebank_result.txt")
