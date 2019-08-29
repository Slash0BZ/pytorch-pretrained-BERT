from __future__ import absolute_import, division, print_function, unicode_literals

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from tqdm import tqdm, trange

import pickle
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np
import gensim
import os


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


class VerbPhysicsClassification(nn.Module):
    def __init__(self, num_labels, input_size, glove_size):
        super(VerbPhysicsClassification, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.hidden_size = 768

        self.classifier_1 = nn.Linear(input_size, self.hidden_size)
        self.classifier_2 = nn.Linear(input_size, self.hidden_size)

        self.glove_classifier_1 = nn.Linear(glove_size, 32)
        self.glove_classifier_2 = nn.Linear(glove_size, 32)

        self.classifier_duration = nn.Linear(self.hidden_size * 2, num_labels)
        self.classifier_glove = nn.Linear(64, num_labels)

        self.classifier_dim = nn.Linear(glove_size, num_labels)

        self.classifier = nn.Linear(num_labels * 2, num_labels)

    def forward(self, input_1s, input_2s, embedding_1s, embedding_2s, dim_embeds, labels=None):
        state_1 = F.relu(self.classifier_1(self.dropout(input_1s)))
        state_2 = F.relu(self.classifier_2(self.dropout(input_2s)))
        state = torch.cat((state_1, state_2), dim=1)
        state = self.classifier_duration(state)

        state_glove_1 = F.relu(self.glove_classifier_1(embedding_1s))
        state_glove_2 = F.relu(self.glove_classifier_2(embedding_2s))
        state_glove = torch.cat((state_glove_1, state_glove_2), dim=1)
        state_glove = F.relu(self.classifier_glove(state_glove))

        state_dim = F.relu(self.classifier_dim(dim_embeds))

        # logits = self.classifier(torch.cat((torch.cat((state, state_glove), dim=1), state_dim), dim=1))
        logits = self.classifier(torch.cat((state, state_glove), dim=1))

        if labels is not None:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class FakeGlove:

    def __init__(self, m):
        self.vocab = m

    def get_vector(self, s):
        return self.vocab[s]


class Runner:

    class Instance:

        def __init__(self, input_1, input_2, glove_1, glove_2, label, num_label,
                     input_size, agreement, dimension, dimension_name="size"):
            self.input_1 = input_1
            self.input_2 = input_2
            self.glove_1 = glove_1
            self.glove_2 = glove_2
            self.label = label
            self.num_label = num_label
            self.input_size = input_size
            self.agreement = agreement
            self.dimension_embed = dimension
            self.dimension_name = dimension_name

    def __init__(self, train_data, dev_data, test_data, embedding_file):
        self.num_label = 3
        self.input_size = 700
        self.glove_size = 100

        # if os.path.isfile("./glove_cache.pkl"):
        #     self.glove_model = FakeGlove(pickle.load(open("./glove_cache.pkl", "rb")))
        # else:
        self.glove_model = gensim.models.KeyedVectors.load_word2vec_format("data/glove_model.txt", binary=False)
        self.embedding = pickle.load(open(embedding_file, "rb"))
        self.cache_glove_map = {}
        self.train_data = self.load_data_file(train_data)
        self.dev_data = self.load_data_file(dev_data)
        self.test_data = self.load_data_file(test_data)
        with open("./glove_cache.pkl", "wb") as f:
            pickle.dump(self.cache_glove_map, f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VerbPhysicsClassification(self.num_label, self.input_size, self.glove_size).to(self.device)

    def sub_lists(self, a, b):
        assert len(a) == len(b)
        for i, _ in enumerate(a):
            a[i] = a[i] - b[i]
        return a

    def adjust_learning_rate(self, optimizer, epoch, orig_lr):
        lr = orig_lr * (0.1 ** (epoch // 200))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def load_data_file(self, file_name):
        instances = []
        for line in [x.strip() for x in open(file_name).readlines()]:
            if line[0] == "," or line[0:4] == "obj1":
                continue
            group = line.split(",")
            obj_1 = group[0]
            obj_2 = group[1]

            label_map = {
                "1": 0,
                "-1": 1,
                "0": 2,
                "-42": 3,
            }

            glove_1 = [0] * self.glove_size
            glove_2 = [0] * self.glove_size

            if obj_1 in self.glove_model.vocab:
                glove_1 = list(self.glove_model.get_vector(obj_1))
                # self.cache_glove_map[obj_1] = glove_1
            if obj_2 in self.glove_model.vocab:
                glove_2 = list(self.glove_model.get_vector(obj_2))
                # self.cache_glove_map[obj_2] = glove_2

            instance = self.Instance(
                self.embedding[obj_1], self.embedding[obj_2],
                # [0] * 100, [0] * 100,
                glove_1, glove_2,
                label_map[group[3]], self.num_label, self.input_size, 3,
                list([0] * 100),
                dimension_name="size"
            )
            if instance.label != 3 and group[2].lower() == "mass":
                instances.append(instance)

            # Adding weight
            # instance = self.Instance(
            #     self.embedding[obj_1], self.embedding[obj_2],
            #     # [0] * 1500, [0] * 1500,
            #     glove_1, glove_2,
            #     label_map[group[6]], self.num_label, self.input_size, int(group[5]),
            #     # list(self.glove_model.get_vector("weight")),
            #     list([1] * 100),
            #     dimension_name="weight"
            # )
            # if instance.label != 3:
            #     instances.append(instance)

            # instance = self.Instance(
            #     self.embedding[obj_1], self.embedding[obj_2],
            #     # [0] * 150, [0] * 150,
            #     glove_1, glove_2,
            #     label_map[group[8]], self.num_label, self.input_size, int(group[7]),
            #     # list(self.glove_model.get_vector("weight")),
            #     list([1] * 100),
            #     dimension_name="strength"
            # )
            # if instance.label != 3:
            #     instances.append(instance)

            # instance = self.Instance(
            #     self.embedding[obj_1], self.embedding[obj_2],
            #     # [0] * 1500, [0] * 1500,
            #     glove_1, glove_2,
            #     label_map[group[10]], self.num_label, self.input_size, int(group[9]),
            #     # list(self.glove_model.get_vector("weight")),
            #     list([1] * 100),
            #     dimension_name="rigid"
            # )
            # if instance.label != 3:
            #     instances.append(instance)

            # instance = self.Instance(
            #     self.embedding[obj_1], self.embedding[obj_2],
            #     # [0] * 150, [0] * 150,
            #     glove_1, glove_2,
            #     label_map[group[12]], self.num_label, self.input_size, int(group[11]),
            #     # list(self.glove_model.get_vector("weight")),
            #     list([1] * 100),
            #     dimension_name="rigid"
            # )
            # if instance.label != 3:
            #     instances.append(instance)

        return instances

    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return max((x-1.)/(warmup-1.), 0)

    def train(self, epoch=5, batch_size=32, learning_rate=2e-5):

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        data = []
        for d in self.train_data:
            data.append(d)

        all_train_input_1 = torch.tensor([f.input_1 for f in data], dtype=torch.float)
        all_train_input_2 = torch.tensor([f.input_2 for f in data], dtype=torch.float)
        all_train_glove_1 = torch.tensor([f.glove_1 for f in data], dtype=torch.float)
        all_train_glove_2 = torch.tensor([f.glove_2 for f in data], dtype=torch.float)
        all_train_dim_embeds = torch.tensor([f.dimension_embed for f in data], dtype=torch.float)
        all_train_labels = torch.tensor([f.label for f in data], dtype=torch.long)

        train_data_tensor = TensorDataset(
            all_train_input_1, all_train_input_2, all_train_glove_1, all_train_glove_2, all_train_dim_embeds, all_train_labels
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
                input_1s, input_2s, glove_1s, glove_2s, dim_embeds, labels = batch

                logits = self.model(input_1s, input_2s, glove_1s, glove_2s, dim_embeds, None)

                loss = loss_fn(logits.view(-1, self.num_label), labels.view(-1))
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                # self.adjust_learning_rate(optimizer, _, learning_rate)
            tmp_loss += epoch_loss
            if _ % 500 == 0:
                for param_group in optimizer.param_groups:
                    print(param_group['lr'])
                print("Avg Epoch Loss: " + str(tmp_loss / 10.0))
                tmp_loss = 0

    def eval(self, data, batch_size=8):
        all_test_input_1 = torch.tensor([f.input_1 for f in data], dtype=torch.float)
        all_test_input_2 = torch.tensor([f.input_2 for f in data], dtype=torch.float)
        all_test_glove_1 = torch.tensor([f.glove_1 for f in data], dtype=torch.float)
        all_test_glove_2 = torch.tensor([f.glove_2 for f in data], dtype=torch.float)
        all_test_dim_embeds = torch.tensor([f.dimension_embed for f in data], dtype=torch.float)
        all_test_labels = torch.tensor([f.label for f in data], dtype=torch.long)
        all_test_dim_names = [f.dimension_name for f in data]

        eval_data = TensorDataset(
            all_test_input_1, all_test_input_2, all_test_glove_1, all_test_glove_2, all_test_dim_embeds, all_test_labels
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        self.model.eval()
        all_preds = []
        all_labels = []
        for input_1s, input_2s, glove_1s, glove_2s, dim_embeds, labels in tqdm(eval_dataloader, desc="Evaluating"):
            input_1s = input_1s.to(self.device)
            input_2s = input_2s.to(self.device)
            glove_1s = glove_1s.to(self.device)
            glove_2s = glove_2s.to(self.device)
            dim_embeds = dim_embeds.to(self.device)
            labels = labels.to(self.device).detach().cpu().numpy()

            with torch.no_grad():
                logits = self.model(input_1s, input_2s, glove_1s, glove_2s, dim_embeds, None).detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)

            all_preds += list(preds)
            all_labels += list(labels)

        assert len(all_preds) == len(all_labels)
        assert len(all_preds) == len(all_test_dim_names)
        correct = 0.0
        size_correct = 0.0
        weight_correct = 0.0
        for i, p in enumerate(all_preds):
            if all_labels[i] == p:
                correct += 1.0
                if all_test_dim_names[i] == "size":
                    size_correct += 1.0
                if all_test_dim_names[i] == "weight":
                    weight_correct += 1.0

        print("Acc.: " + str(correct / float(len(all_preds))))
        # print("Size Acc.: " + str(size_correct / float(all_test_dim_names.count("size"))))
        # print("Weight Acc.: " + str(weight_correct / float(all_test_dim_names.count("weight"))))


if __name__ == "__main__":
    runner = Runner(
        # train_data="samples/verbphysics/train-5/train.csv",
        # dev_data="samples/verbphysics/train-5/dev.csv",
        # test_data="samples/verbphysics/train-5/test.csv",
        train_data="samples/vp_clean/reannotations/train.csv",
        dev_data="samples/vp_clean/reannotations/dev.csv",
        test_data="samples/vp_clean/reannotations/test.csv",
        embedding_file="samples/vp_clean/reannotations/obj_embedding_100v.pkl"
    )
    runner.train(epoch=10000, batch_size=16)
    runner.eval(runner.dev_data)
    runner.eval(runner.test_data)
