import math
from typing import List
from word2number import w2n

import torch
import pickle
import random
from pytorch_pretrained_bert import BertForPreTraining, BertTokenizer

import torch
import os
import numpy as np
from scipy.stats import norm
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForTemporalClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer


class Annotator:

    def __init__(self, bert_model):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))

        self.model = BertForTemporalClassification.from_pretrained(bert_model,
                                                                   cache_dir=cache_dir,
                                                                   num_labels=1).to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.max_seq_length = 128

    def get_cdf(self, upper, lower, mu, sigma, pi):
        mu = mu.detach().cpu().numpy()
        sigma = sigma.detach().cpu().numpy()
        pi = pi.detach().cpu().numpy()
        # For log models
        mu = np.exp(mu) / 290304000.0
        sigma = np.exp(sigma) / 290304000.0
        # end
        accu = 0.0
        for i, _ in enumerate(mu):
            n = norm(mu[i], sigma[i])
            accu += pi[i] * (n.cdf(upper) - n.cdf(lower))
        return accu

    def get_mean(self, mu, pi):
        mu = mu.detach().cpu().numpy()
        pi = pi.detach().cpu().numpy()
        acc = 0.0
        for i, _ in enumerate(mu):
            acc += pi[i] * mu[i]
        return acc

    def get_segment_boundaries(self):
        convert_map = [
            1.0,
            60.0 * 0.5,
            60.0 * 60.0 * 0.5,
            24.0 * 60.0 * 60.0 * 0.5,
            7.0 * 24.0 * 60.0 * 60.0 * 0.5,
            30.0 * 24.0 * 60.0 * 60.0 * 0.5,
            365.0 * 24.0 * 60.0 * 60.0 * 0.5,
            290304000.0,
        ]
        return convert_map

    def annotate(self, sentence, verb_pos, arg0_start, arg0_end, arg1_start, arg1_end, arg2_start, arg2_end):
        tokens = sentence.lower().split()
        for i, _ in enumerate(tokens):
            if tokens[i].lower() not in self.tokenizer.vocab:
                tokens[i] = "[UNK]"

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        arg0_start += 1
        arg0_end += 1
        arg1_start += 1
        arg1_end += 1
        segment_ids = [0] * len(tokens)
        subj_mask = [0] * len(tokens)
        obj_mask = [0] * len(tokens)
        arg2_mask = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        for i in range(arg0_start, arg0_end):
            subj_mask[i] = 1
        for i in range(arg1_start, arg1_end):
            obj_mask[i] = 1
            # subj_mask[i] = 1
        for i in range(arg2_start, arg2_end):
            arg2_mask[i] = 1
            # subj_mask[i] = 1
        subj_mask[verb_pos] = 1

        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        subj_mask += padding
        obj_mask += padding
        arg2_mask += padding

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(subj_mask) == self.max_seq_length
        assert len(obj_mask) == self.max_seq_length

        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.device)
        target_idx = torch.tensor([verb_pos], dtype=torch.long).to(self.device)
        subj_masks = torch.tensor([subj_mask], dtype=torch.long).to(self.device)
        obj_masks = torch.tensor([obj_mask], dtype=torch.long).to(self.device)
        arg2_masks = torch.tensor([arg2_mask], dtype=torch.long).to(self.device)

        with torch.no_grad():
            pi, mu, sigma = self.model(
                input_ids, segment_ids, input_mask, labels=None, target_idx=target_idx, subj_mask=subj_masks,
                obj_mask=obj_masks, arg3_mask=arg2_masks
            )

        ret_cdfs = []
        boundaries = self.get_segment_boundaries()
        for i in range(0, len(boundaries) - 1):
            ret_cdfs.append(self.get_cdf(boundaries[i + 1] / 290304000.0, boundaries[i] / 290304000.0, mu[0], sigma[0], pi[0]))

        return ret_cdfs


class BERT_LM_predictions:
    use_gpu = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

    max_batch_size = 250  # max number of instanes grouped together
    batch_max_length = 10  # max number of tokens in each instance

    bert_model = 'bert-base-uncased'

    def __init__(self):
        # Load pre-trained model tokenizer (vocabulary)
        self.duration_keys = [
            "second",
            "seconds",
            "minute",
            "minutes",
            "hour",
            "hours",
            "day",
            "days",
            "week",
            "weeks",
            "month",
            "months",
            "year",
            "years",
            "century",
            "centuries",
        ]
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        # Load pre-trained model (weights)
        self.model = BertForPreTraining.from_pretrained('bert-base-uncased')
        self.model.eval()

    def vectorize_maked_instance(self, tokenized_text1: List[str], tokenized_text2: List[str]):

        tokens = []
        segment_ids = []
        input_mask = []

        masked_indices = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        input_mask.append(1)

        for token in tokenized_text1:
            t = token.lower()
            if t not in self.tokenizer.vocab:
                t = "[UNK]"
            tokens.append(t)
            segment_ids.append(0)
            input_mask.append(1)

        tokens.append("[SEP]")
        segment_ids.append(0)
        input_mask.append(1)

        second_part_start_index = len(tokens)
        for token in tokenized_text2:
            if token == "@":
                masked_indices.append(len(tokens))
                tokens.append("[MASK]")
            else:
                tokens.append(token)
            segment_ids.append(1)
            input_mask.append(1)

        tokens.append("[SEP]")
        segment_ids.append(1)
        input_mask.append(1)
        token_length = len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return input_ids, segment_ids, input_mask, token_length, second_part_start_index, masked_indices, tokens

    def calculate_bert_masked_per_token(self, tokens, k=16):
        input_ids, segment_ids, input_mask, token_length, second_part_start_index, masked_indices, tokens = self.vectorize_maked_instance(
            tokens, [])

        input_tensor = torch.tensor([input_ids])
        segment_tensor = torch.tensor([segment_ids])
        mask_tensor = torch.tensor([input_mask])

        # Predict all tokens
        predictions, _ = self.model(input_tensor, segment_tensor, mask_tensor)

        predictedTokens = {}

        import numpy as np

        masked_indices = list(np.arange(0, token_length))

        target_indices = [2117, 3823, 3371, 2781, 3178, 2847, 2154, 2420, 2733, 3134, 3204, 2706, 2095, 2086, 2301, 4693]
        target_indices = torch.LongTensor(target_indices)

        # calculating predictions
        for ind in masked_indices:
            top_scores, top_indices = torch.topk(torch.index_select(predictions[0, ind], 0, target_indices), k)
            top_scores = top_scores.cpu().tolist()
            top_indices = top_indices.cpu().tolist()
            # predictedTokens[ind] = [(self.tokenizer.convert_ids_to_tokens([id])[0], normlalize(s)) for id, s in
            #                         zip(top_indices, top_scores)]
            predictedTokens[ind] = [(self.duration_keys[id], normlalize(s)) for id, s in
                                    zip(top_indices, top_scores)]

        return predictedTokens, tokens


def normlalize(number):
    return math.floor(number * 1000) / 1000.0


class DataReader:

    def __init__(self, file_name, normalize=False):
        self.file_name = file_name
        self.lines = []
        self.duration_keys = [
            "second",
            "seconds",
            "minute",
            "minutes",
            "hour",
            "hours",
            "day",
            "days",
            "week",
            "weeks",
            "month",
            "months",
            "year",
            "years",
            "century",
            "centuries",
        ]
        self.convert_map = {
            "seconds": 1.0,
            "minutes": 60.0,
            "hours": 60.0 * 60.0,
            "days": 24.0 * 60.0 * 60.0,
            "weeks": 7.0 * 24.0 * 60.0 * 60.0,
            "months": 30.0 * 24.0 * 60.0 * 60.0,
            "years": 365.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
        }
        self.normalize = normalize
        self.load()

    def get_trivial_floats(self, s):
        try:
            if s == "a" or s == "an":
                return 1.0
            n = float(s)
            return n
        except:
            return None

    def quantity(self, s):
        try:
            if self.get_trivial_floats(s) is not None:
                return self.get_trivial_floats(s)
            cur = w2n.word_to_num(s)
            if cur is not None:
                return float(cur)
            return None
        except:
            return None

    def load(self):
        with open(self.file_name, "r") as f:
            self.lines = f.readlines()
        self.lines = [x.strip() for x in self.lines]

    def plural_units(self, u):
        m = {
            "second": "seconds",
            "minute": "minutes",
            "hour": "hours",
            "day": "days",
            "week": "weeks",
            "month": "months",
            "year": "years",
            "century": "centuries",
        }
        if u in m:
            return m[u]
        return u

    def normalize_timex(self, v_input, u):
        seconds = self.convert_map[self.plural_units(u)] * float(v_input)
        prev_unit = "seconds"
        for i, v in enumerate(self.convert_map):
            if seconds / self.convert_map[v] < 0.5:
                break
            prev_unit = v
        if prev_unit == "seconds" and seconds > 60.0:
            prev_unit = "centuries"
        new_val = int(seconds / self.convert_map[prev_unit])

        return prev_unit, str(new_val)

    def get(self):
        ret = []
        for line in self.lines:
            sentence = line.split("\t")[0]
            tokens = sentence.split()
            for i, t in enumerate(tokens):
                if t.lower() in self.duration_keys and i != 0 and self.quantity(tokens[i - 1]) is not None:
                    if t.lower() == "century" or t.lower() == "centuries":
                        continue
                    if random.random() < 0.01:
                        ret.append((tokens, i))
        return ret

    def get_formatted(self):
        ret = []
        for line in self.lines:
            if "century" in line.split("\t")[2].lower() or "centuries" in line.split("\t")[2].lower():
                continue
            if random.random() < 1:
                ret.append((
                    line.split("\t")[0],
                    int(line.split("\t")[1]),
                    int(line.split("\t")[3]),
                    int(line.split("\t")[4]),
                    int(line.split("\t")[5]),
                    int(line.split("\t")[6]),
                    int(line.split("\t")[7]),
                    int(line.split("\t")[8]),
                    line.split("\t")[2],
                    line
                ))
        return ret


class UDReader:

    def __init__(self, path):
        self.path = path
        self.sentences = {}
        self.data = {}
        self.load()
        self.load_annotation()

    def load_raw_file(self, file_name):
        lines = [x.strip() for x in open(file_name).readlines()]
        sentences = []
        cur_sentence = []
        for line in lines:
            if line.startswith("#"):
                continue
            elif line == "":
                sentences.append(cur_sentence)
                cur_sentence = []
            else:
                token = line.split("\t")[1]
                cur_sentence.append(token)
        if len(cur_sentence) > 0:
            sentences.append(cur_sentence)
        return sentences

    def load(self):
        for key in ["train", "test", "dev"]:
            file_name = self.path + "/" + key + ".conll"
            self.sentences[key] = self.load_raw_file(file_name)

    def load_annotation(self):
        file_name = self.path + "/duration_annotation.tsv"
        lines = [x.strip() for x in open(file_name).readlines()][1:]
        for line in lines:
            groups = line.split("\t")
            key_name = groups[0].lower()
            # print(key_name)
            sentence_index_1 = int(groups[2].split()[1]) - 1
            # print(sentence_index_1)
            token_index_1 = int(groups[4])
            label_1 = int(groups[14])
            sentence_index_2 = int(groups[6].split()[1]) - 1
            # print(sentence_index_2)
            token_index_2 = int(groups[8])
            label_2 = int(groups[15])
            if key_name not in self.data:
                self.data[key_name] = []
            self.data[key_name].append([self.sentences[key_name][sentence_index_1], token_index_1, label_1])
            self.data[key_name].append([self.sentences[key_name][sentence_index_2], token_index_2, label_2])

        for key in self.data:
            for d in self.data[key]:
                assert d[1] < len(d[0])

    def save_to_file(self, output_path):
        unit_list = ["instantaneous", "seconds", "minutes", "hours", "days", "weeks", "months", "years", "decades", "centuries", "forever"]
        for key in self.data:
            cur_path = output_path + "/" + key + ".formatted.txt"
            f = open(cur_path, "w")
            for d in self.data[key]:
                label = d[2]
                """
                Warning: Skipping labels, not a fair comparison
                """
                # if label in [0, 8, 9, 10]:
                #    continue
                f.write(" ".join(d[0]) + "\t" + str(d[1]) + "\t1 " + unit_list[label] + "\t-1\t-1\t-1\t-1\t-1\t-1\n")


class Evaluator:

    def __init__(self, file_name):
        with open(file_name, "rb") as f:
            self.results = pickle.load(f)
        self.true_mapping = {
            "second": "seconds",
            "minute": "minutes",
            "hour": "hours",
            "day": "days",
            "week": "weeks",
            "month": "months",
            "year": "years",
            "century": "centuries",
        }
        self.duration_val = {
            "seconds": 0,
            "minutes": 1,
            "hours": 2,
            "days": 3,
            "weeks": 4,
            "months": 5,
            "years": 6,
            "centuries": 7,
        }
        self.duration_val_full = {
            "instantaneous": 0,
            "seconds": 1,
            "minutes": 2,
            "hours": 3,
            "days": 4,
            "weeks": 5,
            "months": 6,
            "years": 7,
            "decades": 8,
            "centuries": 9,
            "forever": 10,
        }

    def get_true_name(self, s):
        s = s.lower()
        if s not in self.true_mapping:
            return s
        return self.true_mapping[s]

    def print_results(self):
        hard_correct = 0.0
        mpr = 0.0
        mr = 0.0
        mrr = 0.0
        td = 0.0
        for _, gold, prediction, ps in self.results:
            prediction = self.get_true_name(prediction)
            gold = self.get_true_name(gold)
            if self.duration_val[gold] == self.duration_val[prediction]:
                hard_correct += 1.0
            td += float(abs(self.duration_val[gold] - self.duration_val[prediction]))
            prob_sum = 0.0
            unit_map = {}
            for p, s in ps:
                prob_sum += math.exp(s)
                if self.get_true_name(p) not in unit_map:
                    unit_map[self.get_true_name(p)] = 0.0
                unit_map[self.get_true_name(p)] += math.exp(s)
            unit_map = sorted(unit_map.items(), key=lambda x: x[1], reverse=True)
            for i, (p, s) in enumerate(ps):
                if self.get_true_name(p) == gold:
                    mpr += math.exp(s) / prob_sum
            for i, key in enumerate(unit_map):
                if key[0] == gold:
                    mrr += 1 / (float(i) + 1)
                    mr += float(i)

        print("Accuracy: " + str(hard_correct / float(len(self.results))))
        print("Prob Rank: " + str(mpr / float(len(self.results))))
        print("Mean Rank: " + str(mr / float(len(self.results))))
        print("Reciprocal Rank: " + str(mrr / float(len(self.results))))
        print("Mean Distance: " + str(td / float(len(self.results))))


def train_and_eval_bert():
    BLM = BERT_LM_predictions()
    data = DataReader("samples/duration/verb_formatted_all_svo_better_filter_non_mask.txt")
    # data = DataReader("samples/UD_English/test/formatted.txt")
    results = []
    count = 0
    for tokens, pos in data.get():
        gold = tokens[pos].lower()
        val = data.quantity(tokens[pos - 1])
        # gold, new_num = data.normalize_timex(val, gold)
        tokens[pos] = "[MASK]"
        tokens[pos - 1] = "one"
        # new_tokens = []
        # for i in range(0, len(tokens)):
            # if i != pos - 1:
            #     new_tokens.append(tokens[i])
        # pos = pos - 1
        # tokens = new_tokens
        ps = BLM.calculate_bert_masked_per_token(tokens)[0][pos]
        selected_p = '404'
        for p in ps:
            predicted_token = p[0]
            if predicted_token in data.duration_keys:
                selected_p = predicted_token
                break
        results.append([tokens, gold, selected_p, ps])
        count += 1
        if count % 1000 == 0:
            print(count)
    pickle.dump(results, open("unit_exp_output.pkl", "wb"))

    evaluator = Evaluator("unit_exp_output.pkl")
    evaluator.print_results()


def train_and_eval_bert_on_udst():
    BLM = BERT_LM_predictions()
    data = DataReader("samples/UD_English/test.formatted.txt")
    results = []
    count = 0
    for sentence, verb_pos, _, _, _, _, _, _, label, _ in data.get_formatted():
        gold = label.split(" ")[1]
        if gold in ["instantaneous", "decades", "centuries", "forever"]:
            continue
        tokens = sentence.split()
        new_tokens = []
        for t in tokens[:-1]:
            new_tokens.append(t)
        new_tokens.append("for")
        new_tokens.append("one")
        new_tokens.append("[MASK]")
        new_tokens.append(tokens[-1])
        tokens = new_tokens
        ps = BLM.calculate_bert_masked_per_token(tokens)[0][len(new_tokens) - 2]
        selected_p = '404'
        for p in ps:
            predicted_token = p[0]
            if predicted_token in data.duration_keys:
                selected_p = predicted_token
                break
        results.append([tokens, gold, selected_p, ps])
        count += 1
        if count % 1000 == 0:
            print(count)
    pickle.dump(results, open("unit_exp_output.pkl", "wb"))

    evaluator = Evaluator("unit_exp_output.pkl")
    evaluator.print_results()


def train_and_eval():
    annotator = Annotator("models/models_log_44")
    data = DataReader("samples/duration/verb_formatted_all_svo_better_filter_4.txt")
    # data = DataReader("samples/UD_English/test.formatted.txt")
    evaluator = Evaluator("unit_exp_output.pkl")
    f_out = open("filtered_results.txt", "w")
    duration_keys_ordered = [
        "seconds",
        "minutes",
        "hours",
        "days",
        "weeks",
        "months",
        "years",
    ]
    mr_map = {
        "seconds": 0,
        "minutes": 0,
        "hours": 0,
        "days": 0,
        "weeks": 0,
        "months": 0,
        "years": 0,
    }
    id_map = {
        "seconds": 0,
        "minutes": 1,
        "hours": 2,
        "days": 3,
        "weeks": 4,
        "months": 5,
        "years": 6,
    }
    mpr_map = {
        "seconds": 0.0,
        "minutes": 0.0,
        "hours": 0.0,
        "days": 0.0,
        "weeks": 0.0,
        "months": 0.0,
        "years": 0.0,
    }
    mr_count = {
        "seconds": 0,
        "minutes": 0,
        "hours": 0,
        "days": 0,
        "weeks": 0,
        "months": 0,
        "years": 0,
    }
    hard_correct = 0.0
    mpr = 0.0
    mr = 0.0
    mrr = 0.0
    td = 0.0
    sampled_data = data.get_formatted()
    counter = 0
    invalid_counter = 0
    for (a, b, c, d, e, f, g, h, timex, line) in sampled_data:
        if len(a.split()) > 100:
            continue
        rets = annotator.annotate(a, b, c, d, e, f, g, h)

        val = data.quantity(timex.split(" ")[0])
        gold = timex.split(" ")[1]
        gold, new_num = data.normalize_timex(val, gold)
        gold = evaluator.get_true_name(gold)
        if gold == "centuries":
            continue

        label_id = id_map[gold]
        below_sum = 0.0
        for i in range(0, label_id):
            below_sum += rets[i]
        tokens = a.split()
        tokens[b] = "[" + tokens[b] + "]"
        # if rets[label_id] < 0.5:
        if below_sum > 0.3 and rets[label_id] < 0.3:
            invalid_counter += 1
        else:
            f_out.write(line + "\n")
        # print(" ".join(tokens))
        # print(timex)
        # print(below_sum)
        # print(rets[label_id])
        # print("================")

        unit_prob_map = {}
        for i, key in enumerate(duration_keys_ordered):
            unit_prob_map[key] = rets[i]
        ksum = 0.0
        for key in unit_prob_map:
            ksum += unit_prob_map[key]
        for key in unit_prob_map:
            unit_prob_map[key] = unit_prob_map[key] / ksum
        unit_map = sorted(unit_prob_map.items(), key=lambda x: x[1], reverse=True)
        for i, (p, s) in enumerate(unit_map):
            if i == 0 and p == gold:
                hard_correct += 1.0
            if i == 0:
                top_val = evaluator.duration_val[p]
                true_val = evaluator.duration_val[gold]
                td += float(abs(top_val - true_val))
            if p == gold:
                mpr += s
                mrr += 1 / (float(i) + 1)
                mr += float(i)
                mr_map[gold] += float(i)
                mpr_map[gold] += s
                mr_count[gold] += 1
        counter += 1
        if counter % 100 == 0:
            print(counter)
    print("Accuracy: " + str(hard_correct / float(len(sampled_data))))
    print("Prob Rank: " + str(mpr / float(len(sampled_data))))
    print("Mean Rank: " + str(mr / float(len(sampled_data))))
    print("Reciprocal Rank: " + str(mrr / float(len(sampled_data))))
    print("Mean Distance: " + str(td / float(len(sampled_data))))

    for key in mr_count:
        print(key + ": " + str(float(mr_map[key] / float(mr_count[key]))))

    print()
    for key in mr_count:
        print(key + ": " + str(float(mpr_map[key] / float(mr_count[key]))))


def get_optimal_prediction(evaluator, prob_map):
    optimal_val = 10000.0
    optimal_ret = ""
    for key in evaluator.duration_val:
        cur_distance = 0.0
        for tk in prob_map:
            cur_distance += prob_map[tk] * float(abs(evaluator.duration_val[tk] - evaluator.duration_val[key]))
        if cur_distance < optimal_val:
            optimal_val = cur_distance
            optimal_ret = key
    return optimal_ret

def eval_bert_custom():
    pos_to_label = {
        0: "instantaneous",
        1: "seconds",
        2: "minutes",
        3: "hours",
        4: "days",
        5: "weeks",
        6: "months",
        7: "years",
        8: "decades",
        9: "centuries",
        10: "forever",
    }
    evaluator = Evaluator("unit_exp_output.pkl")
    duration_keys_ordered = [
        # "instantaneous",
        "seconds",
        "minutes",
        "hours",
        "days",
        "weeks",
        "months",
        "years",
        # "decades",
        # "centuries",
        # "forever",
    ]
    hard_coded_prediction_a = {
        "seconds": 0.01016,
        "minutes": 0.05843,
        "hours": 0.05700,
        "days": 0.06956,
        "weeks": 0.08648,
        "months": 0.1471,
        "years": 0.5712,
    }
    hard_coded_prediction_b = {
        "seconds": 0.08469,
        "minutes": 0.31640,
        "hours": 0.11630,
        "days": 0.12884,
        "weeks": 0.08407,
        "months": 0.13334,
        "years": 0.133633,
    }
    mr_map = {
        "instantaneous": 0,
        "seconds": 0,
        "minutes": 0,
        "hours": 0,
        "days": 0,
        "weeks": 0,
        "months": 0,
        "years": 0,
        "decades": 0,
        "centuries": 0,
        "forever": 0,
    }
    mpr_map = {
        "instantaneous": 0,
        "seconds": 0.0,
        "minutes": 0.0,
        "hours": 0.0,
        "days": 0.0,
        "weeks": 0.0,
        "months": 0.0,
        "years": 0.0,
        "decades": 0,
        "centuries": 0,
        "forever": 0,
    }
    mr_count = {
        "instantaneous": 0,
        "seconds": 0,
        "minutes": 0,
        "hours": 0,
        "days": 0,
        "weeks": 0,
        "months": 0,
        "years": 0,
        "decades": 0,
        "centuries": 0,
        "forever": 0,
    }
    hard_correct = 0.0
    mpr = 0.0
    mr = 0.0
    mrr = 0.0
    td = 0.0
    counter = 0
    prediction_lines = [x.strip() for x in open("bert_logits.txt").readlines()]

    lines = [x.strip() for x in open("samples/UD_English/test.formatted.txt").readlines()]
    reader = DataReader("temp.txt")
    new_lines = []
    for l in lines:
        if len(l.split("\t")[0].split()) > 120:
            continue
        gold = l.split("\t")[2].split()[1].lower()
        val = float(l.split("\t")[2].split()[0])
        if val < 1.0:
            continue
        # gold, _ = reader.normalize_timex(val, gold)
        if gold in ["instantaneous", "decades", "centuries", "forever"]:
            continue
        new_lines.append(l)

    for i, line in enumerate(new_lines):
        gold = reader.normalize_timex(float(line.split("\t")[2].split()[0]), line.split("\t")[2].split()[1])[0]
        # gold = line.split("\t")[2].split()[1]
        unit_prob_map = {}
        for ii, key in enumerate(duration_keys_ordered):
            if key in ["instantaneous", "decades", "centuries", "forever"]:
                pass
                # continue
            unit_prob_map[key] = prediction_lines[i].split("\t")[ii]
        ksum = 0.0
        for key in unit_prob_map:
            if float(unit_prob_map[key]) > 50:
                unit_prob_map[key] = 50.0
            ksum += math.exp(float(unit_prob_map[key]))
        for key in unit_prob_map:
            unit_prob_map[key] = math.exp(float(unit_prob_map[key])) / ksum
            pass
        # unit_prob_map = hard_coded_prediction_b

        a_sum = 0.0
        for k in unit_prob_map:
            a_sum += unit_prob_map[k]
        assert 0.99 < a_sum < 1.01
        unit_map = sorted(unit_prob_map.items(), key=lambda x: x[1], reverse=True)
        top_prediction = get_optimal_prediction(evaluator, unit_prob_map)
        for i, (p, s) in enumerate(unit_map):
            if i == 0 and p == gold:
                hard_correct += 1.0
            # if i == 0:
            if p == top_prediction:
                top_val = evaluator.duration_val[p]
                true_val = evaluator.duration_val[gold]
                td += float(abs(top_val - true_val))
            if p == gold:
                mpr += s
                mrr += 1 / (float(i) + 1)
                mr += float(i)
                mr_map[gold] += float(i)
                mpr_map[gold] += s
                mr_count[gold] += 1
        counter += 1
    print("Accuracy: " + str(hard_correct / float(len(new_lines))))
    print("Prob Rank: " + str(mpr / float(len(new_lines))))
    print("Mean Rank: " + str(mr / float(len(new_lines))))
    print("Reciprocal Rank: " + str(mrr / float(len(new_lines))))
    print("Mean Distance: " + str(td / float(len(new_lines))))

    for key in mr_count:
        print(key + ": " + str(float(mr_map[key] / float(mr_count[key]))))

    print()
    for key in mr_count:
        print(key + ": " + str(float(mpr_map[key] / float(mr_count[key]))))


def convert_prob_file(input_file, reference_file, output_file):
    prob_lines = [x.strip() for x in open(input_file).readlines()]
    ref_lines = [x.strip() for x in open(reference_file).readlines()]
    ref_lines_new = []
    for l in ref_lines:
        if len(l.split("\t")[0].split()) > 120:
            continue
        gold = l.split("\t")[2].split()[1].lower()
        val = float(l.split("\t")[2].split()[0])
        if val < 1.0:
            continue
        if gold in ["instantaneous", "decades", "centuries", "forever"]:
            continue
        ref_lines_new.append(l)
    ref_lines = ref_lines_new
    assert(len(ref_lines) == len(prob_lines))
    f = open(output_file, "w")
    for i in range(0, len(prob_lines)):
        tokens = ref_lines[i].split("\t")[0].split()
        pos = int(ref_lines[i].split("\t")[1])
        tokens[pos] = "[" + tokens[pos] + "]"
        probs = [math.exp(float(x)) for x in prob_lines[i].split("\t")]
        sum = 0.0
        for p in probs:
            sum += p
        probs = [x / sum for x in probs]
        f.write(" ".join(tokens) + "\t" + "\t".join([str(x) for x in probs]) + "\n")


def get_max_results(file_name):
    evaluator = Evaluator("unit_output.pkl")
    lines = [x.strip() for x in open(file_name).readlines()]
    label_map = {}
    counter = 0.0
    distance = 0.0
    total = 0.0
    correct = 0.0
    for line in lines:
        key = "\t".join(line.split("\t")[:2])
        if key not in label_map:
            label_map[key] = []
        label_map[key].append(line.split("\t")[2].split()[1])
    f_out = open("samples/UD_English/train.filtered.formatted.txt", "w")
    for key in label_map:
        predictions = label_map[key]
        avg_dist = 0.0
        avg_counter = 0.0
        for i in range(0, len(predictions)):
            for j in range(i + 1, len(predictions)):
                avg_dist += abs(evaluator.duration_val_full[predictions[i]] - evaluator.duration_val_full[predictions[j]])
                avg_counter += 1.0
        if avg_counter > 0 and (avg_dist / avg_counter) > 2:
            pass
            # continue
        if "instantaneous" in predictions or "decades" in predictions or "centuries" in predictions or "forever" in predictions:
            pass
            # continue
        # prediction = random.choice(predictions)
        d_min = 100
        acc_max = 0.0
        acc_cur_total = 0.0
        acc_cur_correct = 0.0
        for prediction in evaluator.duration_val_full.keys():
            d_current = 0.0
            acc_correct = 0.0
            acc_total = 0.0
            for gold in predictions:
                d_current += abs(evaluator.duration_val_full[gold] - evaluator.duration_val_full[prediction])
            if d_current < d_min:
                d_min = d_current
            for gold in predictions:
                if gold == prediction:
                    acc_correct += 1.0
                acc_total += 1.0
            cur_acc = acc_correct / acc_total
            if cur_acc > acc_max:
                acc_max = cur_acc
                acc_cur_correct = acc_correct
                acc_cur_total = acc_total
        distance += float(d_min)
        counter += float(len(predictions))
        total += acc_cur_total
        correct += acc_cur_correct
        # for p in predictions:
        #     f_out.write(key + "\t" + "1 " + p + "\t-1\t-1\t-1\t-1\t-1\t-1\n")

    print(counter)
    print(correct / total)
    return float(distance) / counter

def eval_bert_pair_acc():
    prediction_lines = [x.strip() for x in open("bert_logits.txt").readlines()]

    lines = [x.strip() for x in open("samples/comparative/test.formatted.txt").readlines()]
    total = 0
    correct = 0
    for i, line in enumerate(lines):
        if i >= len(prediction_lines):
            break
        prediction = "LESS"
        scores = [float(x) for x in prediction_lines[i].split("\t")[-2:]]
        if scores[1] > scores[0]:
            prediction = "MORE"

        label = line.split("\t")[-1]
        if label == "NONE":
            continue
        total += 1
        if prediction == label:
            correct += 1
            # print(line)
            # print(scores)
            # print()
        else:
            pass
            # print(line)
            # print(scores)
            # print()
    print("Accuracy: " + str(float(correct) / float(total)))


# eval_bert_pair_acc()
# train_and_eval()
# reader = UDReader("samples/UD_English")
# reader.save_to_file("samples/UD_English_untouched")
eval_bert_custom()
# train_and_eval_bert_on_udst()
# convert_prob_file("bert_logits.txt", "samples/UD_English/test.formatted.txt", "bert_probs_noweight.txt")
s = 0.0
for i in range(1):
    s += get_max_results("samples/UD_English/test.formatted.txt")
print(s)
