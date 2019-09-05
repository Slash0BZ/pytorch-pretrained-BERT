import re
import os
import jsonlines
import pickle
import random
import json
import numpy as np
from word2number import w2n
from ccg_nlpy import local_pipeline
from math import floor
from scipy.stats import norm
import matplotlib.pyplot as plt
from allennlp import predictors
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive


class GigawordDocument:

    def __init__(self, _id, _type, _headline, _dateline):
        self.id = _id
        self.type = _type
        self.headline = _headline
        self.dateline = _dateline
        self.paragraphs = []

    def get_content(self):
        return ' '.join(self.paragraphs)


class GigawordExtractor:

    def __init__(self):
        self.pipeline = local_pipeline.LocalPipeline()
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

    @staticmethod
    def read_file(file_name):
        with open(file_name, encoding="ISO-8859-1") as f:
            lines = f.readlines()
        content = ' '.join(line.strip() for line in lines)
        documents = re.findall(r'(<DOC.*?>.+?</DOC>)', content)
        ret = []
        for document in documents:
            _id = re.findall(r'<DOC id=\"(.+?)\"', document)[0]
            _type = re.findall(r'<DOC.*?type=\"(.+?)\"', document)[0]
            _headline = re.findall(r'<HEADLINE>(.+?)</HEADLINE>', document)
            if len(_headline) <= 0:
                _headline = ""
            else:
                _headline = _headline[0].strip()
            _dateline = re.findall(r'<DATELINE>(.+?)</DATELINE>', document)
            if len(_dateline) <= 0:
                _dateline = ""
            else:
                _dateline = _dateline[0].strip()
            doc = GigawordDocument(_id, _type, _headline, _dateline)
            if '<P>' in document:
                doc.paragraphs = re.findall(r'<P>(.+?)</P>', document)
            else:
                doc.paragraphs = re.findall(r'<TEXT>(.+?)</TEXT>', document)
            doc.paragraphs = [x.strip() for x in doc.paragraphs]
            ret.append(doc)
        return ret


    @staticmethod
    def get_trivial_floats(s):
        try:
            if s == "a" or s == "an":
                return 1.0
            n = float(s)
            return n
        except:
            return None

    @staticmethod
    def quantity(s):
        try:
            if GigawordExtractor.get_trivial_floats(s) is not None:
                return GigawordExtractor.get_trivial_floats(s)
            cur = w2n.word_to_num(s)
            if cur is not None:
                return float(cur)
            return None
        except:
            return None

    def process_paragraph(self, p):
        ret = []
        ta = self.pipeline.doc(p)
        sent_view = ta.get_view("SENTENCE")
        for sent in sent_view:
            tokens_lower = sent['tokens'].lower().split()
            for (i, v) in enumerate(tokens_lower):
                if v in self.duration_keys and i > 0:
                    if GigawordExtractor.quantity(tokens_lower[i - 1]) is not None:
                        ret.append(sent['tokens'])
                        break
        return ret

    def tokenize(self, s):
        ta = self.pipeline.doc(s)
        sent_view = ta.get_view("SENTENCE")
        tokens = []
        for sent in sent_view:
            tokens += sent['tokens'].split()
        return tokens

    def process_path(self, path, duration_path=None):
        f_out = None
        if duration_path is not None:
            f_out = open(duration_path, "w")
        for root, dirs, files in os.walk(path):
            files.sort()
            for file in files:
                docs = GigawordExtractor.read_file(path + '/' + file)
                for doc in docs:
                    if doc.type != "story":
                        continue
                    for p in doc.paragraphs:
                        try:
                            cur_list = self.process_paragraph(p)
                            if f_out is not None:
                                for c in cur_list:
                                    f_out.write(c + "\n")
                        except Exception as e:
                            print(e)

    def validate_argument(self, argtmp_tokens):
        filter_list = [
            "after",
            "before",
            "ago",
            "earlier",
            "first",
            "during",
            "times",
            "within",
            "once",
            "twice",
            "early",
            "later",
            "late",
        ]

        filter_phrase_list = [
            "ahead of",
            "in a row",
            "prior to",
            "about the"
        ]

        first_list = [
            "with",
            "at",
            "when",
        ]

        equal_list = [
            "one day",
        ]

        for t in argtmp_tokens:
            if t.lower() in filter_list:
                return False
        if " ".join(argtmp_tokens).lower() in equal_list:
            return False
        if argtmp_tokens[0].lower() in first_list:
            return False
        concat = " ".join(argtmp_tokens).lower()
        for phrase in filter_phrase_list:
            if phrase in concat:
                return False
        quantity_count = 0
        for t in argtmp_tokens:
            if GigawordExtractor.quantity(t.lower()) is not None:
                quantity_count += 1
        if quantity_count > 1:
            return False
        duration_key_count = 0
        for t in argtmp_tokens:
            if t.lower() in self.duration_keys:
                duration_key_count += 1
        if duration_key_count > 1:
            return False
        return True

    def validate_sentence(self, sentence):
        invalid_next_tokens = [
            "ago",
            "before",
            "after",
            "later",
            "-",
            "old",
            "are",
            "is",
            "have",
            "has",
            "earlier",
            "every",
            "next",
        ]
        invalid_prev_tokens = [
            "after",
            "within",
            "every",

        ]
        invalid_prev_5_tokens = [
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
        ]
        invalid_next_two_tokens = [
            "from now",
        ]

        tokens = sentence.split()
        valid = True
        ever = False
        for (i, token) in enumerate(tokens):
            if token in self.duration_keys and i > 0:
                if GigawordExtractor.quantity(tokens[i - 1]) is not None:
                    ever = True
                else:
                    continue
                next_token_idx = min(len(tokens) - 1, i + 1)
                next_token = tokens[next_token_idx]
                if next_token.lower() in invalid_next_tokens:
                    valid = False

                prev_token_idx = max(0, i - 2)
                prev_token = tokens[prev_token_idx]
                if prev_token.lower() in invalid_prev_tokens:
                    valid = False

                prev_5_tokens = set()
                for j in range(i - 2, max(0, i - 7), -1):
                    prev_5_tokens.add(tokens[j])

                for t in prev_5_tokens:
                    if t in invalid_prev_5_tokens:
                        valid = False

                next_two_tokens = tokens[min(len(tokens) - 1, i + 1)] + " " + tokens[min(len(tokens) - 1, i + 2)]
                if next_two_tokens.lower() in invalid_next_two_tokens:
                    valid = False
        return valid and ever

    def process_duration_initial_filter(self, path, out_path):
        invalid_next_tokens = [
            "ago",
            "before",
            "after",
            "later",
            "-",
            "old",
            "are",
            "is",
            "have",
            "has",
            "earlier",
            "every",
            "next",
        ]
        invalid_prev_tokens = [
            "after",
            "within",

        ]
        invalid_prev_5_tokens = [
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
        ]
        invalid_next_two_tokens = [
            "from now",
        ]

        with open(path) as f:
            lines = [x.strip() for x in f.readlines()]

        seen = set()
        f_out = open(out_path, "w")

        for li, line in enumerate(lines):
            line = line.replace("-", " ")
            tokens = line.split()
            valid = True
            for (i, token) in enumerate(tokens):
                if token in self.duration_keys:
                    next_token_idx = min(len(tokens) - 1, i + 1)
                    next_token = tokens[next_token_idx]
                    if next_token.lower() in invalid_next_tokens:
                        valid = False

                    prev_token_idx = max(0, i - 2)
                    prev_token = tokens[prev_token_idx]
                    if prev_token.lower() in invalid_prev_tokens:
                        valid = False

                    prev_5_tokens = set()
                    for j in range(i - 2, max(0, i - 7), -1):
                        prev_5_tokens.add(tokens[j])

                    for t in prev_5_tokens:
                        if t in invalid_prev_5_tokens:
                            valid = False

                    next_two_tokens = tokens[min(len(tokens) - 1, i + 1)] + " " + tokens[min(len(tokens) - 1, i + 2)]
                    if next_two_tokens.lower() in invalid_next_two_tokens:
                        valid = False

            if valid and lines[li] not in seen:
                f_out.write(lines[li] + '\n')
                seen.add(lines[li])

    def conjunct_random_files(self):
        file_list = [
            "samples/duration/nyt_eng_filtered.txt",
            "samples/duration/afp_eng_filtered.txt",
            "samples/duration/apw_eng_filtered.txt",
        ]

        line_set = set()
        for f in file_list:
            f_in = open(f)
            for line in [x.strip() for x in f_in.readlines()]:
                tokens = line.split()
                target_count = 0
                for t in tokens:
                    if t.lower() in self.duration_keys:
                        target_count += 1
                if target_count == 1:
                    line_set.add(line)
        line_list = list(line_set)
        f_out = open("samples/duration/random_all_filtered.txt", "w")
        for line in line_list:
            f_out.write(line + "\n")

    def split_nominals(self):
        lines = [x.strip() for x in open("samples/duration_more/duration_all_filtered.txt").readlines()]
        random.shuffle(lines)
        f_nom = open("samples/duration_more/nominals.txt", "w")
        f_verb = open("samples/duration_more/verbs.txt", "w")
        for line in lines:
            tokens = line.split()
            for i, t in enumerate(tokens):
                if t.lower() in self.duration_keys:
                    next_token = tokens[min(len(tokens) - 1, i + 1)]
                    if next_token.lower() == "of":
                        f_nom.write(line + "\n")
                    else:
                        f_verb.write(line + "\n")

    def prepare_nom_data(self, nom_path, nom_out_path):
        lines = [x.strip() for x in open(nom_path).readlines()]
        f_out = open(nom_out_path, "w")
        for line in lines:
            tokens = line.split()
            set_blank_start = -1
            set_blank_end = -1
            valid = True
            form_validation = ""
            position = -1
            label_string = ""
            for i, t in enumerate(tokens):
                if t.lower() in self.duration_keys:
                    set_blank_start = i - 1
                    set_blank_end = i + 2
                    position = i + 2
                    form_validation = tokens[min(len(tokens) - 1, i + 2)]
                    label_string = str(self.quantity(tokens[i - 1])) + " " + t
                    # if tokens[max(0, i - 2)].lower() == "than" and tokens[max(0, i - 3)].lower() == "more":
                    #     position = position - 2
                    #     set_blank_start = i - 3
                    # if tokens[max(0, i - 2)].lower() == "the":
                    #     position = position - 1
                    #     set_blank_start = i - 2
                    prev_list = ["first", "last", "final"]
                    if tokens[max(0, i - 2)].lower() in prev_list:
                        valid = False
                    break
            for i in range(set_blank_start, set_blank_end):
                tokens[i] = "[MASK]"
            # new_tokens = []
            # for t in tokens:
            #     if t != "":
            #         new_tokens.append(t)
            # tokens = new_tokens

            if position >= len(tokens) or position < 0:
                continue

            if tokens[position] != form_validation:
                valid = False

            if valid:
                f_out.write(' '.join(tokens) + "\t" + str(position) + "\t" + label_string + "\n")

    def get_rid_of_masks(self, path, out_path):
        lines = [x.strip() for x in open(path).readlines()]
        f_out = open(out_path, "w")
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
        # Note: some are ideal values to ensure continuity
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
        for line in lines:
            tokens = line.split("\t")[0].split()
            mask_start = -1
            mask_end = -1
            for i, t in enumerate(tokens):
                if t == "[MASK]":
                    if mask_start == -1:
                        mask_start = i
                    if i + 1 > mask_end:
                        mask_end = i + 1
                    tokens[i] = ""
            target_idx = int(line.split("\t")[1])
            if target_idx >= mask_end:
                target_idx -= mask_end - mask_start

            subj_start = -1
            subj_end = -1
            obj_start = -1
            obj_end = -1
            arg2_start = -1
            arg2_end = -1

            if len(line.split("\t")) > 3:
                subj_start = int(line.split("\t")[3])
                subj_end = int(line.split("\t")[4])
                obj_start = int(line.split("\t")[5])
                obj_end = int(line.split("\t")[6])
                arg2_start = int(line.split("\t")[7])
                arg2_end = int(line.split("\t")[8])

            if subj_start >= mask_end:
                subj_start -= mask_end - mask_start
                subj_end -= mask_end - mask_start
            if obj_start >= mask_end:
                obj_start -= mask_end - mask_start
                obj_end -= mask_end - mask_start
            if arg2_start >= mask_end:
                arg2_start -= mask_end - mask_start
                arg2_end -= mask_end - mask_start

            label = line.split("\t")[2]
            label_num = float(label.split()[0])
            label_unit = label.split()[1].lower()
            label_value = label_num * value_map[label_unit]

            prev_b_value = 0
            selected_unit = ""
            for b_unit in ["second", "minute", "hour", "day", "week", "month", "year", "century"]:
                max_b_value = advance_map[b_unit] * value_map[b_unit]
                if max_b_value >= label_value >= prev_b_value:
                    selected_unit = b_unit
                    break
                prev_b_value = max_b_value
            if selected_unit == "":
                selected_unit = "century"

            label_num = round(label_value / value_map[selected_unit], 2)
            label = str(label_num) + " " + selected_unit
            new_tokens = []
            for t in tokens:
                if t != "":
                    new_tokens.append(t)
            f_out.write(" ".join(new_tokens) + "\t" + str(target_idx) + "\t" + label + "\t" +
                        str(subj_start) + "\t" + str(subj_end) + "\t" + str(obj_start) + "\t" + str(obj_end) +
                        "\t" + str(arg2_start) + "\t" + str(arg2_end) + "\n")


class PretrainedModel:
    """
    A pretrained model is determined by both an archive file
    (representing the trained model)
    and a choice of predictor.
    """
    def __init__(self, archive_file: str, predictor_name: str) -> None:
        self.archive_file = archive_file
        self.predictor_name = predictor_name

    def predictor(self) -> Predictor:
        archive = load_archive(self.archive_file)
        return Predictor.from_archive(archive, self.predictor_name)


class AllenSRL:

    def __init__(self):
        model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz',
                                'semantic-role-labeling')
        self.predictor = model.predictor()

    def predict_batch(self, sentences):
        for sentence in sentences:
            prediction = self.predictor.predict(sentence)
            print(prediction)

    def predict_file(self, path):
        with open(path) as f:
            lines = [x.strip() for x in f.readlines()]
        self.predict_batch(lines)


class SRLRunner:

    def __init__(self):
        self.extractor = GigawordExtractor()

    def parse_single_seq(self, tokens, tags):
        ret = {}
        cur_list = []
        cur_tag = "O"
        for i, t in enumerate(tags):
            if t == "O":
                if cur_tag != "O":
                    ret[cur_tag] = cur_list
                    cur_tag = "O"
                    cur_list = []
                continue
            if t.startswith("B"):
                if cur_tag != "O":
                    ret[cur_tag] = cur_list
                    cur_list = []
                cur_tag = "".join(t.split("-")[1:])
            cur_list.append(tokens[i])
        return ret

    def get_related_masks(self, tags):
        ret = [0] * len(tags)
        for i, t in enumerate(tags):
            if t.startswith("B") or t.startswith("I"):
                if "ARGM-TMP" not in t:
                    ret[i] = 1
        return ret

    def get_verb_position(self, tags):
        for i, t in enumerate(tags):
            if t == "B-V":
                return i
        return -1

    def get_subj_position(self, tags):
        start = -1
        end = -1
        for i, t in enumerate(tags):
            if t == "B-ARG0":
                start = i
                end = i + 1
            if t == "I-ARG0":
                end += 1
        return start, end

    def get_obj_position(self, tags):
        start = -1
        end = -1
        for i, t in enumerate(tags):
            if t == "B-ARG1":
                start = i
                end = i + 1
            if t == "I-ARG1":
                end += 1
        return start, end

    def get_arg3_position(self, tags):
        start = -1
        end = -1
        for i, t in enumerate(tags):
            if t == "B-ARG2":
                start = i
                end = i + 1
            if t == "I-ARG2":
                end += 1
        return start, end

    def get_tmp_range(self, tags):
        start = -1
        end = -1
        for i, t in enumerate(tags):
            if t == "B-ARGM-TMP":
                start = i
                end = i + 1
            if t == "I-ARGM-TMP":
                end += 1
        return start, end

    def has_negation(self, tags):
        for t in tags:
            if t == "B-ARGM-NEG":
                return True
        return False

    def parse_srl_file(self, path):
        lines = [x.strip() for x in open(path).readlines()]
        reader = jsonlines.Reader(lines)

        f_out_s = jsonlines.open("samples/duration/duration_srl_succeed.jsonl", mode="a")
        f_out_f = jsonlines.open("samples/duration/duration_srl_fail.jsonl", mode="a")
        for obj in reader:
            temporal_value = None
            parsed_select = {}
            for verbs_obj in obj['verbs']:
                tokens = obj['words']
                tags = verbs_obj['tags']
                parsed = self.parse_single_seq(tokens, tags)
                if "ARGMTMP" in parsed:
                    tmp_arg = parsed["ARGMTMP"]
                    for i, v in enumerate(tmp_arg):
                        if v.lower() in self.extractor.duration_keys:
                            q = self.extractor.quantity(tmp_arg[max(0, i - 1)])
                            if q is not None:
                                parsed_select = parsed
                                parsed_select["TAGS"] = tags
                                temporal_value = str(q) + " " + v.lower()

                parsed_select["TOKENS"] = tokens
                parsed_select["TMPVAL"] = temporal_value

            if temporal_value is not None:
                f_out_s.write(parsed_select)
            else:
                f_out_f.write(obj)

    def print_file(self, path):
        lines = [x.strip() for x in open(path).readlines()]
        reader = jsonlines.Reader(lines)

        for obj in reader:
            if "words" in obj:
                print(' '.join(obj["words"]))
            elif "TOKENS" in obj:
                print(' '.join(obj["TOKENS"]))
            else:
                print("ERR")

    def prepare_verb_file(self):
        lines = [x.strip() for x in open("samples/duration/duration_srl_succeed.jsonl").readlines()]
        reader = jsonlines.Reader(lines)

        f_out = open("samples/duration/verb_formatted_all_svo_better_filter_5.txt", "w")
        for obj in reader:
            tokens = obj["TOKENS"]
            sentence = ' '.join(tokens)
            if self.extractor.validate_sentence(sentence) and self.extractor.validate_argument(obj["ARGMTMP"]):
                tags = obj["TAGS"]
                verb_pos = self.get_verb_position(tags)
                start, end = self.get_tmp_range(tags)
                subj_start, subj_end = self.get_subj_position(tags)
                obj_start, obj_end = self.get_obj_position(tags)
                arg3_start, arg3_end = self.get_arg3_position(tags)
                if self.has_negation(tags):
                    continue
                if verb_pos == -1 or start == -1 or end == -1:
                    continue
                if subj_start == -1 and obj_start == -1 and arg3_start == -1:
                    continue
                for j in range(start, end):
                    pass
                    # tokens[j] = "[MASK]"
                f_out.write(' '.join(tokens) + "\t" + str(verb_pos) + "\t" + obj["TMPVAL"] + "\t"
                            + str(subj_start) + "\t" + str(subj_end) + "\t" + str(obj_start) + "\t" + str(obj_end) +
                            "\t" + str(arg3_start) + "\t" + str(arg3_end) + "\n")

    def prepare_simple_verb_file(self):
        lines = [x.strip() for x in open("samples/duration/duration_srl_succeed.jsonl").readlines()]
        reader = jsonlines.Reader(lines)

        f_out = open("samples/duration/verb_formatted_all_simple.txt", "w")
        for obj in reader:
            tokens = obj["TOKENS"]
            sentence = ' '.join(tokens)
            if self.extractor.validate_sentence(sentence) and self.extractor.validate_argument(obj["ARGMTMP"]):
                tags = obj["TAGS"]
                mask = self.get_related_masks(tags)
                verb_pos = self.get_verb_position(tags)
                start, end = self.get_tmp_range(tags)
                subj_start, subj_end = self.get_subj_position(tags)
                obj_start, obj_end = self.get_obj_position(tags)
                arg3_start, arg3_end = self.get_arg3_position(tags)
                if self.has_negation(tags):
                    continue
                if verb_pos == -1 or start == -1 or end == -1:
                    continue
                if subj_start == -1 and obj_start == -1 and arg3_start == -1:
                    continue
                new_tokens = []
                new_verb_pos = -1
                for i, t in enumerate(tokens):
                    if mask[i] > 0:
                        new_tokens.append(t)
                    if i == verb_pos:
                        new_verb_pos = len(new_tokens) - 1

                f_out.write(' '.join(new_tokens) + "\t" + str(new_verb_pos) + "\t" + obj["TMPVAL"] + "\t"
                            + str(subj_start) + "\t" + str(subj_end) + "\t" + str(obj_start) + "\t" + str(obj_end) +
                            "\t" + str(arg3_start) + "\t" + str(arg3_end) + "\n")

    def filter_short_verbs(self, path, out_path):
        lines = [x.strip() for x in open(path).readlines()]
        f_out = open(out_path, "w")

        # for line in lines:

    def prepare_verb_file_for_failures(self):
        lines = [x.strip() for x in open("samples/duration/duration_srl_fail.jsonl").readlines()]
        reader = jsonlines.Reader(lines)

        # f_out = open("samples/duration/verb_formatted_all_svo_from_fail.txt", "w")
        care_list = ['take', 'took', 'spend', 'spent']
        for obj in reader:
            tokens = obj["words"]
            verbs = obj["verbs"]
            chosen_range = [-1, -1]
            for verb in verbs:
                if verb['verb'].lower() in care_list:
                    start, end = self.get_obj_position(verb['tags'])
                    if start == -1:
                        continue
                    valid = False
                    for i in range(start + 1, end):
                        if (tokens[i].lower() in self.extractor.duration_keys) and (self.extractor.quantity(tokens[i - 1]) is not None):
                            valid = True
                            break
                    if valid:
                        v_start, v_end = self.get_subj_position(verb['tags'])
                        if v_start == -1:
                            continue
                        if tokens[v_start].lower() != "to":
                            continue
                        chosen_range[0] = v_start + 1
                        chosen_range[1] = v_start + 3
            if chosen_range[0] > -1:
                for verb in verbs:
                    verb_pos = self.get_verb_position(verb['tags'])
                    if chosen_range[1] > verb_pos >= chosen_range[0]:
                        if "description" in verb:
                            print(verb['description'])
                        else:
                            print(verb)

    def print_random_srl(self):
        # lines = [x.strip() for x in open("samples/duration/duration_srl_succeed.jsonl").readlines()]
        lines = [x.strip() for x in open("samples/duration/verb_formatted_all_svo_better_filter_0.txt").readlines()]
        random.shuffle(lines)
        printed = 0
        for line in lines:
            print(line)
            printed += 1
            if printed > 100:
                break

    def print_random_srl_json(self):
        lines = [x.strip() for x in open("samples/duration/duration_srl_fail.jsonl").readlines()]
        random.shuffle(lines)
        reader = jsonlines.Reader(lines)
        printed = 0
        for obj in reader:
            print(" ".join(obj["words"]))
            printed += 1
            if printed > 100:
                break

    def prepare_nyt_srl_file(self):
        main_dir = "/Users/xuanyuzhou/Downloads/nyt-srl"
        # f_out = jsonlines.open("samples/duration/verb_nyt_svo.jsonl", "w")

        token_count = 0
        doc_count = 0
        for root, dirs, files in os.walk(main_dir):
            for sub_dir in dirs:
                sub_path = main_dir + "/" + sub_dir + "/srl"
                for _, _, fs in os.walk(sub_path):
                    for f in fs:
                        file_path = sub_path + "/" + f
                        content = "".join([x.strip() for x in open(file_path).readlines()])
                        if len(content) < 1:
                            continue
                        doc_count += 1
                        j = json.loads(content)
                        for obj in j:
                            if len(obj) < 1:
                                continue
                            if "words" not in obj:
                                continue
                            token_count += len(obj["words"])
                        #     if self.extractor.validate_sentence(" ".join(obj["words"])):
                        #         f_out.write(obj)
                            # tokens = obj["TOKENS"]
                            # sentence = ' '.join(tokens)
                            # if self.extractor.validate_sentence(sentence):
                            #     tags = obj["TAGS"]
                            #     verb_pos = self.get_verb_position(tags)
                            #     start, end = self.get_tmp_range(tags)
                            #     subj_start, subj_end = self.get_subj_position(tags)
                            #     obj_start, obj_end = self.get_obj_position(tags)
                            #     if verb_pos == -1 or start == -1 or end == -1:
                            #         continue
                            #     for j in range(start, end):
                            #         tokens[j] = "[MASK]"
                            #     f_out.write(' '.join(tokens) + "\t" + str(verb_pos) + "\t" + obj["TMPVAL"] + "\t"
                            #                 + str(subj_start) + "\t" + str(subj_end) + "\t" + str(obj_start) + "\t" + str(obj_end) + "\n")
        print(token_count)
        print(doc_count)

    def tokenize_timebank_sentence(self, sentence):
        pause = False
        char_list = []
        entity_indices = []
        entity_tags = []
        cur_pause_tag = ""
        cur_inner_start = -1
        cur_inner_end = -1
        ret = []
        ret_time = []
        for i, c in enumerate(sentence):
            if c == "<":
                pause = True
                if cur_inner_start > -1 and cur_inner_end > -1:
                    entity_indices.append((cur_inner_start, cur_inner_end))
                    lower_bound = re.findall(r'lowerBoundDuration=\"(.+?)\"', cur_pause_tag)[0]
                    upper_bound = re.findall(r'upperBoundDuration=\"(.+?)\"', cur_pause_tag)[0]
                    entity_tags.append((lower_bound, upper_bound))
                cur_pause_tag = ""
                cur_inner_start = -1
                cur_inner_end = -1
                continue
            if c == ">":
                pause = False
                continue
            if pause:
                cur_pause_tag += c
                continue

            if c != " ":
                char_list.append(c)
            if cur_pause_tag.startswith("EVENT"):
                char_idx = len(char_list) - 1
                if cur_inner_start == -1:
                    cur_inner_start = char_idx
                if char_idx + 1 > cur_inner_end:
                    cur_inner_end = char_idx + 1

        tokens = self.extractor.tokenize(re.sub("<.*?>", " ", sentence))
        reverse_map = {}
        cur_index = -1
        for i, token in enumerate(tokens):
            for c in token:
                if c != " ":
                    cur_index += 1
                    reverse_map[cur_index] = i

        for i, (start, end) in enumerate(entity_indices):
            token_idx = reverse_map[start]
            ret.append(token_idx)
            ret_time.append(entity_tags[i])

        return tokens, ret, ret_time

    def prepare_timebank_file(self):
        path = "samples/timebank"
        f_out = open("samples/duration/timebank_formatted.txt", "w")
        for root, dirs, files in os.walk(path):
            files.sort()
            for file in files:
                content = " ".join([x.strip() for x in open(path + '/' + file).readlines()])
                sentences = re.findall(r'<s>(.+?)</s>', content)
                for sentence in sentences:
                    tokens, idxs, tags = self.tokenize_timebank_sentence(sentence)
                    for i, idx in enumerate(idxs):
                        output = ' '.join(tokens) + "\t" + str(idx) + "\t" + str(tags[i][0]) + "\t" + str(tags[i][1])
                        f_out.write(output + "\n")

    def get_label(self, exp):
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
        groups = exp.split()
        label_num = float(groups[0])
        if label_num < 1.0:
            return -1
        label_unit = groups[1].lower()
        label_idx = int(floor(label_num * 5.0 / advance_map[label_unit]))
        if label_idx > 4:
            label_idx = 4
        if label_idx < 0:
            label_idx = 0
        label_idx = unit_map[label_unit] * 5.0 + label_idx

        return label_idx

    def count_label(self, path):
        lines = [x.strip() for x in open(path).readlines()]

        count_map = {}
        for line in lines:
            label = self.get_label(line.split("\t")[2])
            if label == -1:
                continue
            if label not in count_map:
                count_map[label] = 0
            count_map[label] += 1
        for i in range(0, 40):
            if float(i) in count_map:
                print(str(count_map[float(i)]) + ", ")
            else:
                print(str(1) + ", ")

        return count_map

    def prepare_timebank_srl(self):
        source_lines = [x.strip() for x in open("samples/duration/timebank_filtered.txt").readlines()]
        reader = jsonlines.Reader([x.strip() for x in open("samples/timebank_srl.jsonl").readlines()])
        f_out = open("samples/duration/timebank_svo.txt", "w")
        objs = []
        for obj in reader:
            objs.append(obj)
        assert len(source_lines) == len(objs)
        for i, line in enumerate(source_lines):
            tokens = line.split("\t")[0].split()
            target_idx = int(line.split("\t")[1])
            verb_form = tokens[target_idx]
            obj = objs[i]

            new_tokens = obj["words"]
            wrote = False
            for verb in obj["verbs"]:
                if verb["verb"] == verb_form:
                    tags = verb["tags"]

                    valid = False
                    for j in range(target_idx - 2, target_idx + 3):
                        if 0 <= j < len(tags):
                            if tags[j] == "B-V":
                                valid = True
                    if valid:
                        verb_pos = self.get_verb_position(tags)
                        subj_start, subj_end = self.get_subj_position(tags)
                        obj_start, obj_end = self.get_obj_position(tags)
                        arg2_start, arg2_end = self.get_arg3_position(tags)
                        if verb_pos == -1:
                            continue
                        f_out.write(" ".join(new_tokens) + "\t" + str(verb_pos) + "\t" + line.split("\t")[2] + "\t" + line.split("\t")[3] + "\t" +
                                    str(subj_start) + "\t" + str(subj_end) + "\t" + str(obj_start) + "\t" + str(obj_end) +
                                    "\t" + str(arg2_start) + "\t" + str(arg2_end) + "\n")
                        wrote = True
                        break
            if not wrote:
                f_out.write(" ".join(tokens) + "\t" + str(target_idx) + "\t" + line.split("\t")[2] + "\t" + line.split("\t")[3] + "\t" +
                            str(-1) + "\t" + str(-1) + "\t" + str(-1) + "\t" + str(-1) + "\t" + str(-1) + "\t" + str(-1) + "\n")

    def filter_only_verb_timebank(self):
        source_lines = [x.strip() for x in open("samples/duration/timebank_svo.txt").readlines()]
        srl_objects = list(jsonlines.Reader([x.strip() for x in open("samples/timebank_srl.jsonl").readlines()]))
        f_out = open("samples/duration/timebank_svo_verbonly_strict.txt", "w")
        for i, line in enumerate(source_lines):
            tokens = line.split("\t")[0].split()
            cont = True
            for t in tokens:
                if t.lower() in self.extractor.duration_keys:
                    cont = False
            if not cont:
                continue
            ta = self.extractor.pipeline.doc([tokens], pretokenized=True)
            verb_pos = int(line.split("\t")[1])
            pos_1 = int(line.split("\t")[4])
            pos_2 = int(line.split("\t")[6])
            pos_3 = int(line.split("\t")[8])
            neg_count = 0
            if pos_1 == -1:
                neg_count += 1
            if pos_2 == -1:
                neg_count += 1
            if pos_3 == -1:
                neg_count += 1
            if ta.get_pos[verb_pos]['label'][0] == "V" and neg_count < 3:
                tokens[verb_pos] = "[" + tokens[verb_pos] + "]"
                f_out.write(line + "\n")
                pass
            # elif ta.get_pos[verb_pos]['label'] == "JJ":
            #     if int(line.split("\t")[4]) > -1 or int(line.split("\t")[6]) > -1 or int(line.split("\t")[8]) > -1:
            #         f_out.write(line + "\n")
            #         continue
            #     obj = srl_objects[i]
            #     found = False
            #     for verb in obj['verbs']:
            #         tags = verb['tags']
            #         arg0_s, arg0_e = self.get_subj_position(tags)
            #         arg1_s, arg1_e = self.get_obj_position(tags)
            #         arg2_s, arg2_e = self.get_arg3_position(tags)
            #         groups = line.split("\t")
            #         if arg2_s <= verb_pos < arg2_e:
            #             v_pos = self.get_verb_position(tags)
            #             f_out.write(groups[0] + "\t" + str(v_pos) + "\t" + groups[2] + "\t" + groups[3] + "\t"
            #                         + str(arg0_s) + "\t" + str(arg0_e) + "\t"
            #                         + str(arg1_s) + "\t" + str(arg1_e) + "\t"
            #                         + str(arg2_s) + "\t" + str(arg2_e) + "\n")
            #             found = True
            #             break
            #     if not found:
            #         f_out.write(line + "\n")
            #         pass
            # else:
            #     f_out.write(line + "\n")
            #     # tokens[verb_pos] = "[" + tokens[verb_pos] + "]"
            #     # print(" ".join(tokens))
            #     pass

    def remove_dups(self, input_file, output_file):
        source_lines = [x.strip() for x in open(input_file).readlines()]
        f_out = open(output_file, "w")
        s = set()
        for line in source_lines:
            if line not in s:
                s.add(line)
                f_out.write(line + "\n")

    def produce_simple_sents(self, input_file, output_file):
        source_lines = [x.strip() for x in open(input_file).readlines()]
        f_out = open(output_file, "w")
        for line in source_lines:
            tokens = line.split("\t")[0].split()
            mask = [0] * len(tokens)
            subj_start = int(line.split("\t")[3])
            subj_end = int(line.split("\t")[4])
            obj_start = int(line.split("\t")[5])
            obj_end = int(line.split("\t")[6])
            arg2_start = int(line.split("\t")[7])
            arg2_end = int(line.split("\t")[8])
            if subj_start > -1:
                for i in range(subj_start, subj_end):
                    mask[i] = 1
            if obj_start > -1:
                for i in range(obj_start, obj_end):
                    mask[i] = 1
            if arg2_start > -1:
                for i in range(arg2_start, arg2_end):
                    mask[i] = 1
            mask[int(line.split("\t")[1])] = 2

            final_tokens = []
            verb_pos = -1
            for i in range(len(tokens)):
                if mask[i] > 0:
                    final_tokens.append(tokens[i])
                if mask[i] == 2:
                    verb_pos = len(final_tokens) - 1
            f_out.write(" ".join(final_tokens) + "\t" + str(verb_pos) + "\n")

    def eval_temporal(self):
        lines = [x.strip() for x in open("samples/tempeval/base-segmentation.tab").readlines()]
        sentences = []
        doc_ids = []
        sent_ids = []
        cur_doc_id = ""
        cur_sent_id = ""
        cur_tokens = []
        for line in lines:
            group = line.split("\t")
            doc_id = group[0]
            sent_id = int(group[1])
            token = group[3]
            if cur_sent_id != "" and sent_id != cur_sent_id:
                sentences.append(cur_tokens)
                doc_ids.append(cur_doc_id)
                sent_ids.append(cur_sent_id)
                cur_tokens = []
            cur_sent_id = sent_id
            cur_doc_id = doc_id
            cur_tokens.append(token)

        predictions = []
        for i, tokens in enumerate(sentences):
            for j, t in enumerate(tokens[1:]):
                if t in self.extractor.duration_keys and self.extractor.quantity(tokens[j - 1]) is not None and self.extractor.validate_sentence(" ".join(tokens)):
                    predictions.append([doc_ids[i], sent_ids[i], j - 2])

        gold_lines = [x.strip() for x in open("samples/tempeval/timex-attributes.tab").readlines()]
        answers = []
        for line in gold_lines:
            group = line.split("\t")
            if group[7] == "DURATION":
                answers.append([group[0], int(group[1]), int(group[2])])

        correct = 0.0
        print(answers)
        for p in predictions:
            found = False
            for a in answers:
                if a[0] == p[0] and a[1] == p[1] and a[2] + 2 >= p[2] >= a[2] - 2:
                    found = True
                    correct += 1.0
                    break
            if not found:
                print(p)

        print(len(predictions))
        print("Precision: " + str(correct / float(len(predictions))))
        print("Recall: " + str(correct / float(len(answers))))


class VerbBaseline:

    def __init__(self, source_path):
        self.pipeline = local_pipeline.LocalPipeline()
        self.path = source_path
        self.output_map = {}
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
            "inf": 20.0 * 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
        }

    def get_seconds_from_timex(self, timex):
        quantity = float(timex.split()[0])
        unit_val = self.convert_map[timex.split()[1].lower()]
        return quantity * unit_val

    def get_top_verbs(self, in_file, out_file):
        lines = [x.strip() for x in open(in_file).readlines()]
        output_map = {}
        count = 0
        for line in lines:
            tokens = line.split("\t")[0].split()
            verb_pos = int(line.split("\t")[1])
            seconds = self.get_seconds_from_timex(line.split("\t")[2])
            if seconds > 290304000.0:
                continue
            count += 1
            if count % 1000 == 0:
                print("Processed: " + str(count))
            doc = self.pipeline.doc([tokens], pretokenized=True)
            key = list(doc.get_lemma)[verb_pos]["label"].lower()

            if key not in output_map:
                output_map[key] = 0

            output_map[key] += 1

        pickle.dump(output_map, open(out_file, "wb"))

    def print_top_verbs(self, in_pickle):
        m = pickle.load(open(in_pickle, "rb"))
        count = 0
        for f in sorted(m.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
            print(f[0])
            count += 1
            if count > 100:
                break

    def save_map(self, output_path):
        pickle.dump(self.output_map, open(output_path, "wb"))

    def process(self, output_prefix):
        lines = [x.strip() for x in open(self.path).readlines()]
        counter = 0
        file_idx = 14
        for line in lines:
            counter += 1
            if counter <= 1400000:
                continue
            tokens = line.split()
            left_search = -1
            right_search = -1
            target = -1
            value = ""
            for (i, t) in enumerate(tokens):
                if t.lower() in self.convert_map:
                    target = i
                    left_search = i - 2
                    right_search = i + 1
                    value = str(GigawordExtractor.quantity(tokens[i - 1])) + " " + t

            doc = self.pipeline.doc([tokens], pretokenized=True)
            pos_view = doc.get_pos

            pos_tags = list(pos_view)
            select_left = -1
            for i in range(left_search, -1, -1):
                label = pos_tags[i]["label"]
                if label.startswith("VB") and label != "VBP":
                    select_left = i
            select_right = -1
            for i in range(right_search, len(pos_tags)):
                label = pos_tags[i]["label"]
                if label.startswith("VB") and label != "VBP":
                    select_right = i

            selected = select_left
            if abs(select_right - target) < abs(select_left - target) or select_left == -1:
                selected = select_right

            if selected < 0 or selected >= len(pos_tags):
                continue

            key = list(doc.get_lemma)[selected]["label"].lower()

            if key not in self.output_map:
                self.output_map[key] = []

            self.output_map[key].append(self.get_seconds_from_timex(value))

            if counter % 10000 == 0:
                print("Processed: " + str(counter))

            if counter % 100000 == 0:
                self.save_map(output_prefix + str(file_idx) + ".pkl")
                self.output_map = {}
                file_idx += 1

        self.save_map(output_prefix + str(file_idx) + ".pkl")

    def save_map(self, output_path):
        pickle.dump(self.output_map, open(output_path, "wb"))

    @staticmethod
    def merge_map(input_prefix, output_path):
        ret_map = {}
        for i in range(0, 21):
            file_path = input_prefix + str(i) + ".pkl"
            with open(file_path, "rb") as f_in:
                cur_map = pickle.load(f_in)
                for key in cur_map:
                    if key not in ret_map:
                        ret_map[key] = []
                    ret_map[key] = ret_map[key] + cur_map[key]
        pickle.dump(ret_map, open(output_path, "wb"))

    @staticmethod
    def exp_output(map_path, key):
        with open(map_path, "rb") as f_in:
            cur_map = pickle.load(f_in)
        plt.hist(cur_map[key], bins=100, log=True, range=(0, 3153600000), density=True)
        plt.show()

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

    def transform_expression(self, exp):
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
        label_num = int(exp)

        return str(label_num) + " " + unit

    def get_label(self, lowerbound, upperbound):
        lower_val = self.get_seconds(lowerbound)
        upper_val = self.get_seconds(upperbound)

        if lower_val is None or upper_val is None or lower_val > upper_val:
            return None

        val = (lower_val + upper_val) / 2.0

        if val < 1.0 * self.convert_map["day"]:
            return 0
        else:
            return 1

    def find_distribution(self):
        with open("samples/duration/verb_formatted_all_svo_better_filter_4.txt") as f_in:
            lines = [x.strip() for x in f_in.readlines()]

        result_map = {}
        cared_verb = ["make", "made", "making"]
        cared_arg = ["breakfast", "lunch", "dinner", "coffee", "tea", "trip", "money"]
        for line in lines:
            groups = line.split("\t")
            tokens = groups[0].split()
            verb = tokens[int(groups[1])]
            arg_start = int(groups[5])
            arg_end = int(groups[6])

            if verb.lower() not in cared_verb:
                continue
            for i in range(arg_start, arg_end):
                if tokens[i].lower() in cared_arg:
                    key = tokens[i].lower()
                    if key not in result_map:
                        result_map[key] = []
                    result_map[key].append(float(groups[2].split()[0]) * self.convert_map[groups[2].split()[1]])
        f_out = open("tmp_output.txt", "w")
        for key in result_map:
            mu, std = norm.fit(np.array(result_map[key]))
            plt.hist(np.array(result_map[key]), bins=25, density=True, alpha=0.6, color='g')

            # Plot the PDF.
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
            title = key
            plt.title(title)

            plt.show()

            f_out.write(key + "\n")
            f_out.write(str(result_map[key]) + "\n")

    def get_timex_from_seconds(self, seconds):
        prev_unit = "seconds"
        for i, v in enumerate(self.convert_map):
            if seconds / self.convert_map[v] < 0.5:
                break
            prev_unit = v
        if prev_unit == "seconds" and seconds > 60.0:
            prev_unit = "centuries"
        new_val = str(round(seconds / self.convert_map[prev_unit], 2))

        return new_val + " " + prev_unit

    def find_distribution_raw(self):
        import math
        with open("samples/duration/verb_formatted_all_svo_better_filter_4.txt") as f_in:
            lines = [x.strip() for x in f_in.readlines()]

        result_map = {}
        cared_verb = {
            "attack": ["attack", "attacked", "attacking", "attacks"],
            "meet": ["meet", "met", "meeting", "meets"],
            "run": ["run", "ran", "running", "runs"],
            "pay": ["pay", "paid", "paying", "pays"],
            "learn": ["learn", "learns", "learned", "learnt"],
            "watch": ["watch", "watched", "watching", "watches"],
            "grow": ["grow", "grew", "growing", "grows"],
            "walk": ["walk", "walked", "walking", "walks"],
            "consider": ["consider", "considers", "considering", "considered"],
            "buy": ["buys", "buy", "bought", "buying"],
            "kill": ["kills", "kill", "killed", "killing"],
            "ask": ["asks", "ask", "asking", "asked"],
            "call": ["calls", "call", "called", "calling"],
                      }
        for k in cared_verb:
            result_map[k] = []
        result_map["all"] = []
        for line in lines:
            groups = line.split("\t")
            tokens = groups[0].split()
            verb = tokens[int(groups[1])].lower()
            norm_key = ""
            for key in cared_verb:
                if verb in cared_verb[key]:
                    norm_key = key
            norm_key = "all"
            if norm_key == "":
                continue
            val = float(groups[2].split()[0]) * float(self.convert_map[groups[2].split()[1]])
            if val <= 0:
                continue
            result_map[norm_key].append(
                val
                # math.log(val)
            )
        for key in result_map:
            plt.subplots(4, 2, figsize=(60, 30))
            for j, max_unit in enumerate(["hours", "days", "weeks", "months", "years", "centuries", "inf"]):
                max_val = self.convert_map[max_unit]
                # max_val_log = math.log(max_val)
                max_val_log = max_val
                print(key)
                print(max_unit)
                bin_count = 20
                plt.subplot(4, 2, j + 1)
                n, bins, patches = plt.hist(np.array(result_map[key]), bins=bin_count, range=(0, max_val_log))
                interval = max_val_log / float(bin_count)
                labels = []
                for i in range(bin_count + 1):
                    cur_val = interval * float(i)
                    labels.append(self.get_timex_from_seconds(cur_val))
                plt.xticks(bins, labels)
                plt.title(key + ", " + max_unit + ", " + "log")
            plt.savefig("figures/" + key + "_" + max_unit + "_log.png")

    def test_file(self, map_path, path):
        with open(map_path, "rb") as f_in:
            cur_map = pickle.load(f_in)
        lines = [x.strip() for x in open(path).readlines()]

        labels = [0, 1]

        correct = {}
        predicted = {}
        labeled = {}

        f_out = open("samples/duration/timebank_filtered.txt", "w")
        for line in lines:
            tokens = line.split("\t")[0].split()
            target_idx = int(line.split("\t")[1])
            lowerbound = line.split("\t")[2]
            upperbound = line.split("\t")[3]

            label = self.get_label(lowerbound, upperbound)
            if label is None:
                continue

            f_out.write(line + "\n")
            if label not in labeled:
                labeled[label] = 0.0
            labeled[label] += 1.0

            doc = self.pipeline.doc([tokens], pretokenized=True)
            key = (doc.get_lemma)[target_idx]["label"].lower()

            predicted_label = random.choice(labels)

            day_val = 1.0 * self.convert_map["day"]
            if key in cur_map:
                pos = -1
                data_points = sorted(cur_map[key])
                data_maps = {}
                for d in data_points:
                    if d not in data_maps:
                        data_maps[d] = 0.0
                    data_maps[d] += 1.0
                most_popular_val = sorted(data_maps.items(), key=lambda item: item[1], reverse=True)[0][1]
                if most_popular_val < day_val:
                    predicted_label = 0
                else:
                    predicted_label = 1
                # for i, d in enumerate(data_points):
                #     if day_val > d:
                #         pos = i
                #         break
                # if pos > -1:
                #     if float(pos) / float(len(data_points)) > 0.5:
                #         predicted_label = 0
                #     else:
                #         predicted_label = 1

            if predicted_label not in predicted:
                predicted[predicted_label] = 0.0
            predicted[predicted_label] += 1.0

            if predicted_label == label:
                if label not in correct:
                    correct[label] = 0.0
                correct[label] += 1.0

        for key in correct:
            name = "Less than a day"
            if key == 1:
                name = "Longer than a day"
            p = correct[key] / predicted[key]
            r = correct[key] / labeled[key]
            f = 2 * p * r / (p + r)

            print(name)
            print(str(p) + ", " + str(r) + ", " + str(f))
            print()

    def prepare_annotation(self):
        import math
        lines = [x.strip() for x in open("samples/duration/timebank_svo_verbonly_strict.txt").readlines()]
        f_out = open("samples/duration/timebank_mturk.txt", "w")
        for line in lines:
            groups = line.split("\t")
            tokens = groups[0].split()
            verb_pos = int(groups[1])
            tokens[verb_pos] = "<strong><font color='red'>" + tokens[verb_pos] + "</font></strong>"
            lower_val = self.get_seconds(groups[2])
            upper_val = self.get_seconds(groups[3])
            lower_e = math.log(lower_val)
            upper_e = math.log(upper_val)

            if (lower_e + upper_e) / 2.0 >= 11.3666:
                true_label = "long"
            else:
                true_label = "short"

            # phrase = "LONGER than a day"
            # if true_label == "long":
            #     phrase = "SHORTER than a day"
            phrase_1 = "SHORTER than " + self.transform_expression(groups[2])
            phrase_2 = "LONGER than " + self.transform_expression(groups[3])

            f_out.write(" ".join(tokens) + "\t" + phrase_1 + "\t" + phrase_2 + "\n")

    def prepare_pair_annotation(self):
        import math
        lines = [x.strip() for x in open("samples/duration/timebank_svo_verbonly_strict.txt").readlines()]
        f_out = open("samples/duration/timebank_mturk_pair.txt", "w")
        f_out_reserve = open("samples/duration/timebank_mturk_pair_train.txt", "w")
        random.shuffle(lines)
        train_lines = lines[:1000]
        for l in train_lines:
            f_out_reserve.write(l + "\n")
        eval_lines = lines[1000:]
        selected_set = set()
        for i in range(0, 1000):
            line_1 = random.choice(eval_lines)
            line_2 = random.choice(eval_lines)
            if line_1 == line_2:
                continue
            if line_1 + line_2 in selected_set or line_2 + line_1 in selected_set:
                continue
            selected_set.add(line_1 + line_2)
            selected_set.add(line_2 + line_1)

            groups = line_1.split("\t")
            tokens = groups[0].split()
            verb_pos = int(groups[1])
            tokens[verb_pos] = "<strong><font color='red'>" + tokens[verb_pos] + "</font></strong>"

    def print_errors(self):
        import math
        data_lines = [x.strip() for x in open("samples/duration/timebank_svo_improved.txt").readlines()]
        prediction_lines = [x.strip() for x in open("results.txt").readlines()]
        f_out = open("tmp_output.txt", "w")
        verb_freq = {}
        correct = 0.0
        all_preds = []
        all_labels = []
        for i, line in enumerate(data_lines):
            groups = line.split("\t")
            lower_val = self.get_seconds(groups[2])
            upper_val = self.get_seconds(groups[3])
            lower_e = math.log(lower_val)
            upper_e = math.log(upper_val)

            if (lower_e + upper_e) / 2.0 >= 11.3666:
                true_label = "long"
            else:
                true_label = "short"
            all_labels.append(true_label)

            prediction = prediction_lines[i]
            if prediction[:3] == "1.0":
                predicted_label = "short"
            else:
                predicted_label = "long"

            tokens = groups[0].split()
            doc = self.pipeline.doc([tokens], pretokenized=True)
            v = list(doc.get_lemma)[int(groups[1])]["label"].lower()
            if v == "say":
                predicted_label = "short"
            if v == "talk":
                predicted_label = "short"
            if v == "announce":
                predicted_label = "short"
            if v == "report":
                predicted_label = "short"
            if v == "touch":
                predicted_label = "short"
            if v == "kill":
                predicted_label = "short"
            if v == "think":
                predicted_label = "long"
            if v == "believe":
                predicted_label = "long"
            all_preds.append(predicted_label)
            tokens[int(groups[1])] = "[" + tokens[int(groups[1])] + "]"
            # if v not in verb_freq:
            #     verb_freq[v] = 0
            # verb_freq[v] += 1

            if true_label != predicted_label:
                f_out.write(" ".join(tokens) + "\t" + true_label + "\t" + predicted_label + "\t" + groups[2] + "\t" + groups[3] + "\n")
                if v not in verb_freq:
                    verb_freq[v] = 0
                verb_freq[v] += 1
            else:
                correct += 1.0
        for f in sorted(verb_freq.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
            print(f)
        print(correct)

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
                if p == "short":
                    s_correct += 1.0
                else:
                    l_correct += 1.0
        for p in all_preds:
            if p == "long":
                l_predicted += 1.0
            else:
                s_predicted += 1.0
        for l in all_labels:
            if l == "long":
                l_labeled += 1.0
            else:
                s_labeled += 1.0

        print("Acc.: " + str(correct / float(len(all_preds))))
        print("Less than a day: " + str(s_correct / s_predicted) + ", " + str(s_correct / s_labeled))
        print("Longer than a day: " + str(l_correct / l_predicted) + ", " + str(l_correct / l_labeled))

    def print_verb_distribution(self):
        data_lines = [x.strip() for x in open("samples/duration/verb_formatted_all_svo_better_filter_non_mask.txt").readlines()]
        random.shuffle(data_lines)
        data_lines = data_lines[:100000]
        data_points = []
        bin_1 = 0
        bin_2 = 0
        bin_3 = 0
        bin_4 = 0
        for i, line in enumerate(data_lines):
            groups = line.split("\t")
            tokens = groups[0].split()
            v = tokens[int(groups[1])]
            # if v == "say" or v == "said" or v == "saying":
            # if v == "call" or v == "calling" or v == "called":
            # if v == "see" or v == "seeing" or v == "saw":
            if True:
                val = self.get_seconds_from_timex(groups[2]) / 3600.0
                if val < 100.0:
                    bin_1 += 1
                elif val < 200.0:
                    bin_2 += 1
                elif val < 300.0:
                    bin_3 += 1
                else:
                    bin_4 += 1
            # doc = self.pipeline.doc([tokens], pretokenized=True)
            # v = list(doc.get_lemma)[int(groups[1])]["label"].lower()
            # if v == "say":
                data_points.append(self.get_seconds_from_timex(groups[2]) / 3600.0)
        print(bin_1 / float(len(data_points)))
        print(bin_2 / float(len(data_points)))
        print(bin_3 / float(len(data_points)))
        print(bin_4 / float(len(data_points)))
        plt.hist(np.array(data_points), bins=range(0, 500, 10))
        plt.show()

    def sampling_data(self):
        import math
        import random
        lines = [x.strip() for x in open("samples/duration/verb_formatted_all_svo_better_filter_3.txt").readlines()]
        bucket_a = []
        bucket_b = []
        bucket_c = []
        bucket_d = []
        for line in lines:
            timex = line.split("\t")[2]
            seconds = self.get_seconds_from_timex(timex)
            if math.exp(5.0) > seconds:
                bucket_a.append(line)
            elif math.exp(10.0) > seconds:
                bucket_b.append(line)
            elif math.exp(15.0) > seconds:
                bucket_c.append(line)
            elif 290304000.0 > seconds:
                bucket_d.append(line)

        len_a = float(len(bucket_a))
        len_b = float(len(bucket_b))
        len_c = float(len(bucket_c))
        len_d = float(len(bucket_d))
        total_len = len(bucket_a) + len(bucket_b) + len(bucket_c) + len(bucket_d)
        total_len = float(total_len)

        scores = np.array([total_len / len_a, total_len / len_b, total_len / len_c, total_len / len_d])
        norm = scores / np.linalg.norm(scores)
        print(norm)

        bucket_a = []
        bucket_b = []
        for line in lines:
            timex = line.split("\t")[2]
            seconds = self.get_seconds_from_timex(timex)
            if 24 * 3600.0 > seconds:
                bucket_a.append(line)
            elif 290304000.0 > seconds:
                bucket_b.append(line)
        len_a = float(len(bucket_a))
        len_b = float(len(bucket_b))
        print(len_a)
        print(len_b)



        # random.shuffle(bucket_a)
        # random.shuffle(bucket_b)
        # random.shuffle(bucket_c)
        # random.shuffle(bucket_d)
        #
        # bucket_a = bucket_a[:50]
        # bucket_b = bucket_b[:50]
        # bucket_c = bucket_c[:50]
        # bucket_d = bucket_d[:50]
        #
        # f_out = open("samples/duration/verb_bucket_samples.txt", "w")
        # for line in bucket_a + bucket_b + bucket_c + bucket_d:
        #     tokens = line.split("\t")[0].split()
        #     verb_idx = int(line.split("\t")[1])
        #     tokens[verb_idx] = "[" + tokens[verb_idx] + "]"
        #     sent = " ".join(tokens)
        #     f_out.write(sent + "\t" + line.split("\t")[2] + "\n")


class VerbPhysicsInstance:
    def __init__(self, obj_1, obj_2, label):
        self.obj_1 = obj_1
        self.obj_2 = obj_2
        self.label = label


class VerbPhysicsEval:

    def __init__(self):
        pass

    def process_raw_file(self, paths, out_path):
        lines = []
        for path in paths:
            lines += [x.strip() for x in open(path).readlines()]
        obj_set = set()
        for line in lines:
            if line[0] == ",":
                continue
            obj_set.add(line.split(",")[0])
            obj_set.add(line.split(",")[1])

        verbs = [
            "clean", "make", "build", "use", "move",
            "lift", "take", "try", "play", "hold",
            "turn", "cut", "throw", "open", "wash",
            "buy", "receive", "write", "kick", "start",
            "break", "destroy", "smash", "split", "fight",
            "climb", "dig", "stack", "shake", "ride",
        ]
        # verbs_size = ["clean", "move", "make", "store", "hide", "climb", "hold", "wash"]
        verbs_20 = ["cleaned", "moved", "made", "stored", "hid", "climbed", "held", "washed", "carried", "threw",
                    "built", "kicked", "split", "cut", "broke", "smashed", "opened", "cleaned", "shook", "turned",
                    ]
        # verbs_100 = ["be", "hold", "play", "have", "go", "work", "make", "take", "wait", "do", "live", "stay", "run", "lose", "kill", "win", "come", "see", "keep", "get", "rise", "suspend", "leave", "score", "meet", "fall", "pay", "remain", "spend", "sell", "increase", "serve", "cook", "put", "grow", "cut", "give", "close", "double", "receive", "sit", "become", "raise", "detain", "use", "complete", "return", "continue", "try", "hit", "open", "begin", "reach", "send", "drop", "fight", "die", "say", "start", "move", "turn", "build", "lead", "delay", "change", "bring", "speak", "miss", "provide", "ban", "stand", "set", "find", "finish", "reduce", "expect", "report", "jail", "beat", "talk", "happen", "sideline", "add", "simmer", "face", "sign", "invest", "show", "visit", "buy", "look", "stop", "know", "shut", "cost", "drive", "carry", "decide", "save", "bake"]

        verbs = verbs_20
        pronouns = ["he", "they", "I", "she"]
        rest = "\t1\t1.0 hour\twalk\t0\t1.0 hour\tNONE\n"
        f_out = open(out_path, "w")
        for obj in obj_set:
            for v in verbs:
                for p in pronouns:
                    pro = "a"
                    if obj[0] in ['a', 'e', 'i', 'o', 'u']:
                        pro = "an"
                    sentence = p + " " + v + " " + pro + " " + obj + " ."
                    f_out.write(sentence + rest)

    def add_list(self, a, b):
        for i, _ in enumerate(a):
            a[i] += b[i]
        return a

    def div_list(self, a, b):
        for i, _ in enumerate(a):
            a[i] = a[i] / b
        return a

    def process_embedding_file(self, sent_path, embed_path):
        import math
        sent_lines = [x.strip() for x in open(sent_path).readlines()]
        embed_lines = [x.strip() for x in open(embed_path).readlines()]

        obj_map = {}
        for i, l in enumerate(sent_lines):
            sent = l.split("\t")[0]
            obj = sent.split()[3]
            verb = sent.split()[1]

            if obj not in obj_map:
                obj_map[obj] = {}
            if verb not in obj_map[obj]:
                obj_map[obj][verb] = [0.0] * 9

            probs = [math.exp(float(x)) for x in embed_lines[i].split('\t')[0:9]]
            prob_sum = sum(probs)
            probs = [x / float(prob_sum) for x in probs]
            # vector = [float(x) * float(i) for i, x in enumerate(probs)]
            # mean = sum(vector)
            # embed = mean
            obj_map[obj][verb] = self.add_list(obj_map[obj][verb], probs)

        # verbs = ["clean", "make", "build", "use", "move"]
        # verbs = ["clean", "make", "build", "use", "move", "lift", "take", "try", "play", "hold"]
        # verbs = [
        #     "clean", "make", "build", "use", "move",
        #     "lift", "take", "try", "play", "hold",
        #     "turn", "cut", "throw", "open", "wash",
        #     "buy", "receive", "write", "kick", "start",
        #     "break", "destroy", "smash", "split", "fight",
        #     "climb", "dig", "stack", "shake", "ride",
        # ]
        # verbs = ["be", "hold", "play", "have", "go", "work", "make", "take", "wait", "do", "live", "stay", "run", "lose", "kill", "win", "come", "see", "keep", "get", "rise", "suspend", "leave", "score", "meet", "fall", "pay", "remain", "spend", "sell", "increase", "serve", "cook", "put", "grow", "cut", "give", "close", "double", "receive", "sit", "become", "raise", "detain", "use", "complete", "return", "continue", "try", "hit", "open", "begin", "reach", "send", "drop", "fight", "die", "say", "start", "move", "turn", "build", "lead", "delay", "change", "bring", "speak", "miss", "provide", "ban", "stand", "set", "find", "finish", "reduce", "expect", "report", "jail", "beat", "talk", "happen", "sideline", "add", "simmer", "face", "sign", "invest", "show", "visit", "buy", "look", "stop", "know", "shut", "cost", "drive", "carry", "decide", "save", "bake"]
        verbs = ["cleaned", "moved", "made", "stored", "hid", "climbed", "held", "washed", "carried", "threw",
                 "built", "kicked", "split", "cut", "broke", "smashed", "opened", "cleaned", "shook", "turned",
                ]
        embedding_file = open("samples/vp_clean/reannotations/obj_embedding_20v_joint.pkl", "wb")
        embed_map = {}
        for key in obj_map:
            if key not in embed_map:
                embed_map[key] = []
            for v in verbs:
                embed_map[key].extend(self.div_list(obj_map[key][v], 4.0))
            print(len(embed_map[key]))

        pickle.dump(embed_map, embedding_file)

    def read_instances(self, file_name):
        instances = []
        for line in [x.strip() for x in open(file_name).readlines()]:
            if line[0] == ",":
                continue
            group = line.split(",")
            obj_1 = group[1]
            obj_2 = group[2]

            label_map = {
                "1": 0,
                "-1": 1,
                "0": 2,
                "-42": 3,
            }
            instance = VerbPhysicsInstance(obj_1, obj_2, label_map[group[4]])
            if instance.label < 2:
                instances.append(instance)
        return instances

    def run_unsupervisied(self):
        obj_means = pickle.load(open("samples/verbphysics/train-5/obj_embedding_30v_mean.pkl", "rb"))
        instances = self.read_instances("samples/verbphysics/train-5/dev.csv")

        all_preds = []
        all_labels = []

        for instance in instances:
            mean_1 = obj_means[instance.obj_1]
            mean_2 = obj_means[instance.obj_2]
            label = instance.label

            # if float(abs(mean_2 - mean_1)) / float(min(mean_1, mean_2)) < 0.000000001:
            #     prediction = 2
            if mean_2 > mean_1:
                prediction = 1
            else:
                prediction = 0

            all_labels.append(label)
            all_preds.append(prediction)
        assert len(all_preds) == len(all_labels)
        correct = 0.0
        for i, p in enumerate(all_preds):
            if all_labels[i] == p:
                correct += 1.0
        print("Acc.: " + str(correct / float(len(all_preds))))


if __name__ == "__main__":
    # extractor = GigawordExtractor()
    # extractor.get_rid_of_masks("samples/verbs/test.txt", "samples/verbs_nonmask/test.txt")
    # extractor.prepare_nom_data("samples/duration/all/nominals.txt", "samples/duration/all/nominals_formatted.txt")
    # extractor.split_nominals()
    # extractor.read_file("/Users/xuanyuzhou/Downloads/tmp/apw_eng/apw_eng_199411")
    # extractor.process_path("/Users/xuanyuzhou/Downloads/tmp/all", duration_path="samples/duration_more/duration_all.txt")
    # extractor.process_duration_initial_filter("samples/duration_more/duration_all.txt", "samples/duration_more/duration_all_filtered.txt")

    # srl = AllenSRL()
    # srl.predict_file("samples/duration_afp_eng_filtered.txt")

    # runner = SRLRunner()
    # runner.eval_temporal()
    # runner.count_label("samples/duration/verb_formatted_all_svo.txt")
    # runner.prepare_timebank_srl()
    # runner.filter_only_verb_timebank()
    # runner.prepare_timebank_file()
    # runner.parse_srl_file("samples/duration_srl_verbs.476452.jsonl")
    # runner.parse_srl_file("samples/duration_srl_verbs_3.jsonl")
    # runner.parse_srl_file("samples/duration/verb_nyt_svo.jsonl")
    # runner.prepare_verb_file()
    # runner.prepare_simple_verb_file()
    # runner.remove_dups("samples/duration/verb_formatted_all_svo_better_filter_5.txt", "samples/duration/verb_formatted_all_svo_better_filter_5.txt")
    # runner.produce_simple_sents("samples/duration/verb_formatted_all_svo_better_filter_4.txt", "samples/duration/verb_formatted_all_simple.txt")
    # runner.prepare_verb_file_for_failures()
    # runner.print_random_srl_json()
    # runner.print_random_srl()
    # runner.prepare_nyt_srl_file()
    # runner.print_file("samples/duration/duration_srl_fail.jsonl")

    # extractor = GigawordExtractor()
    # extractor.get_rid_of_masks("samples/duration/verb_formatted_all_svo_better_filter_4.txt", "samples/duration/verb_formatted_all_svo_better_filter_4.txt")
    # extractor.get_rid_of_masks("samples/duration/all/nominals_formatted.txt", "samples/duration/all/nominals_formatted_train.txt")

    # baseline = VerbBaseline("samples/duration/all/verbs.txt")
    # baseline.find_distribution_raw()
    # baseline.process("samples/duration/all/nearest_verb_cont/partition_")
    # VerbBaseline.merge_map("samples/duration/all/nearest_verb_all/partition_", "samples/duration/all/nearest_verb.pkl")
    # VerbBaseline.exp_output("samples/duration/all/nearest_verb.pkl", "work")

    # baseline.prepare_annotation()
    # baseline.get_top_verbs("samples/duration/verb_formatted_all_svo_better_filter_3.txt", "samples/tmp_map.pkl")
    # baseline.print_top_verbs("samples/tmp_map.pkl")
    # baseline.print_errors()
    # baseline.print_verb_distribution()
    # baseline.sampling_data()
    # baseline.find_distribution()
    # baseline.test_file("samples/duration/all/nearest_verb.pkl", "samples/duration/timebank_formatted.txt")

    verbphysics = VerbPhysicsEval()
    # verbphysics.process_raw_file([
    #     "samples/vp_clean/reannotations/train.csv",
    #     "samples/vp_clean/reannotations/test.csv",
    #     "samples/vp_clean/reannotations/dev.csv"], "samples/vp_clean/reannotations/obj_file_20v.txt")
    verbphysics.process_embedding_file("samples/vp_clean/reannotations/obj_file_20v.txt", "bert_vp_eval/bert_logits.txt")
    # verbphysics.process_raw_file([
    #     "samples/verbphysics/train-5/train.csv",
    #     "samples/verbphysics/train-5/test.csv",
    #     "samples/verbphysics/train-5/dev.csv",], "samples/verbphysics/train-5/obj_file_100v.txt")
    # verbphysics.process_embedding_file("samples/verbphysics/train-5/obj_file_100v.txt", "result_logits.txt")
    # verbphysics.run_unsupervisied()
