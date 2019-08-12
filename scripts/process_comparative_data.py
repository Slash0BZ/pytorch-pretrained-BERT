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

    def process_paragraph(self, p):
        ret = []
        ta = self.pipeline.doc(p)
        sent_view = ta.get_view("SENTENCE")
        for sent in sent_view:
            tokens_lower = sent['tokens'].lower().split()
            for (i, v) in enumerate(tokens_lower):
                if v in ["while", "during", "when"]:
                    ret.append(sent['tokens'])
                    break
        return ret

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

    def __init__(self, output_path):
        model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz',
                                'semantic-role-labeling')
        self.predictor = model.predictor()
        self.output_path = output_path

    def predict_batch(self, sentences):
        f_out = jsonlines.open(self.output_path, "w")
        for sentence in sentences:
            prediction = self.predictor.predict(sentence)
            f_out.write(prediction)

    def predict_file(self, path):
        with open(path) as f:
            lines = [x.strip() for x in f.readlines()]
        self.predict_batch(lines)


class SentenceProcessor:

    def __init__(self):
        pass

    def process_line(self, line):
        tokens = line.split()
        for t in tokens:
            if t == "while":
                return True
        return False

    def process_document(self, path):
        lines = [x.strip() for x in open(path).readlines()]
        for line in lines:
            status = self.process_line(line)
            if status:
                print(line)

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

    def get_all_related_tokens_with_verb_pos(self, tokens, tags, exclude_tmp=True):
        continued_count = 0
        verb_pos = -1
        ret_tokens = []
        for i, t in enumerate(tags):
            if t == "O":
                continued_count += 1
                continue
            elif t == "B-V":
                verb_pos = i - continued_count
                ret_tokens.append(tokens[i])
            elif t in ["B-ARGM-TMP", "I-ARGM-TMP"]:
                if exclude_tmp:
                    continued_count += 1
                    continue
                else:
                    ret_tokens.append(tokens[i])
            else:
                ret_tokens.append(tokens[i])
        return ret_tokens, verb_pos

    def get_tmp_range(self, tags):
        start = -1
        end = -1
        for i, t in enumerate(tags):
            if t == "B-ARGM-TMP":
                start = i
                end = i + 1
            if t == "I-ARGM-TMP":
                end += 1
        return [start, end]

    def get_verb_position(self, tags):
        for i, t in enumerate(tags):
            if t == "B-V":
                return i
        return -1

    def parse_srl_file(self, path, out_path):
        lines = [x.strip() for x in open(path).readlines()]
        reader = jsonlines.Reader(lines)

        f_out = jsonlines.open(out_path, mode="a")
        for obj in reader:
            valid = False
            for verbs_obj in obj['verbs']:
                tokens = obj['words']
                tags = verbs_obj['tags']
                parsed = self.parse_single_seq(tokens, tags)
                if "ARGMTMP" in parsed:
                    tmp_arg = parsed["ARGMTMP"]
                    for i, v in enumerate(tmp_arg):
                        if v.lower() in ["while"]:
                            valid = True
            if valid:
                f_out.write(obj)

    def format_srl_file(self, path, out_path):
        lines = [x.strip() for x in open(path).readlines()]
        reader = jsonlines.Reader(lines)
        f_out = open(out_path, "w")

        for obj in reader:
            tmp_range = [-1, -1]
            short_event_verb = -1
            long_event_verb = -1
            for verbs_obj in obj['verbs']:
                tokens = obj['words']
                tags = verbs_obj['tags']
                parsed = self.parse_single_seq(tokens, tags)
                if "ARGMTMP" in parsed:
                    tmp_arg = parsed["ARGMTMP"]
                    for i, v in enumerate(tmp_arg):
                        if v.lower() in ["while"]:
                            short_event_verb = self.get_verb_position(tags)
                            tmp_range = self.get_tmp_range(tags)
                            break
                if tmp_range[0] <= self.get_verb_position(tags) < tmp_range[1]:
                    long_event_verb = self.get_verb_position(tags)
            if short_event_verb > -1 and long_event_verb > -1:
                f_out.write(" ".join(obj["words"]) + "\t" + str(short_event_verb) + "\t" + str(long_event_verb) + "\n")

    def format_srl_file_simplified_pair(self, path, out_path):
        lines = [x.strip() for x in open(path).readlines()]
        reader = jsonlines.Reader(lines)
        f_out = open(out_path, "w")

        for obj in reader:
            tmp_range = [-1, -1]
            short_event_verb = -1
            long_event_verb = -1
            short_event_tokens = []
            long_event_tokens = []
            for verbs_obj in obj['verbs']:
                tokens = obj['words']
                tags = verbs_obj['tags']
                parsed = self.parse_single_seq(tokens, tags)
                if "ARGMTMP" in parsed:
                    tmp_arg = parsed["ARGMTMP"]
                    for i, v in enumerate(tmp_arg):
                        if v.lower() in ["while"]:
                            tmp_range = self.get_tmp_range(tags)
                            short_event_tokens, short_event_verb = self.get_all_related_tokens_with_verb_pos(tokens, tags)
                            break
                if tmp_range[0] <= self.get_verb_position(tags) < tmp_range[1]:
                    long_event_tokens, long_event_verb = self.get_all_related_tokens_with_verb_pos(tokens, tags)
                    if len(long_event_tokens) < 3:
                        long_event_verb = -1
                        long_event_tokens = []
                    else:
                        break
            if short_event_verb > -1 and long_event_verb > -1:
                f_out.write(" ".join(short_event_tokens) + "\t" + str(short_event_verb) + "\t" + "NONE\t" +
                            " ".join(long_event_tokens) + "\t" + str(long_event_verb) + "\tNONE\n")

    def combine_abs_comp(self, abs_path, comp_path, out_path):
        lines = [x.strip() for x in open(abs_path).readlines()]
        f_out = open(out_path, "w")
        for i in range(0, len(lines) - 1):
            f_out.write("\t".join(lines[i].split("\t")[:3]) + "\t" + "\t".join(lines[i + 1].split("\t")[:3]) + "\tNONE\n")
        lines = [x.strip() for x in open(comp_path).readlines()]
        for line in lines:
            r = random.random()
            groups = line.split("\t")
            if r < 0.5:
                f_out.write(line + "\tLESS\n")
            else:
                f_out.write("\t".join(groups[3:6]) + "\t" + "\t".join(groups[:3]) + "\tMORE\n")

    def randomize_file(self, path, out_path):
        lines = [x.strip() for x in open(path).readlines()]
        random.shuffle(lines)
        f_out = open(out_path, "w")
        for l in lines:
            f_out.write(l + "\n")

    def print_readable_files(self, path, out_path):
        lines = [x.strip() for x in open(path).readlines()]
        f_out = open(out_path, "w")
        for l in lines:
            tokens = l.split("\t")[0].split()
            verb = int(l.split("\t")[1])
            label = l.split("\t")[2]
            tokens[verb] = "[" + tokens[verb] + "]"
            f_out.write(" ".join(tokens) + "\t" + label + "\n")

# g = GigawordExtractor()
# g.process_path("/Volumes/SSD/gigaword/data/afp_eng", "samples/comparative_afp_eng.txt")

processor = SentenceProcessor()
processor.format_srl_file_simplified_pair("samples/comparative_afp_eng_while_srl_valid.jsonl", "samples/comparative_afp_eng_while_srl_simplied_pairs.txt")
processor.combine_abs_comp(
    "samples/UD_English_finetune_comparative/train.original.formatted.txt",
    "samples/comparative_afp_eng_while_srl_simplied_pairs.txt",
    "samples/UD_English_finetune_comparative/train.pair.formatted.txt",
)
processor.randomize_file("samples/UD_English_finetune_comparative/train.pair.formatted.txt","samples/UD_English_finetune_comparative/train.pair.formatted.txt")
# processor.print_readable_files("samples/UD_English/dev.formatted.txt", "samples/UD_English/dev.readable.txt")
# processor.randomize_file("samples/UD_English_finetune_comparative/train.formatted.txt", "samples/UD_English_finetune_comparative/train.formatted.txt")
# processor.randomize_file("samples/UD_English_finetune_comparative/train.original.formatted.txt", "samples/UD_English_finetune_comparative/train.original.formatted.txt")
# processor.format_srl_file("samples/comparative_afp_eng_while_srl_valid.jsonl", "samples/comparative_afp_eng_while_srl_formatted.txt")
# processor.parse_srl_file("samples/comparative_afp_eng_while_srl.jsonl", "samples/comparative_afp_eng_while_srl_valid.jsonl")
# processor.process_document("samples/comparative_afp_eng.txt")

# srl = AllenSRL("samples/comparative_afp_eng_while_srl.jsonl")
# srl.predict_file("samples/comparative_afp_eng_while.txt")

