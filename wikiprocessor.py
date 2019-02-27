import sys
import os
import random
from gensim.corpora import WikiCorpus
from ccg_nlpy import local_pipeline
from word2number import w2n


class TextProcessor:

    def __init__(self):
        self.pipeline = local_pipeline.LocalPipeline()

    def num(self, n):
        try:
            return float(n)
        except:
            return None

    def word2num_raw(self, w):
        try:
            return w2n.word_to_num(w)
        except:
            return None

    def word2num(self, w):
        try:
            num = w2n.word_to_num(w)
            # Check for special cases like "2-year-old"
            if "-" in w:
                satisfied = True
                for s in w.split("-"):
                    if self.word2num_raw(s) == None:
                        satisfied = False
                        break
                if not satisfied:
                    return None
            return num
        except:
            return None

    def process_text(self, text):
        ta = self.pipeline.doc(text)
        sent_view = ta.get_view("SENTENCE")
        ret_sents = []
        ret_sents_original = []
        for sent in sent_view:
            tokens = sent['tokens'].split()
            modified = False
            for idx, token in enumerate(tokens):
                any_num = self.num(token.replace(",", ""))
                if any_num is None:
                    any_num = self.num(token)
                if any_num is None:
                    any_num = self.word2num_raw(token)
                if any_num is not None:
                    if any_num < 1700 or any_num > 2200:
                        modified = True
                if self.num(token.replace(",", "")) is not None and self.num(token) is None:
                    token[idx] = self.num(token)
                if self.word2num(token) is not None and self.num(token) is None:
                    num = self.word2num(token)
                    left_bound = idx
                    right_bound = idx + 1
                    for i in range(idx - 1, max(0, idx - 10), -1):
                        combine = " ".join(tokens[i:idx+1])
                        combined_num = self.word2num(combine)
                        if combined_num == num:
                            break
                        else:
                            num = combined_num
                        left_bound = i
                    for j in range(idx + 1, min(len(tokens) - 1, idx + 10)):
                        combine = " ".join(tokens[idx:j+1])
                        combined_num = self.word2num(combine)
                        if combined_num == num:
                            break
                        else:
                            num = combined_num
                        right_bound = j + 1
                    final_num = self.word2num(" ".join(tokens[left_bound:right_bound]))
                    for i in range(left_bound, right_bound):
                        tokens[i] = ""
                    tokens[idx] = str(final_num) + "|||W2N"
            sent_intermediate = []
            for t in tokens:
                if t != "":
                    sent_intermediate.append(t)
            tokens = sent_intermediate
            for idx, t in enumerate(tokens):
                if t == "":
                    continue
                from_w2n = False
                if t.endswith("|||W2N"):
                    t = t.split("|||")[0]
                    from_w2n = True
                if self.num(t) is not None and from_w2n:
                    r1_token_from_w2n = False
                    if idx < len(tokens) - 1:
                        r1_token = tokens[idx + 1]
                        if r1_token.endswith("|||W2N"):
                            r1_token = r1_token.split("|||")[0]
                            r1_token_from_w2n = True
                    else:
                        r1_token = ""
                    r2_token_from_w2n = False
                    if idx < len(tokens) - 2:
                        r2_token = tokens[idx + 2]
                        if r2_token.endswith("|||W2N"):
                            r2_token = r2_token.split("|||")[0]
                            r2_token_from_w2n = True
                    else:
                        r2_token = ""
                    if r1_token == "and" and self.num(r2_token) is not None and r2_token_from_w2n:
                        tokens[idx + 2] = str(self.num(t) + self.num(r2_token))
                        tokens[idx + 1] = ""
                        tokens[idx] = ""
                    if self.num(r1_token) is not None and r1_token_from_w2n:
                        tokens[idx + 1] = str(self.num(t) + self.num(r1_token))
                        tokens[idx] = ""
            ret_sent = []
            for t in tokens:
                if t == "":
                    continue
                if t.endswith("|||W2N"):
                    t = t.split("|||")[0]
                ret_sent.append(t)
            if modified:
                ret_sents.append(" ".join(ret_sent))
                ret_sents_original.append(" ".join(sent["tokens"].split()))

        assert(len(ret_sents) == len(ret_sents_original))

        return ret_sents, ret_sents_original


def gen_all_data():
    processor = TextProcessor()
    f_out = open("./samples/wiki-processed-full.txt", 'w')
    rootDir = "/Users/xuanyuzhou/Downloads/enwiki-out"
    file_counter = 0
    for dirName, subdirList, fileList in os.walk(rootDir):
        for subdir in subdirList:
            for dirName, subdirList, fileList in os.walk(rootDir + "/" + subdir):
                for file in fileList:
                    file_counter += 1
                    print("Processing file " + str(file_counter))
                    cur_dir = rootDir + "/" + subdir + "/" + file
                    f_cur = open(cur_dir)
                    lines = [x.strip() for x in f_cur.readlines()]
                    for line in lines:
                        if line.startswith("<doc") or line.startswith("</doc>"):
                            continue
                        if len(line.split()) < 5:
                            continue
                        try:
                            sents, original_sents = processor.process_text(line)
                        except Exception as e:
                            sents = []
                            original_sents = []
                        for idx, s in enumerate(sents):
                            f_out.write(s + "\n")


def split_all_data():
    f = open("samples/wiki-25/wiki-processed-25.txt")
    f_train = open("samples/wiki-25/train.txt", "w")
    f_dev = open("samples/wiki-25/dev.txt", "w")
    f_eval = open("samples/wiki-25/eval.txt", "w")
    lines = [x.strip() for x in f.readlines()]
    for line in lines:
        rg = random.random()
        if rg < 0.45:
            f_train.write(line.lower() + "\n")
        elif rg < 0.5:
            f_dev.write(line.lower() + "\n")
        else:
            f_eval.write(line.lower() + "\n")


split_all_data()
