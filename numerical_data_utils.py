import os
import re
import pickle

from word2number import w2n
from ccg_nlpy import local_pipeline
from sklearn.preprocessing import normalize
import numpy as np


class ParagraphConverter:

    def __init__(self):
        self.pipeline = local_pipeline.LocalPipeline()
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

        self.inverse_map = {
            "second": "second",
            "seconds": "second",
            "minute": "minute",
            "minutes": "minute",
            "hour": "hour",
            "hours": "hour",
            "day": "day",
            "days": "day",
            "week": "week",
            "weeks": "week",
            "month": "month",
            "months": "month",
            "year": "year",
            "years": "year",
            "century": "century",
            "centuries": "century",
        }
        with open("unit_values.pkl", "rb") as f:
            self.unit_vals = pickle.load(f)

    @staticmethod
    def num(s):
        try:
            return float(s)
        except:
            if s == 'a' or s == 'an':
                return None
            if s == 'few':
                return 3.0
            return None

    @staticmethod
    def quantity(s):
        try:
            cur = w2n.word_to_num(s)
            if cur is not None:
                return float(cur)
            return None
        except:
            return None

    @staticmethod
    def combine_timex(tokens):
        modified_tokens = []
        cont = 0
        for idx, token in enumerate(tokens[0:]):
            if cont > 0:
                cont -= 1
                continue
            appended = False
            if 0 < idx < len(tokens) - 2:
                if token == "and" and tokens[idx - 1] == "seconds" and tokens[idx + 2] == "seconds":
                    num_left = ParagraphConverter.num(tokens[idx - 2])
                    num_right = ParagraphConverter.num(tokens[idx + 1])
                    if num_left is None or num_right is None:
                        continue
                    num_total = num_left + num_right
                    modified_tokens.pop()
                    modified_tokens.pop()
                    modified_tokens.append(str(num_total))
                    modified_tokens.append("seconds")
                    cont = 2
                    appended = True
            if not appended:
                modified_tokens.append(token)
        return modified_tokens

    """
    Expect lowered cased tokens
    """
    def convert_sentence(self, tokens):
        modified_tokens = []
        for idx, token in enumerate(tokens):
            appended = False
            if token in self.convert_map:
                prev = max(0, idx - 1)
                value = ParagraphConverter.num(tokens[prev])
                num_position = idx - 1
                if value is None:
                    for minus in range(-10, 0):
                        min_start = max(0, idx + minus)
                        quantity_list = tokens[min_start:idx]
                        if len(quantity_list) == 0:
                            break
                        quantity_str = ""
                        for q in quantity_list:
                            quantity_str += q + " "
                        if len(quantity_str) > 0:
                            quantity_str = quantity_str[:-1]
                        quantity = ParagraphConverter.quantity(quantity_str)
                        if quantity is None:
                            break
                        if value != quantity and value is not None:
                            break
                        value = quantity
                        num_position = min_start
                if value is not None:
                    for i in range(0, idx - num_position):
                        modified_tokens.pop()
                    value *= self.convert_map[token]
                    modified_tokens.append(str(value))
                    modified_tokens.append("seconds")
                    appended = True
            if not appended:
                modified_tokens.append(token)
        return ParagraphConverter.combine_timex(modified_tokens)

    def convert_sentence_with_unit(self, tokens):
        modified_tokens = []
        for idx, token in enumerate(tokens):
            appended = False
            if token in self.convert_map:
                prev = max(0, idx - 1)
                value = ParagraphConverter.num(tokens[prev])
                num_position = idx - 1
                if value is None:
                    for minus in range(-10, 0):
                        min_start = max(0, idx + minus)
                        quantity_list = tokens[min_start:idx]
                        if len(quantity_list) == 0:
                            break
                        quantity_str = ""
                        for q in quantity_list:
                            quantity_str += q + " "
                        if len(quantity_str) > 0:
                            quantity_str = quantity_str[:-1]
                        quantity = ParagraphConverter.quantity(quantity_str)
                        if quantity is None:
                            break
                        if value != quantity and value is not None:
                            break
                        value = quantity
                        num_position = min_start
                if value is not None:
                    mean, std = self.unit_vals[self.inverse_map[token]]
                    max_value = mean + 10.0 * std
                    value = min(max_value, value)
                    value = max(value, 0.0)
                    value = value / max_value
                    for i in range(0, idx - num_position):
                        modified_tokens.pop()
                    modified_tokens.append(str(value))
                    modified_tokens.append(token)
                    appended = True
            if not appended:
                modified_tokens.append(token)
        if modified_tokens == tokens:
            # pass
            return None
        return modified_tokens

    def process_paragraph(self, paragraph):
        ret = []
        paragraph = paragraph.lower()
        ta = self.pipeline.doc(paragraph)
        sent_view = ta.get_view("SENTENCE")
        for sent in sent_view:
            processed_tokens = self.convert_sentence_with_unit(sent['tokens'].split())
            if processed_tokens is None:
                continue
            combined_str = ""
            for t in processed_tokens:
                combined_str += t + " "
            if len(combined_str) > 0:
                combined_str = combined_str[:-1]
                ret.append(combined_str)
        return ret

    def convert_test(self, dummy_str):
        tokens = dummy_str.split()
        print(str(tokens))
        print(str(self.convert_sentence(tokens)))
        print()


class GigawordExtractor:

    def __init__(self, path):
        self.converter = ParagraphConverter()
        self.path = path
        self.processed_docs = []

    @staticmethod
    def read_file(file_name):
        with open(file_name) as f:
            lines = f.readlines()
        content = ' '.join(line.strip() for line in lines)
        return re.findall(r'<P>(.+?)</P>', content)

    def run(self):
        for root, dirs, files in os.walk(self.path):
            files.sort()
            for file in files:
                paragraphs = GigawordExtractor.read_file(self.path + '/' + file)
                doc_list = []
                for p in paragraphs:
                    cur_list = []
                    try:
                        cur_list = self.converter.process_paragraph(p)
                    except Exception as e:
                        print(e)
                    doc_list += cur_list
                self.processed_docs.append(doc_list)

    def output_to_file(self, file_path):
        f = open(file_path, "w")
        self.run()
        for doc in self.processed_docs:
            for line in doc:
                f.write(line + "\n")
            f.write("\n")


class LineExtractor:

    def __init__(self, path):
        self.converter = ParagraphConverter()
        self.path = path
        self.processed_docs = []

    def run(self):
        f = open(self.path, "r")
        lines = [x.strip() for x in f.readlines()]
        for l in lines:
            premise = l.split("\t")[0]
            answer = l.split("\t")[1]
            label = l.split("\t")[2]
            try:
                premise_concat = ""
                premise = self.converter.process_paragraph(premise)
                for p in premise:
                    premise_concat += p + " "
                if premise_concat.endswith(" "):
                    premise_concat = premise_concat[:-1]
                premise = premise_concat

                answer_concat = ""
                answer = self.converter.process_paragraph(answer)
                for a in answer:
                    answer_concat += a + " "
                if answer_concat.endswith(" "):
                    answer_concat = answer_concat[:-1]
                answer = answer_concat

            except Exception as e:
                print(e)
            self.processed_docs.append(premise + "\t" + answer + "\t" + label)

    def output_to_file(self, file_path):
        f = open(file_path, "w")
        self.run()
        for doc in self.processed_docs:
            f.write(doc + "\n")


class GigawordUnitStat:

    def __init__(self, path):
        self.pipeline = local_pipeline.LocalPipeline()
        self.unit_file = "./units.txt"
        self.path = path
        self.name_map = {}
        self.inverse_name_map = {}
        self.distribution_map = {}
        self.read_units()

    @staticmethod
    def insert_map(m, k, v):
        if k not in m:
            m[k] = []
        m[k].append(v)

    def read_units(self):
        f = open(self.unit_file)
        lines = [x.strip() for x in f.readlines()]
        for line in lines:
            key = line.split(":")[0]
            vals = line.split(":")[1].split("/")
            for v in vals:
                GigawordUnitStat.insert_map(self.name_map, key, v)
                self.inverse_name_map[v] = key

    def run(self):
        for root, dirs, files in os.walk(self.path):
            files.sort()
            for file in files:
                paragraphs = GigawordExtractor.read_file(self.path + '/' + file)
                for p in paragraphs:
                    try:
                        self.process_paragraph(p)
                    except Exception as e:
                        print(e)

    def process_paragraph(self, paragraph):
        paragraph = paragraph.lower()
        ta = self.pipeline.doc(paragraph)
        sent_view = ta.get_view("SENTENCE")
        for sent in sent_view:
            tokens = sent['tokens'].split()
            for idx, token in enumerate(tokens):
                prev_token_idx = max(0, idx - 1)
                prev_num = ParagraphConverter.num(tokens[prev_token_idx])
                if token in self.inverse_name_map and prev_num is not None:
                    self.insert_map(self.distribution_map, self.inverse_name_map[token], prev_num)

    def save(self, path):
        with open(path, "wb") as sf:
            pickle.dump(self.distribution_map, sf)


def pc_test():
    converter = ParagraphConverter()
    strs = [""] * 6
    strs[0] = "I have spent 3 days here."
    strs[1] = "I spent three hours here."
    strs[2] = "I spent thirty one hours here."
    strs[3] = "I spent one hour and thirty one minutes here."
    strs[4] = "minutes after the explosion."
    strs[5] = "I spent an hour here."
    for s in strs:
        converter.convert_test(s)
    paragraph = ""
    for s in strs:
        paragraph += s + " "
    print(converter.process_paragraph(paragraph))


class UnitAnalyzer:

    def __init__(self, data_path):
        with open(data_path, "rb") as df:
            self.db = pickle.load(df)

    def analyze(self, out_path):
        out_map = {}
        for key in self.db:
            vals = np.array(self.db[key])
            out_map[key] = [np.mean(vals), np.std(vals)]
            print(key)
            print(out_map[key][0])
            print(out_map[key][1])
            print()
        with open(out_path, "wb") as of:
            pickle.dump(out_map, of)

extractor = GigawordExtractor("/Users/xuanyuzhou/Downloads/tmp/afp_eng")
extractor.output_to_file("samples/gigaword_ultra_normalized_01.txt")

# extractor = LineExtractor("samples/temporal_data_split/test_vanilla.txt")
# extractor.output_to_file("samples/temporal_data_split/test_normalized.txt")

# unit_extractor = GigawordUnitStat("/Users/xuanyuzhou/Downloads/tmp/2doc")
# unit_extractor.run()
# unit_extractor.save("unit_output_no_one.pkl")

# analyzer = UnitAnalyzer("./unit_output.pkl")
# analyzer.analyze("./unit_values.pkl")


