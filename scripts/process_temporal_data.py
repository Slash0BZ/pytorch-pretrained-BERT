import re
import os
import jsonlines
import pickle
import random
from word2number import w2n
from ccg_nlpy import local_pipeline
import matplotlib.pyplot as plt
# from allennlp import predictors
# from allennlp.predictors import Predictor
# from allennlp.models.archival import load_archive


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
        ]
        invalid_prev_tokens = [
            "after",
            "within",

        ]
        invalid_next_two_tokens = [
            "from now",
        ]

        with open(path) as f:
            lines = [x.strip() for x in f.readlines()]

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

                    next_two_tokens = tokens[min(len(tokens) - 1, i + 1)] + " " + tokens[min(len(tokens) - 1, i + 2)]
                    if next_two_tokens.lower() in invalid_next_two_tokens:
                        valid = False

            if valid:
                f_out.write(lines[li] + '\n')

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
        lines = [x.strip() for x in open("samples/duration/all/all.txt").readlines()]
        f_nom = open("samples/duration/all/nominals.txt", "w")
        f_verb = open("samples/duration/all/verbs.txt", "w")
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
                    position = i + 2 - 3
                    form_validation = tokens[min(len(tokens) - 1, i + 2)]
                    label_string = str(self.quantity(tokens[i - 1])) + " " + t
                    if tokens[max(0, i - 2)].lower() == "than" and tokens[max(0, i - 3)].lower() == "more":
                        position = position - 2
                        set_blank_start = i - 3
                    if tokens[max(0, i - 2)].lower() == "the":
                        position = position - 1
                        set_blank_start = i - 2
                    prev_list = ["first", "last", "final"]
                    if tokens[max(0, i - 2)].lower() in prev_list:
                        valid = False
            for i in range(set_blank_start, set_blank_end):
                tokens[i] = ""
            new_tokens = []
            for t in tokens:
                if t != "":
                    new_tokens.append(t)
            tokens = new_tokens

            if position >= len(tokens):
                continue

            if tokens[position] != form_validation:
                valid = False

            if valid:
                f_out.write(' '.join(tokens) + "\t" + str(position) + "\t" + label_string + "\n")


# class PretrainedModel:
#     """
#     A pretrained model is determined by both an archive file
#     (representing the trained model)
#     and a choice of predictor.
#     """
#     def __init__(self, archive_file: str, predictor_name: str) -> None:
#         self.archive_file = archive_file
#         self.predictor_name = predictor_name
#
#     def predictor(self) -> Predictor:
#         archive = load_archive(self.archive_file)
#         return Predictor.from_archive(archive, self.predictor_name)
#
#
# class AllenSRL:
#
#     def __init__(self):
#         model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz',
#                                 'semantic-role-labeling')
#         self.predictor = model.predictor()
#
#     def predict_batch(self, sentences):
#         for sentence in sentences:
#             prediction = self.predictor.predict(sentence)
#             print(prediction)
#
#     def predict_file(self, path):
#         with open(path) as f:
#             lines = [x.strip() for x in f.readlines()]
#         self.predict_batch(lines)


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

    def parse_srl_file(self, path):
        lines = [x.strip() for x in open(path).readlines()]
        reader = jsonlines.Reader(lines)

        f_out_s = jsonlines.open("samples/duration/duration_srl_succeed.jsonl", mode="w")
        f_out_f = jsonlines.open("samples/duration/duration_srl_fail.jsonl", mode="w")
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
        }

    def get_seconds_from_timex(self, timex):
        quantity = float(timex.split()[0])
        unit_val = self.convert_map[timex.split()[1].lower()]
        return quantity * unit_val

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
        plt.hist(cur_map[key], bins=400, log=True, range=(0, 3153600000))
        plt.show()


if __name__ == "__main__":
    extractor = GigawordExtractor()
    # extractor.prepare_nom_data("samples/duration/all/nominals.txt", "samples/duration/all/nominals_formatted.txt")
    # extractor.split_nominals()
    # extractor.read_file("/Users/xuanyuzhou/Downloads/tmp/apw_eng/apw_eng_199411")
    extractor.process_path("/Users/xuanyuzhou/Downloads/tmp/all", duration_path="samples/duration_more/duration_all.txt")
    extractor.process_duration_initial_filter("samples/duration_more/duration_all.txt", "samples/duration_more/duration_all_filtered.txt")

    # srl = AllenSRL()
    # srl.predict_file("samples/duration_afp_eng_filtered.txt")

    # runner = SRLRunner()
    # runner.print_file("samples/duration/duration_srl_fail.jsonl")

    # baseline = VerbBaseline("samples/duration/all/verbs.txt")
    # baseline.process("samples/duration/all/nearest_verb_cont/partition_")
    # VerbBaseline.merge_map("samples/duration/all/nearest_verb_all/partition_", "samples/duration/all/nearest_verb.pkl")
    # VerbBaseline.exp_output("samples/duration/all/nearest_verb.pkl", "talk")
