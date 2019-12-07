import jsonlines
import os
import random
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer
from word2number import w2n
import copy
from spacy.lang.en import English
import re
import nltk
# nltk.download('wordnet', download_dir="/shared/xzhou45/nltk_data/")
nltk.data.path.append("/shared/xzhou45/nltk_data/")
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

class AdditionalPatternProcessor:

    def __init__(self):
        # self.output_path = "samples/gigaword/tmparg_collection_with_prev_and_next.txt"
        # self.f_out = open(self.output_path, "w")
        self.root_path = "samples/gigaword"
        self.context_table = {}
        # self.build_context_table()
        self.value_map = {
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
            "decade": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "decades": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "century": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "dawn": [1, 0],
            "morning": [1, 1],
            "noon": [1, 2],
            "afternoon": [1, 3],
            "evening": [1, 4],
            "dusk": [1, 5],
            "night": [1, 6],
            "midnight": [1, 7],
        }
        self.keywords = {
            "dawns": [1, 0],
            "mornings": [1, 1],
            "noons": [1, 2],
            "afternoons": [1, 3],
            "evenings": [1, 4],
            "dusks": [1, 5],
            "nights": [1, 6],
            "midnights": [1, 7],
            "dawn": [1, 0],
            "morning": [1, 1],
            "noon": [1, 2],
            "afternoon": [1, 3],
            "evening": [1, 4],
            "dusk": [1, 5],
            "night": [1, 6],
            "midnight": [1, 7],
            "monday": [2, 0],
            "tuesday": [2, 1],
            "wednesday": [2, 2],
            "thursday": [2, 3],
            "friday": [2, 4],
            "saturday": [2, 5],
            "sunday": [2, 6],
            "mondays": [2, 0],
            "tuesdays": [2, 1],
            "wednesdays": [2, 2],
            "thursdays": [2, 3],
            "fridays": [2, 4],
            "saturdays": [2, 5],
            "sundays": [2, 6],
            "january": [3, 0],
            "february": [3, 1],
            "march": [3, 2],
            "april": [3, 3],
            "may": [3, 4],
            "june": [3, 5],
            "july": [3, 6],
            "august": [3, 7],
            "september": [3, 8],
            "october": [3, 9],
            "november": [3, 10],
            "december": [3, 11],
            "januarys": [3, 0],
            "januaries": [3, 0],
            "februarys": [3, 1],
            "februaries": [3, 1],
            "marches": [3, 2],
            "marchs": [3, 2],
            "aprils": [3, 3],
            "mays": [3, 4],
            "junes": [3, 5],
            "julys": [3, 6],
            "julies": [3, 6],
            "augusts": [3, 7],
            "septembers": [3, 8],
            "octobers": [3, 9],
            "novembers": [3, 10],
            "decembers": [3, 11],
            "springs": [4, 0],
            "summers": [4, 1],
            "autumns": [4, 2],
            "falls": [4, 2],
            "winters": [4, 3],
            "spring": [4, 0],
            "summer": [4, 1],
            "autumn": [4, 2],
            "fall": [4, 2],
            "winter": [4, 3],
        }
        self.process()

    def build_context_table(self):
        lines = [x.strip() for x in open("samples/gigaword/raw_collection_contextsent_tokenized.txt").readlines()]
        print("Loaded all lines")
        for i, line in enumerate(lines):
            if i % 1000000 == 0:
                print("Added " + str(i) + " sentences to context table.")
            key = line.split("\t")[1].replace(" ", "")
            if len(line.split("\t")) < 3:
                continue
            self.context_table[key] = (line.split("\t")[0], line.split("\t")[2])
        print("Finished building context table")

    def get_arg1_range(self, tags):
        rets = []
        start = -1
        end = -1
        for i, t in enumerate(tags):
            if t == "B-ARG1":
                if start > -1:
                    rets.append((start, end))
                start = i
                end = i + 1
            if t == "I-ARG1":
                end += 1
        if start > -1:
            rets.append((start, end))
        return rets

    def get_tmp_arg_range(self, tags):
        rets = []
        start = -1
        end = -1
        for i, t in enumerate(tags):
            if t == "B-ARGM-TMP":
                if start > -1:
                    rets.append((start, end))
                start = i
                end = i + 1
            if t == "I-ARGM-TMP":
                end += 1
        if start > -1:
            rets.append((start, end))
        return rets

    def get_verb_pos(self, tags):
        for i, t in enumerate(tags):
            if t == "B-V":
                return i
        return -1

    def get_stripped(self, tokens, tags, verb_pos, tmp_start, tmp_end):
        new_tokens = []
        new_verb_pos = -1
        new_tmp_start = -1
        new_tmp_end = -1
        for i in range(0, len(tokens)):
            if tags[i] != "O":
                new_tokens.append(tokens[i])
            if i == verb_pos:
                new_verb_pos = len(new_tokens) - 1
            if i == tmp_start:
                new_tmp_start = len(new_tokens) - 1
            if i == tmp_end - 1:
                new_tmp_end = len(new_tokens)
        return new_tokens, new_verb_pos, new_tmp_start, new_tmp_end

    def process_single_file(self, file_name):
        lines = [x.strip() for x in open(file_name).readlines()]
        reader = jsonlines.Reader(lines)
        counter = 0
        for obj_list in reader:
            for obj in obj_list:
                key = "".join(obj['words'])
                # if key not in self.context_table:
                #     continue
                valid = False
                verb_pos_final = -1
                tmp_string = ""
                for verb in obj['verbs']:
                    arg1s = self.get_arg1_range(verb['tags'])
                    tmps = self.get_tmp_arg_range(verb['tags'])
                    verb_pos = self.get_verb_pos(verb['tags'])
                    tokens = obj['words']
                    for tmp_start, tmp_end in tmps:
                        if tokens[tmp_start] in ["for", "over"]:
                            for i, t in enumerate(tokens[tmp_start:tmp_end]):
                                if t.lower() in self.keywords and tokens[tmp_start + i - 1].lower() in ["entire", "whole", "all"]:
                                    valid = True
                                    verb_pos_final = verb_pos
                                    tmp_string = " ".join(tokens[tmp_start:tmp_end])
                                    break
                #     if verb in ['spends', "spend", "spent", "spending"]:
                #         for arg_start, arg_end in arg1s:
                #             for i in range(arg_start, arg_end):
                #                 t = obj['words'][i].lower()
                #                 if t in self.value_map:
                #                     valid = True
                #                     verb_pos_final = verb_pos
                if valid:
                    tokens = obj['words']
                    tokens[verb_pos_final] = "[[" + tokens[verb_pos_final] + "]]"
                    counter += 1
                    print(" ".join(tokens))
                    print(tmp_string)
        return counter

    def process(self):
        total_count = 0
        for dirName, subdirList, fileList in os.walk(self.root_path):
            for fname in fileList:
                if fname.startswith("srl"):
                    total_count += self.process_single_file(self.root_path + "/" + fname)
                    print("Finished " + fname)
        print(total_count)


class TmpArgProcessor:

    def __init__(self):
        self.output_path = "samples/gigaword/tmparg_collection_realnews.txt"
        self.f_out = open(self.output_path, "w")
        self.root_path = "samples/gigaword"
        self.context_table = {}
        self.build_context_table()
        self.process()

    def build_context_table(self):
        lines = [x.strip() for x in open("samples/gigaword/raw_collection_realnews_tokenized.txt").readlines()]
        print("Loaded all lines")
        for i, line in enumerate(lines):
            if i % 1000000 == 0:
                print("Added " + str(i) + " sentences to context table.")
            if len(line.split("\t")) < 3:
                continue
            key = line.split("\t")[1].replace(" ", "")
            self.context_table[key] = (line.split("\t")[0], line.split("\t")[2])
        print("Finished building context table")

    def get_tmp_arg_range(self, tags):
        rets = []
        start = -1
        end = -1
        for i, t in enumerate(tags):
            if t == "B-ARGM-TMP":
                if start > -1:
                    rets.append((start, end))
                start = i
                end = i + 1
            if t == "I-ARGM-TMP":
                end += 1
        if start > -1:
            rets.append((start, end))
        return rets

    def get_verb_pos(self, tags):
        for i, t in enumerate(tags):
            if t == "B-V":
                return i
        return -1

    def get_stripped(self, tokens, tags, verb_pos, tmp_start, tmp_end):
        new_tokens = []
        new_verb_pos = -1
        new_tmp_start = -1
        new_tmp_end = -1
        for i in range(0, len(tokens)):
            if tags[i] != "O":
                new_tokens.append(tokens[i])
            if i == verb_pos:
                new_verb_pos = len(new_tokens) - 1
            if i == tmp_start:
                new_tmp_start = len(new_tokens) - 1
            if i == tmp_end - 1:
                new_tmp_end = len(new_tokens)
        return new_tokens, new_verb_pos, new_tmp_start, new_tmp_end

    def process_single_file(self, file_name):
        lines = [x.strip() for x in open(file_name).readlines()]
        reader = jsonlines.Reader(lines)
        for obj_list in reader:
            for obj in obj_list:
                key = "".join(obj['words'])
                if key not in self.context_table:
                    continue
                for verb in obj['verbs']:
                    tmps = self.get_tmp_arg_range(verb['tags'])
                    if len(tmps) == 0:
                        continue
                    verb_pos = self.get_verb_pos(verb['tags'])
                    left_sent, right_sent = self.context_table[key]
                    for tmp_start, tmp_end in tmps:
                        # """CHANGING TO SRL ONLY!"""
                        # srl_tokens, srl_verb_pos, srl_tmp_start, srl_tmp_end = self.get_stripped(obj['words'], verb['tags'], verb_pos, tmp_start, tmp_end)
                        # self.f_out.write(" ".join(srl_tokens) + "\t" + left_sent + "\t" + right_sent + "\t" + str(srl_verb_pos) + "\t" + str(srl_tmp_start) + "\t" + str(srl_tmp_end) + "\n")
                        self.f_out.write(" ".join(obj['words']) + "\t" + left_sent + "\t" + right_sent + "\t" + str(verb_pos) + "\t" + str(tmp_start) + "\t" + str(tmp_end) + "\n")

    def process(self):
        for dirName, subdirList, fileList in os.walk(self.root_path):
            for fname in fileList:
                if fname.startswith("srl_realnews"):
                    self.process_single_file(self.root_path + "/" + fname)
                    print("Finished " + fname)


class TmpArgDimensionFilter:
    def __init__(self):
        # self.file_path = "samples/gigaword/tmparg_collection_all.txt"
        self.file_path = "samples/gigaword/tmparg_collection_realnews.txt"
        ## Required
        self.lines = list(set([x.strip() for x in open(self.file_path).readlines()]))
        # self.rand_file_path = "samples/gigaword/raw_collection_additional_tokenized.txt"
        # self.rand_lines = [x.strip() for x in open(self.rand_file_path).readlines()]
        self.output_path = "samples/gigaword/twosent_small_lesstyp"
        self.output_path_singlesent = "samples/gigaword/onesent_small_lesstyp"
        self.value_map = {
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
            "decade": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "decades": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "century": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
        }
        self.process_new_forms()

    def get_trivial_floats(self, s):
        try:
            n = float(s)
            return n
        except:
            return None

    def get_surface_floats(self, tokens):
        if tokens[-1] in ["a", "an"]:
            return 1.0
        if tokens[-1] == "several":
            return 4.0
        if tokens[-1] == "many":
            return 10.0
        if tokens[-1] == "some":
            return 3.0
        if tokens[-1] == "few":
            return 3.0
        if tokens[-1] == "tens" or " ".join(tokens[-2:]) == "tens of":
            return 10.0
        if tokens[-1] == "hundreds" or " ".join(tokens[-2:]) == "hundreds of":
            return 100.0
        if tokens[-1] == "thousands" or " ".join(tokens[-2:]) == "thousands of":
            return 1000.0
        if " ".join(tokens[-2:]) in ["a few", "a couple"]:
            return 3.0
        if " ".join(tokens[-3:]) == "a couple of":
            return 2.0
        return None

    def quantity(self, tokens):
        try:
            if self.get_trivial_floats(tokens[-1]) is not None:
                return self.get_trivial_floats(tokens[-1])
            if self.get_surface_floats(tokens) is not None:
                return self.get_surface_floats(tokens)
            string_comb = tokens[-1]
            cur = w2n.word_to_num(string_comb)
            for i in range(-2, -1, -1):
                if tokens[i] in ["-", "and"] or w2n.word_to_num(tokens[i]) is not None:
                    string_comb = tokens[i] + " " + string_comb
                    update = w2n.word_to_num(string_comb)
                    if update is not None:
                        cur = update
                else:
                    break
            if cur is not None:
                return float(cur)
        except:
            return None

    def order_number_convert(self, input):
        m = {"first": 1.0, "second": 2.0, "third": 3.0, "fourth": 4.0, "fifth": 5.0, "sixth": 6.0, "seventh": 7.0,
             "eighth": 8.0, "ninth": 9.0, "tenth": 10.0}
        if input in m:
            return m[input]
        return 0.0

    def transform_plural(self, unit):
        transform_map = {
            "second": "seconds",
            "seconds": "seconds",
            "minute": "minutes",
            "minutes": "minutes",
            "hour": "hours",
            "hours": "hours",
            "day": "days",
            "days": "days",
            "week": "weeks",
            "weeks": "weeks",
            "month": "months",
            "months": "months",
            "year": "years",
            "years": "years",
            "decade": "decades",
            "decades": "decades",
            "century": "centuries",
            "centuries": "centuries",
        }
        if unit in transform_map:
            return transform_map[unit]
        return unit

    """
    Requires tokens to be lower cased
    """
    def check_duration_sentences(self, tmparg_tokens):
        unit = ""
        num = -1.0
        for i, token in enumerate(tmparg_tokens):
            if token in self.value_map:
                num_args = []
                for t in tmparg_tokens[0:i]:
                    num_args.append(t)
                num = self.quantity(num_args)
                unit = self.transform_plural(token)
                if num is None:
                    if unit == token:
                        num = 4.0
                    else:
                        num = 1.0
                break
        if unit == "":
            return "NO_UNIT_FOUND"
        ret_str = str(num) + " " + unit
        if "for a second" in " ".join(tmparg_tokens):
            if tmparg_tokens[-1] != "second":
                return "NO_UNIT_FOUND"
        for t in tmparg_tokens:
            if self.order_number_convert(t) > 0.0:
                return "FOUND_UNIT_BUT_NOT_DURATION"
        if tmparg_tokens[0] in ["for"] and "second time" not in " ".join(tmparg_tokens) and "for the second" not in " ".join(tmparg_tokens):
            return ret_str
        return "FOUND_UNIT_BUT_NOT_DURATION"

    def check_frequency_sentences(self, tmparg_tokens):
        unit = ""
        num = -1.0
        quantity_stop = -1
        for i, token in enumerate(tmparg_tokens):
            if token in self.value_map:
                num_args = []
                for t in tmparg_tokens[0:i]:
                    num_args.append(t)
                num = self.quantity(num_args)
                unit = self.transform_plural(token)
                quantity_stop = i
                if num is None:
                    if unit == token:
                        num = 4.0
                    else:
                        num = 1.0
                break
        if unit == "":
            return "NO_UNIT_FOUND"
        valid = False
        start_anchor = max(0, quantity_stop - 5)
        for i, token in enumerate(tmparg_tokens[start_anchor:quantity_stop]):
            if token == "every" or token == "once" or token == "per" or token == "each":
                num /= 1.0
                valid = True
            if token == "twice":
                num /= 2.0
                valid = True
            if token == "times":
                div = self.quantity([tmparg_tokens[start_anchor:quantity_stop][i-1]])
                if div is not None and div > 0.0:
                    num /= div
                    valid = True
            if token == "time":
                num_key = tmparg_tokens[start_anchor:quantity_stop][i-1]
                converted_num_key = self.order_number_convert(num_key)
                if converted_num_key > 0.0:
                    num /= converted_num_key
                    valid = True
                else:
                    return "SKIP_DURATION"

        if tmparg_tokens[0] == "when":
            valid = False
        ret_str = "FOUND_UNIT_BUT_NOT_FREQUENCY"
        if valid:
            ret_str = str(num) + " " + unit
        return ret_str

    def check_typical_sentences(self, tmparg_tokens):
        keywords = {
            "dawns": [1, 0],
            "mornings": [1, 1],
            "noons": [1, 2],
            "afternoons": [1, 3],
            "evenings": [1, 4],
            "dusks": [1, 5],
            "nights": [1, 6],
            "midnights": [1, 7],
            "dawn": [1, 0],
            "morning": [1, 1],
            "noon": [1, 2],
            "afternoon": [1, 3],
            "evening": [1, 4],
            "dusk": [1, 5],
            "night": [1, 6],
            "midnight": [1, 7],
            "monday": [2, 0],
            "tuesday": [2, 1],
            "wednesday": [2, 2],
            "thursday": [2, 3],
            "friday": [2, 4],
            "saturday": [2, 5],
            "sunday": [2, 6],
            "mondays": [2, 0],
            "tuesdays": [2, 1],
            "wednesdays": [2, 2],
            "thursdays": [2, 3],
            "fridays": [2, 4],
            "saturdays": [2, 5],
            "sundays": [2, 6],
            "january": [3, 0],
            "february": [3, 1],
            "march": [3, 2],
            "april": [3, 3],
            "may": [3, 4],
            "june": [3, 5],
            "july": [3, 6],
            "august": [3, 7],
            "september": [3, 8],
            "october": [3, 9],
            "november": [3, 10],
            "december": [3, 11],
            "januarys": [3, 0],
            "januaries": [3, 0],
            "februarys": [3, 1],
            "februaries": [3, 1],
            "marches": [3, 2],
            "marchs": [3, 2],
            "aprils": [3, 3],
            "mays": [3, 4],
            "junes": [3, 5],
            "julys": [3, 6],
            "julies": [3, 6],
            "augusts": [3, 7],
            "septembers": [3, 8],
            "octobers": [3, 9],
            "novembers": [3, 10],
            "decembers": [3, 11],
            "springs": [4, 0],
            "summers": [4, 1],
            "autumns": [4, 2],
            "falls": [4, 2],
            "winters": [4, 3],
            "spring": [4, 0],
            "summer": [4, 1],
            "autumn": [4, 2],
            "fall": [4, 2],
            "winter": [4, 3],
        }
        if tmparg_tokens[0].lower() in ["until", "when", "while", "during", "as", "since", "following", "after", "before"]:
            return "NO_TYPICAL_FOUND", ""
        ret_pairs = []
        for t in tmparg_tokens:
            if t in keywords:
                ret_pairs.append([t, keywords[t][0]])
        min_val = 10
        selected_t = None
        for t, group_num in ret_pairs:
            if group_num < min_val:
                min_val = group_num
                selected_t = t
        if len(ret_pairs) > 0:
            return selected_t, min_val
        return "NO_TYPICAL_FOUND", ""

    def check_ordering_sentences(self, tokens, tmp_start, tmp_end):
        unit = ""
        num = -1.0
        tmparg_tokens = tokens[tmp_start:tmp_end]
        for i, token in enumerate(tmparg_tokens):
            if token in self.value_map:
                num_args = []
                for t in tmparg_tokens[0:i]:
                    num_args.append(t)
                num = self.quantity(num_args)
                unit = self.transform_plural(token)
                if num is None:
                    if unit == token:
                        num = 4.0
                    else:
                        num = 1.0
                break

        if tmp_start > 0 or unit == "":
            return "NO_ORDERING_FOUND"
        if tmparg_tokens[0] in ["after", "later", "while", "during"] and tmparg_tokens[-1] in self.value_map:
            return str(num) + " " + unit
        if tmparg_tokens[-1] == "later":
            return str(num) + " " + unit

        return "NO_ORDERING_FOUND"

    def check_ordering_phrases(self, tmp_tokens):
        if len(tmp_tokens) < 5:
            return "NO_ORDERING_FOUND"
        if tmp_tokens[0] in ["after", "before", "while", "during", "when"]:
            return " ".join(tmp_tokens)
        return "NO_ORDERING_FOUND"

    def check_boundary_durations(self, tmp_tokens):
        if tmp_tokens[0] in ["today", "yesterday", "tomorrow"]:
            return "1.0 days"
        find_unit = False
        for i, t in enumerate(tmp_tokens):
            if t in ["next", "last", "following", "previous", "recent", "this", "earlier"]:
                find_unit = True
        if tmp_tokens[0] == "in":
            for t in tmp_tokens:
                match = re.match(r'.*([1-2][0-9]{3})', t)
                if match is not None:
                    return "1.0 years"
            find_unit = True
        if find_unit:
            unit = ""
            num = 1.0
            for i, token in enumerate(tmp_tokens):
                if token in self.value_map:
                    num_args = []
                    for t in tmp_tokens[0:i]:
                        num_args.append(t)
                    num = self.quantity(num_args)
                    unit = self.transform_plural(token)
                    if num is None:
                        if unit == token:
                            num = 4.0
                        else:
                            num = 1.0
                    break
            if unit != "":
                return str(num) + " " + unit
        return "NO_BOUNDARY_FOUND"

    def normalize_timex(self, expression):

        u = expression.split()[1]
        v_input = float(expression.split()[0])

        if u in ["instantaneous", "forever"]:
            return u, str(1)

        convert_map = {
            "seconds": 1.0,
            "minutes": 60.0,
            "hours": 60.0 * 60.0,
            "days": 24.0 * 60.0 * 60.0,
            "weeks": 7.0 * 24.0 * 60.0 * 60.0,
            "months": 30.0 * 24.0 * 60.0 * 60.0,
            "years": 365.0 * 24.0 * 60.0 * 60.0,
            "decades": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
        }
        seconds = convert_map[u] * float(v_input)
        prev_unit = "seconds"
        for i, v in enumerate(convert_map):
            if seconds / convert_map[v] < 0.1:
                break
            prev_unit = v
        if prev_unit == "seconds" and seconds > 60.0:
            prev_unit = "centuries"

        return prev_unit

    def assign_round_soft_label(self, length, target):
        label_vector = [0.157, 0.001, 0.0]
        label_vector_rev = [0.0, 0.001, 0.157]
        ret_vec = [0.0] * length
        ret_vec[target] = 0.683
        for i in range(target + 1, target + 4):
            cur_target = i
            if i >= length:
                cur_target -= length
            ret_vec[cur_target] = label_vector[i - target - 1]
        for i in range(target - 1, target - 4, -1):
            cur_target = i
            ret_vec[cur_target] = max(ret_vec[cur_target], label_vector_rev[i - target + 3])
        return ret_vec

    def get_soft_labels(self, orig_label, tokenizer):
        markers = ["_dur", "_freq", "_bnd"]
        keywords = {
            "second" + markers[0]: [0, 0],
            "seconds" + markers[0]: [0, 0],
            "minute" + markers[0]: [0, 1],
            "minutes" + markers[0]: [0, 1],
            "hour" + markers[0]: [0, 2],
            "hours" + markers[0]: [0, 2],
            "day" + markers[0]: [0, 3],
            "days" + markers[0]: [0, 3],
            "week" + markers[0]: [0, 4],
            "weeks" + markers[0]: [0, 4],
            "month" + markers[0]: [0, 5],
            "months" + markers[0]: [0, 5],
            "year" + markers[0]: [0, 6],
            "years" + markers[0]: [0, 6],
            "decade" + markers[0]: [0, 7],
            "decades" + markers[0]: [0, 7],
            "century" + markers[0]: [0, 8],
            "centuries" + markers[0]: [0, 8],
            "dawns": [1, 0],
            "mornings": [1, 1],
            "noons": [1, 2],
            "afternoons": [1, 3],
            "evenings": [1, 4],
            "dusks": [1, 5],
            "nights": [1, 6],
            "midnights": [1, 7],
            "dawn": [1, 0],
            "morning": [1, 1],
            "noon": [1, 2],
            "afternoon": [1, 3],
            "evening": [1, 4],
            "dusk": [1, 5],
            "night": [1, 6],
            "midnight": [1, 7],
            "monday": [2, 0],
            "tuesday": [2, 1],
            "wednesday": [2, 2],
            "thursday": [2, 3],
            "friday": [2, 4],
            "saturday": [2, 5],
            "sunday": [2, 6],
            "mondays": [2, 0],
            "tuesdays": [2, 1],
            "wednesdays": [2, 2],
            "thursdays": [2, 3],
            "fridays": [2, 4],
            "saturdays": [2, 5],
            "sundays": [2, 6],
            "january": [3, 0],
            "february": [3, 1],
            "march": [3, 2],
            "april": [3, 3],
            "may": [3, 4],
            "june": [3, 5],
            "july": [3, 6],
            "august": [3, 7],
            "september": [3, 8],
            "october": [3, 9],
            "november": [3, 10],
            "december": [3, 11],
            "januarys": [3, 0],
            "januaries": [3, 0],
            "februarys": [3, 1],
            "februaries": [3, 1],
            "marches": [3, 2],
            "marchs": [3, 2],
            "aprils": [3, 3],
            "mays": [3, 4],
            "junes": [3, 5],
            "julys": [3, 6],
            "julies": [3, 6],
            "augusts": [3, 7],
            "septembers": [3, 8],
            "octobers": [3, 9],
            "novembers": [3, 10],
            "decembers": [3, 11],
            "springs": [4, 0],
            "summers": [4, 1],
            "falls": [4, 2],
            "autumns": [4, 2],
            "winters": [4, 3],
            "spring": [4, 0],
            "summer": [4, 1],
            "autumn": [4, 2],
            "fall": [4, 2],
            "winter": [4, 3],
            "after": [5, 0],
            "before": [5, 1],
            "while": [5, 2],
            "during": [5, 2],
            "when": [5, 3],
            "second" + markers[1]: [6, 0],
            "seconds" + markers[1]: [6, 0],
            "minute" + markers[1]: [6, 1],
            "minutes" + markers[1]: [6, 1],
            "hour" + markers[1]: [6, 2],
            "hours" + markers[1]: [6, 2],
            "day" + markers[1]: [6, 3],
            "days" + markers[1]: [6, 3],
            "week" + markers[1]: [6, 4],
            "weeks" + markers[1]: [6, 4],
            "month" + markers[1]: [6, 5],
            "months" + markers[1]: [6, 5],
            "year" + markers[1]: [6, 6],
            "years" + markers[1]: [6, 6],
            "decade" + markers[1]: [6, 7],
            "decades" + markers[1]: [6, 7],
            "century" + markers[1]: [6, 8],
            "centuries" + markers[1]: [6, 8],
            "second" + markers[2]: [7, 0],
            "seconds" + markers[2]: [7, 0],
            "minute" + markers[2]: [7, 1],
            "minutes" + markers[2]: [7, 1],
            "hour" + markers[2]: [7, 2],
            "hours" + markers[2]: [7, 2],
            "day" + markers[2]: [7, 3],
            "days" + markers[2]: [7, 3],
            "week" + markers[2]: [7, 4],
            "weeks" + markers[2]: [7, 4],
            "month" + markers[2]: [7, 5],
            "months" + markers[2]: [7, 5],
            "year" + markers[2]: [7, 6],
            "years" + markers[2]: [7, 6],
            "decade" + markers[2]: [7, 7],
            "decades" + markers[2]: [7, 7],
            "century" + markers[2]: [7, 8],
            "centuries" + markers[2]: [7, 8],
        }
        vocab_indices = {
            0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
            1: [10, 11, 12, 13, 14, 15, 16, 17],
            2: [18, 19, 20, 21, 22, 23, 24],
            3: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
            4: [37, 38, 39, 40],
            # Note: not in order
            5: [41, 42, 61, 62],
            6: [43, 44, 45, 46, 47, 48, 49, 50, 51],
            7: [52, 53, 54, 55, 56, 57, 58, 59, 60],
        }
        if orig_label not in keywords:
            print("ERROR WHEN MAPPING SOFT LABELS")

        label_group = keywords[orig_label][0]
        label_id = keywords[orig_label][1]
        soft_labels = [0.0] * len(vocab_indices[label_group])
        if label_group == 0:
            label_vector_map = [
                [0.7412392700826488, 0.24912843928090708, 0.009458406390501939, 0.00016606286478555588,
                 7.30646267660146e-06, 5.120999881870982e-07, 2.807187666675083e-09, 1.1281434169352893e-11,
                 2.2746998844715235e-14],
                [0.1966797321503831, 0.5851870686462161, 0.1966797321503831, 0.018764436502991182,
                 0.0023274596334087274, 0.0003541235975250493, 7.345990219746334e-06, 1.0063714849200163e-07,
                 6.917246071508814e-10],
                [0.006037846064281208, 0.15903304473396812, 0.47317575760473235, 0.2453125506936087,
                 0.08577848758081653, 0.028331933195186416, 0.002224080407310708, 0.00010386603946867244,
                 2.4336806272483765e-06],
                [7.670003601399247e-05, 0.010977975780067789, 0.17749198395637333, 0.3423587734906385,
                 0.26762063340149095, 0.1613272650199883, 0.03558053856215351, 0.004304815288057253,
                 0.00026131446521643803],
                [2.9988887071575075e-06, 0.0012100380728822476, 0.055152791268338504, 0.23782074256001023,
                 0.3042366976664677, 0.26508603808222186, 0.11004905411557955, 0.02384861332046269,
                 0.0025930260253300397],
                [2.0746396147545137e-07, 0.00018172156454292432, 0.017980428741889258, 0.14150527501625504,
                 0.2616505043598367, 0.3002937686386626, 0.20006984338208827, 0.06704560384872892,
                 0.011272646984034582],
                [1.2229783553502831e-09, 4.053791134884565e-06, 0.0015178670785093325, 0.03356114730820127,
                 0.11681011704848474, 0.21514985378380605, 0.32292803014499954, 0.22873846705624448,
                 0.08129046256564142],
                [5.572730351111348e-12, 6.296884648723474e-08, 8.037356297882278e-05, 0.004603998728773229,
                 0.02870209972984776, 0.08174969138345407, 0.25935558127072855, 0.3661526110790698,
                 0.25935558127072855],
                [1.5291249697362676e-14, 5.890006839530834e-10, 2.5628217033951704e-06, 0.0003803288616050351,
                 0.004246905247620703, 0.018704964177227536, 0.12543280829498968, 0.35294802591125274,
                 0.4982844040965849],
            ]
            soft_labels = label_vector_map[label_id]
        elif label_group == 5:
            label_vector_map = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            soft_labels = label_vector_map[label_id]
        else:
            soft_labels = self.assign_round_soft_label(len(soft_labels), label_id)

        label_vocab_id = vocab_indices[label_group][label_id]
        max_len = 12
        pad_size = max_len - len(vocab_indices[label_group])
        assert len(soft_labels) == len(vocab_indices[label_group])
        pad_vec = []
        for i in range(120, pad_size + 120):
            pad_vec.append(i)

        return vocab_indices[label_group] + pad_vec, soft_labels + [0.0] * pad_size, tokenizer.ids_to_tokens[label_vocab_id], tokenizer.convert_ids_to_tokens(vocab_indices[label_group])

    def word_piece_tokenize(self, tokens, verb_pos, tokenizer):
        if verb_pos < 0:
            return None, -1

        ret_tokens = []
        ret_verb_pos = -1
        for i, token in enumerate(tokens):
            if i == verb_pos:
                ret_verb_pos = len(ret_tokens)
                ret_tokens.append(token)
                continue
            sub_tokens = tokenizer.tokenize(token)
            ret_tokens.extend(sub_tokens)

        return ret_tokens, ret_verb_pos

    def process_new_forms(self):
        import time
        all_instances_map = {}
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, do_basic_tokenize=False)
        total_lines = float(len(self.lines))
        # lemmatizer = WordNetLemmatizer()
        # nlp = English()
        # nlp.add_pipe(nlp.create_pipe('sentencizer'))

        start_time = time.time()
        cur_i = 0
        for i, line in enumerate(self.lines):
            skip_prob = random.random()
            # if skip_prob > 0.0001:
            #     continue
            if i % 10000 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(float(i) / total_lines)
                print("Avg Time: " + str(elapsed / float(i - cur_i)))
                print("Remaining: " + str(total_lines - i))
                start_time = time.time()
                cur_i = i
            group = line.split("\t")
            sent = group[0]
            tokens_lower = sent.lower().split()
            prev_sent = group[1].lower()
            next_sent = group[2].lower()
            verb_pos = int(group[3])
            tmp_start = int(group[4])
            tmp_end = int(group[5])
            verb_no_lemma = tokens_lower[verb_pos]

            """ATTENTION"""
            # prev_sent = "[UNK]"
            # next_sent = "[UNK]"

            # verb_lemma = lemmatizer.lemmatize(tokens_lower[verb_pos], "v")
            # sysnets = wn.synsets(verb_lemma, pos=wn.VERB)
            # sysnet_definitions = []
            # for net in sysnets:
            #     doc = nlp(net.definition())
            #     doc_tokens = []
            #     for tt in doc:
            #         doc_tokens.append(str(tt))
            #     doc_tokens.append(",")
            #     sysnet_definitions.append(" ".join(doc_tokens))

            duration_check = self.check_duration_sentences(tokens_lower[tmp_start:tmp_end])
            frequency_check = self.check_frequency_sentences(tokens_lower[tmp_start:tmp_end])
            typical_check, _ = self.check_typical_sentences(tokens_lower[tmp_start:tmp_end])
            ordering_check = self.check_ordering_phrases(tokens_lower[tmp_start:tmp_end])
            boundary_check = self.check_boundary_durations(tokens_lower[tmp_start:tmp_end])

            """IF FREQ, NO DUR"""
            if (frequency_check != "FOUND_UNIT_BUT_NOT_FREQUENCY" or frequency_check == "SKIP_DURATION") and frequency_check != "NO_UNIT_FOUND":
                duration_check = "FOUND_UNIT_BUT_NOT_DURATION"

            no_tmp_token = []
            new_verb_pos = -1
            for j, t in enumerate(tokens_lower):
                if tmp_end > j >= tmp_start:
                    continue
                no_tmp_token.append(t)
                if j == verb_pos:
                    new_verb_pos = len(no_tmp_token) - 1

            no_tmp_token, new_verb_pos = self.word_piece_tokenize(no_tmp_token, new_verb_pos, tokenizer)

            if new_verb_pos < 0:
                continue

            verb_separator = "[unused500]"
            no_tmp_token.insert(new_verb_pos, verb_separator)

            prev_two_sents_tokens = ["[CLS]"] + prev_sent.split() + ["[SEP]"] + no_tmp_token
            if len(prev_two_sents_tokens) + len(next_sent.split()) + 4 > 128:
                continue

            mask_mode = False
            dimension_mode = False
            mlm_labels = [-1] * 128
            r = random.random()
            if r < 0.6:
                mask_mode = True
            elif r < 0.7:
                dimension_mode = True
            else:
                verb_no_lemma = "[MASK]"
                for it in range(2 + len(prev_sent.split()), 2 + len(prev_sent.split()) + len(no_tmp_token)):
                    if prev_two_sents_tokens[it] in ["[unused500]"]:
                        continue
                    prob = random.random()

                    if prob < 0.60:
                        prob /= 0.60

                        orig_token = prev_two_sents_tokens[it]
                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            prev_two_sents_tokens[it] = "[MASK]"

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            prev_two_sents_tokens[it] = random.choice(list(tokenizer.vocab.items()))[0]

                        # -> rest 10% randomly keep current token

                        # append current token to output (we will predict these later)
                        try:
                            mlm_labels[it] = tokenizer.vocab[orig_token]
                        except KeyError:
                            """CHANGED: NEVER PREDICT [UNK]"""
                            mlm_labels[it] = -1
                    else:
                        # no masking token (will be ignored by loss function later)
                        mlm_labels[it] = -1
                """NOTE: NEW"""
                for it in range(0, 2 + len(prev_sent.split())):
                    if prev_two_sents_tokens[it] not in ["[SEP]", "[CLS]"]:
                        prev_two_sents_tokens[it] = "[MASK]"
                new_next_sent = []
                for iit in range(0, len(next_sent.split())):
                    new_next_sent.append("[MASK]")
                next_sent = " ".join(new_next_sent)

            if duration_check != "NO_UNIT_FOUND" and duration_check != "FOUND_UNIT_BUT_NOT_DURATION":
                unit_string = self.normalize_timex(duration_check) + "_dur"
                soft_label_indices, soft_label_values, rep_label, other_possible_labels = self.get_soft_labels(unit_string, tokenizer)
                target_index = -1
                dimension_marker = "[unused501]"
                if mask_mode or dimension_mode:
                    rr = random.random()
                    if rr < 0.8:
                        rep_label = "[MASK]"
                    elif rr < 0.9:
                        rep_label = random.choice(other_possible_labels)
                if dimension_mode:
                    mlm_labels[len(prev_two_sents_tokens) + 3 + len(next_sent.split())] = tokenizer.vocab[dimension_marker]
                    dimension_marker = "[MASK]"
                sent_string = " ".join(prev_two_sents_tokens) + " [SEP] " + next_sent + " [SEP] [unused500] " + dimension_marker + " "
                if mask_mode:
                    target_index = len(sent_string.split())
                sent_string += rep_label + " [SEP]"

                # left_space = 128 - len(sent_string.split()) - 3
                # if left_space > 0:
                #     sent_string += " " + verb_no_lemma + " : "
                # added_count = 0
                # for sys_iter in range(0, min(3, len(sysnet_definitions))):
                #     tokens_def = sysnet_definitions[sys_iter].split()
                #     added_count += len(tokens_def)
                #     if added_count < left_space:
                #         sent_string += " " + " ".join(tokens_def)
                #     else:
                #         break
                # if not sent_string.endswith("[SEP]"):
                #     sent_string += " [SEP]"

                key = " ".join(tokens_lower) + " " + str(verb_pos)
                if key not in all_instances_map:
                    all_instances_map[key] = []
                all_instances_map[key].append(["DUR", sent_string, target_index, soft_label_indices, soft_label_values, mlm_labels])

            if frequency_check != "NO_UNIT_FOUND" and frequency_check != "FOUND_UNIT_BUT_NOT_FREQUENCY" and frequency_check != "SKIP_DURATION":
                unit_string = self.normalize_timex(frequency_check) + "_freq"
                soft_label_indices, soft_label_values, rep_label, other_possible_labels = self.get_soft_labels(unit_string, tokenizer)
                target_index = -1
                dimension_marker = "[unused502]"
                if mask_mode or dimension_mode:
                    rr = random.random()
                    if rr < 0.8:
                        rep_label = "[MASK]"
                    elif rr < 0.9:
                        rep_label = random.choice(other_possible_labels)
                if dimension_mode:
                    mlm_labels[len(prev_two_sents_tokens) + 3 + len(next_sent.split())] = tokenizer.vocab[dimension_marker]
                    dimension_marker = "[MASK]"
                sent_string = " ".join(prev_two_sents_tokens) + " [SEP] " + next_sent + " [SEP] [unused500] " + dimension_marker + " "
                if mask_mode:
                    target_index = len(sent_string.split())
                sent_string += rep_label + " [SEP]"

                # left_space = 128 - len(sent_string.split()) - 3
                # if left_space > 0:
                #     sent_string += " " + verb_no_lemma + " : "
                # added_count = 0
                # for sys_iter in range(0, min(3, len(sysnet_definitions))):
                #     tokens_def = sysnet_definitions[sys_iter].split()
                #     added_count += len(tokens_def)
                #     if added_count < left_space:
                #         sent_string += " " + " ".join(tokens_def)
                #     else:
                #         break
                # if not sent_string.endswith("[SEP]"):
                #     sent_string += " [SEP]"

                key = " ".join(tokens_lower) + " " + str(verb_pos)
                if key not in all_instances_map:
                    all_instances_map[key] = []
                all_instances_map[key].append(["FREQ", sent_string, target_index, soft_label_indices, soft_label_values, mlm_labels])

            if typical_check != "NO_TYPICAL_FOUND":
                unit_string = typical_check
                soft_label_indices, soft_label_values, rep_label, other_possible_labels = self.get_soft_labels(unit_string, tokenizer)
                target_index = -1
                dimension_marker = "[unused503]"
                if mask_mode or dimension_mode:
                    rr = random.random()
                    if rr < 0.8:
                        rep_label = "[MASK]"
                    elif rr < 0.9:
                        rep_label = random.choice(other_possible_labels)
                if dimension_mode:
                    mlm_labels[len(prev_two_sents_tokens) + 3 + len(next_sent.split())] = tokenizer.vocab[dimension_marker]
                    dimension_marker = "[MASK]"
                sent_string = " ".join(prev_two_sents_tokens) + " [SEP] " + next_sent + " [SEP] [unused500] " + dimension_marker + " "
                if mask_mode:
                    target_index = len(sent_string.split())
                sent_string += rep_label + " [SEP]"

                # left_space = 128 - len(sent_string.split()) - 3
                # if left_space > 0:
                #     sent_string += " " + verb_no_lemma + " : "
                # added_count = 0
                # for sys_iter in range(0, min(3, len(sysnet_definitions))):
                #     tokens_def = sysnet_definitions[sys_iter].split()
                #     added_count += len(tokens_def)
                #     if added_count < left_space:
                #         sent_string += " " + " ".join(tokens_def)
                #     else:
                #         break
                # if not sent_string.endswith("[SEP]"):
                #     sent_string += " [SEP]"

                key = " ".join(tokens_lower) + " " + str(verb_pos)
                if key not in all_instances_map:
                    all_instances_map[key] = []
                all_instances_map[key].append(["TYP", sent_string, target_index, soft_label_indices, soft_label_values, mlm_labels])

            if ordering_check != "NO_ORDERING_FOUND":
                unit_string = ordering_check.split()[0]
                content_string = " ".join(ordering_check.split()[1:])
                soft_label_indices, soft_label_values, rep_label, other_possible_labels = self.get_soft_labels(unit_string, tokenizer)
                target_index = -1
                dimension_marker = "[unused504]"
                if mask_mode or dimension_mode:
                    rr = random.random()
                    if rr < 0.8:
                        rep_label = "[MASK]"
                    elif rr < 0.9:
                        rep_label = random.choice(other_possible_labels)
                if dimension_mode:
                    mlm_labels[len(prev_two_sents_tokens) + 3 + len(next_sent.split())] = tokenizer.vocab[dimension_marker]
                    dimension_marker = "[MASK]"
                sent_string = " ".join(prev_two_sents_tokens) + " [SEP] " + next_sent + " [SEP] [unused500] " + dimension_marker + " "
                if mask_mode:
                    target_index = len(sent_string.split())
                sent_string += rep_label + " " + content_string + " [SEP]"

                # left_space = 128 - len(sent_string.split()) - 3
                # if left_space > 0:
                #     sent_string += " " + verb_no_lemma + " : "
                # added_count = 0
                # for sys_iter in range(0, min(3, len(sysnet_definitions))):
                #     tokens_def = sysnet_definitions[sys_iter].split()
                #     added_count += len(tokens_def)
                #     if added_count < left_space:
                #         sent_string += " " + " ".join(tokens_def)
                #     else:
                #         break
                # if not sent_string.endswith("[SEP]"):
                #     sent_string += " [SEP]"

                key = " ".join(tokens_lower) + " " + str(verb_pos)
                if key not in all_instances_map:
                    all_instances_map[key] = []
                all_instances_map[key].append(["ORD", sent_string, target_index, soft_label_indices, soft_label_values, mlm_labels])

            if boundary_check != "NO_BOUNDARY_FOUND":
                unit_string = self.normalize_timex(boundary_check) + "_bnd"
                soft_label_indices, soft_label_values, rep_label, other_possible_labels = self.get_soft_labels(unit_string, tokenizer)
                target_index = -1
                dimension_marker = "[unused505]"
                if mask_mode or dimension_mode:
                    rr = random.random()
                    if rr < 0.8:
                        rep_label = "[MASK]"
                    elif rr < 0.9:
                        rep_label = random.choice(other_possible_labels)
                if dimension_mode:
                    mlm_labels[len(prev_two_sents_tokens) + 3 + len(next_sent.split())] = tokenizer.vocab[dimension_marker]
                    dimension_marker = "[MASK]"
                sent_string = " ".join(prev_two_sents_tokens) + " [SEP] " + next_sent + " [SEP] [unused500] " + dimension_marker + " "
                if mask_mode:
                    target_index = len(sent_string.split())
                sent_string += rep_label + " [SEP]"

                # left_space = 128 - len(sent_string.split()) - 3
                # if left_space > 0:
                #     sent_string += " " + verb_no_lemma + " : "
                # added_count = 0
                # for sys_iter in range(0, min(3, len(sysnet_definitions))):
                #     tokens_def = sysnet_definitions[sys_iter].split()
                #     added_count += len(tokens_def)
                #     if added_count < left_space:
                #         sent_string += " " + " ".join(tokens_def)
                #     else:
                #         break
                # if not sent_string.endswith("[SEP]"):
                #     sent_string += " [SEP]"

                key = " ".join(tokens_lower) + " " + str(verb_pos)
                if key not in all_instances_map:
                    all_instances_map[key] = []
                all_instances_map[key].append(["BND", sent_string, target_index, soft_label_indices, soft_label_values, mlm_labels])

        care_order = ["DUR", "ORD", "FREQ", "BND", "TYP"]
        f_out = open("samples/gigaword/lm_format_realnews.txt", "w")
        count_map = {}
        for key in all_instances_map:
            select = None
            for care_target in care_order:
                for inst in all_instances_map[key]:
                    if inst[0] == care_target:
                        select = inst
                        break
                    if select is not None:
                        break
            if select is None:
                print("ERROR: SELECT CANNOT BE NONE")
            if select[0] not in count_map:
                count_map[select[0]] = 0
            count_map[select[0]] += 1
            f_out.write(select[1] + "\t" + str(select[2]) + "\t" + " ".join([str(x) for x in select[3]]) + "\t" + " ".join([str(x) for x in select[4]]) + "\t" + " ".join([str(x) for x in select[5]]) + "\n")
        print(count_map)

    def process(self):
        duration_strings = []
        frequency_strings = []
        typical_strings = []
        ordering_strings = []
        for i, line in enumerate(self.lines):
            group = line.split("\t")
            sent = group[0]
            tokens_lower = sent.lower().split()
            # tokens_srl_lower = self.srl_lines[i].split("\t")[0].lower().split()
            prev_sent = group[1].lower()
            next_sent = group[2].lower()
            verb_pos = int(group[3])
            tmp_start = int(group[4])
            tmp_end = int(group[5])

            # verb_pos_srl = int(self.srl_lines[i].split("\t")[3])
            # tmp_start_srl = int(self.srl_lines[i].split("\t")[4])
            # tmp_end_srl = int(self.srl_lines[i].split("\t")[5])

            duration_check = self.check_duration_sentences(tokens_lower[tmp_start:tmp_end])
            frequency_check = self.check_frequency_sentences(tokens_lower[tmp_start:tmp_end])
            typical_check, typ_group = self.check_typical_sentences(tokens_lower[tmp_start:tmp_end])
            ordering_check = self.check_ordering_phrases(tokens_lower[tmp_start:tmp_end])
            boundary_check = self.check_boundary_durations(tokens_lower[tmp_start:tmp_end])

            """IF FREQ, NO DUR"""
            if (frequency_check != "FOUND_UNIT_BUT_NOT_FREQUENCY" or frequency_check == "SKIP_DURATION") and frequency_check != "NO_UNIT_FOUND":
                duration_check = "FOUND_UNIT_BUT_NOT_DURATION"

            no_tmp_token = []
            new_verb_pos = -1
            for j, t in enumerate(tokens_lower):
                if tmp_end > j >= tmp_start:
                    continue
                no_tmp_token.append(t)
                if j == verb_pos:
                    new_verb_pos = len(no_tmp_token) - 1
            if new_verb_pos < 0:
                continue

            # no_tmp_token_srl = []
            # new_verb_pos_srl = -1
            # for j, t in enumerate(tokens_srl_lower):
            #     if tmp_end_srl > j >= tmp_start_srl:
            #         continue
            #     no_tmp_token_srl.append(t)
            #     if j == verb_pos_srl:
            #         new_verb_pos_srl = len(no_tmp_token_srl) - 1


            twosent_verb_pos = 2 + len(prev_sent.split()) + new_verb_pos
            onesent_verb_pos = 1 + new_verb_pos
            # onesent_verb_pos_srl = 1 + new_verb_pos_srl

            # print(prev_sent)
            # tokens_lower[verb_pos] = "[[" + tokens_lower[verb_pos] + "]]"
            # print(" ".join(tokens_lower))
            # print(next_sent)
            # print(" ".join(tokens_lower[tmp_start:tmp_end]))
            # print("Duration: " + duration_check)
            # print("Frequency: " + frequency_check)
            # print("Typical: " + typical_check)
            # print("Ordering: " + ordering_check)
            # print("Boundary: " + boundary_check)
            # print()

            if duration_check != "NO_UNIT_FOUND" and duration_check != "FOUND_UNIT_BUT_NOT_DURATION":
                duration_strings.append([
                    "[CLS] " + prev_sent + " [SEP] " + " ".join(no_tmp_token) + " [SEP]\t" + str(twosent_verb_pos) + "\t" + duration_check + "\t" + "DUR" + "\n",
                    "[CLS] " + " ".join(no_tmp_token) + " [SEP]\t" + str(onesent_verb_pos) + "\t" + duration_check + "\t" + "DUR" + "\n",
                    # "[CLS] " + " ".join(no_tmp_token_srl) + " [SEP]\t" + str(onesent_verb_pos_srl) + "\t" + duration_check + "\t" + "DUR" + "\n",
                ])
            if frequency_check != "NO_UNIT_FOUND" and frequency_check != "FOUND_UNIT_BUT_NOT_FREQUENCY":
                frequency_strings.append([
                    "[CLS] " + prev_sent + " [SEP] " + " ".join(no_tmp_token) + " [SEP]\t" + str(twosent_verb_pos) + "\t" + frequency_check + "\t" + "FREQ" + "\n",
                    "[CLS] " + " ".join(no_tmp_token) + " [SEP]\t" + str(onesent_verb_pos) + "\t" + frequency_check + "\t" + "FREQ" + "\n",
                    # "[CLS] " + " ".join(no_tmp_token_srl) + " [SEP]\t" + str(onesent_verb_pos_srl) + "\t" + frequency_check + "\t" + "FREQ" + "\n",
                ])
            if typical_check != "NO_TYPICAL_FOUND":
                """SAMPLING!"""
                r = random.random()
                sample_map = {
                    1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1
                }
                if r < sample_map[typ_group]:
                    typical_strings.append([
                        "[CLS] " + prev_sent + " [SEP] " + " ".join(no_tmp_token) + " [SEP]\t" + str(twosent_verb_pos) + "\t" + typical_check + "\t" + "TYP" + "\n",
                        "[CLS] " + " ".join(no_tmp_token) + " [SEP]\t" + str(onesent_verb_pos) + "\t" + typical_check + "\t" + "TYP" + "\n",
                        # "[CLS] " + " ".join(no_tmp_token_srl) + " [SEP]\t" + str(onesent_verb_pos_srl) + "\t" + typical_check + "\t" + "TYP" + "\n",
                    ])
            if ordering_check != "NO_ORDERING_FOUND":
                ordering_strings.append([
                    "[CLS] " + prev_sent + " [SEP] " + " ".join(no_tmp_token) + " [SEP]\t" + str(0) + "\t" + "yes" + "\t" + "ORD" + "\n",
                    "[CLS] " + prev_sent + " [SEP] " + " ".join(no_tmp_token) + " [SEP]\t" + str(0) + "\t" + "yes" + "\t" + "ORD" + "\n",
                    "[CLS] " + prev_sent + " [SEP] " + " ".join(no_tmp_token) + " [SEP]\t" + str(0) + "\t" + "yes" + "\t" + "ORD" + "\n",
                    ])

        ordering_strings_neg = []
        while len(ordering_strings_neg) < len(ordering_strings):
            rand_line = random.choice(self.rand_lines)
            if len(rand_line.split("\t")) < 3:
                continue
            start_idx = random.choice([0, 1])
            prev = rand_line.split("\t")[start_idx].lower()
            next = rand_line.split("\t")[start_idx+1].lower()
            if prev != "NONE" and next != "NONE":
                ordering_strings_neg.append([
                    "[CLS] " + prev + " [SEP] " + next + " [SEP] \t" + str(0) + "\t" + "no" + "\t" + "ORD" + "\n",
                    "[CLS] " + prev + " [SEP] " + next + " [SEP] \t" + str(0) + "\t" + "no" + "\t" + "ORD" + "\n",
                    "[CLS] " + prev + " [SEP] " + next + " [SEP] \t" + str(0) + "\t" + "no" + "\t" + "ORD" + "\n",
                    ])

        random.shuffle(duration_strings)
        random.shuffle(frequency_strings)
        random.shuffle(typical_strings)
        random.shuffle(ordering_strings)
        random.shuffle(ordering_strings_neg)

        # splits = [0.89, 0.9, 1.0]
        """SMALL SPLIT"""
        splits = [0.39, 0.4, 1.0]
        splits_name = ["train", "dev", "test"]
        data_lists = [duration_strings, frequency_strings, typical_strings, ordering_strings, ordering_strings_neg]
        # data_lists = [duration_strings, frequency_strings, typical_strings]
        # prev_split = 0.0
        # for i, s in enumerate(splits):
        #     f_out = open(self.output_path + "/" + splits_name[i] + ".formatted.txt", "w")
        #     f_out_onesent = open(self.output_path_singlesent + "/" + splits_name[i] + ".formatted.txt", "w")
        #     # f_out_onesent_srl = open(self.output_path_srl + "/" + splits_name[i] + ".formatted.txt", "w")
        #     for data_list in data_lists:
        #         start_idx = int(len(data_list) * prev_split)
        #         end_idx = int(len(data_list) * s)
        #         for item in data_list[start_idx:end_idx]:
        #             f_out.write(item[0])
        #             f_out_onesent.write(item[1])
        #             # f_out_onesent_srl.write(item[2])
        #     prev_split = s


class TmpArgFilter:

    def __init__(self):
        self.file_path = "samples/wikipedia/tmparg_collection_with_prev_and_next.txt"
        self.lines = [x.strip() for x in open(self.file_path).readlines()]
        self.output_path = "samples/wikipedia/tmparg_index.txt"
        self.f_out = open(self.output_path, "w")
        self.process()

    def check_keywords(self, tmp_args):
        # keywords = [
        #     # Durational
        #     "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years", "decade", "decades", "century", "centuries",
        #     # Time of the day
        #     "dawn", "morning", "noon", "afternoon", "evening", "dusk", "night", "midnight",
        #     "dawns", "mornings", "noons", "afternoons", "evenings", "dusks", "nights", "midnights",
        #     # Time of the week
        #     "monday", "tuesday", "wednesday", 'thursday', "friday", "saturday", "sunday",
        #     "mondays", "tuesdays", "wednesdays", 'thursdays', "fridays", "saturdays", "sundays",
        #     # Months
        #     "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        #     "januarys", "januaries", "februarys", "februaries", "marches", "marchs", "aprils", "mays", "junes", "julys", "julies", "augusts", "septembers", "octobers", "novembers", "decembers",
        #     # Seasons
        #     "spring", "summer", "autumn", "fall", "winter",
        #     "springs", "summers", "autumns", "falls", "winters",
        #     ]
        keywords = {
            "second": [0, 0],
            "seconds": [0, 0],
            "minute": [0, 1],
            "minutes": [0, 1],
            "hour": [0, 2],
            "hours": [0, 2],
            "day": [0, 3],
            "days": [0, 3],
            "week": [0, 4],
            "weeks": [0, 4],
            "month": [0, 5],
            "months": [0, 5],
            "year": [0, 6],
            "years": [0, 6],
            "decade": [0, 7],
            "decades": [0, 7],
            "century": [0, 8],
            "centuries": [0, 8],
            "dawns": [1, 0],
            "mornings": [1, 1],
            "noons": [1, 2],
            "afternoons": [1, 3],
            "evenings": [1, 4],
            "dusks": [1, 5],
            "nights": [1, 6],
            "midnights": [1, 7],
            "dawn": [1, 0],
            "morning": [1, 1],
            "noon": [1, 2],
            "afternoon": [1, 3],
            "evening": [1, 4],
            "dusk": [1, 5],
            "night": [1, 6],
            "midnight": [1, 7],
            "monday": [2, 0],
            "tuesday": [2, 1],
            "wednesday": [2, 2],
            "thursday": [2, 3],
            "friday": [2, 4],
            "saturday": [2, 5],
            "sunday": [2, 6],
            "mondays": [2, 0],
            "tuesdays": [2, 1],
            "wednesdays": [2, 2],
            "thursdays": [2, 3],
            "fridays": [2, 4],
            "saturdays": [2, 5],
            "sundays": [2, 6],
            "january": [3, 0],
            "february": [3, 1],
            "march": [3, 2],
            "april": [3, 3],
            "may": [3, 4],
            "june": [3, 5],
            "july": [3, 6],
            "august": [3, 7],
            "september": [3, 8],
            "october": [3, 9],
            "november": [3, 10],
            "december": [3, 11],
            "januarys": [3, 0],
            "januaries": [3, 0],
            "februarys": [3, 1],
            "februaries": [3, 1],
            "marches": [3, 2],
            "marchs": [3, 2],
            "aprils": [3, 3],
            "mays": [3, 4],
            "junes": [3, 5],
            "julys": [3, 6],
            "julies": [3, 6],
            "augusts": [3, 7],
            "septembers": [3, 8],
            "octobers": [3, 9],
            "novembers": [3, 10],
            "decembers": [3, 11],
            "springs": [4, 0],
            "summers": [4, 1],
            "autumns": [4, 2],
            "winters": [4, 3],
            "spring": [4, 0],
            "summer": [4, 1],
            "autumn": [4, 2],
            "winter": [4, 3],
        }
        keywords_num = {
            "0": [5, 0], "1": [5, 1], "2": [5, 2], "3": [5, 3], "4": [5, 4], "5": [5, 5], "6": [5, 6], "7": [5, 7],
            "8": [5, 8], "9": [5, 9], "10": [5, 10],
            "zero": [5, 0], "one": [5, 1], "two": [5, 2], "three": [5, 3], "four": [5, 4], "five": [5, 5],
            "six": [5, 6], "seven": [5, 7], "eight": [5, 8], "nine": [5, 9], "ten": [5, 10],
            "no": [5, 0], "none": [5, 0], "a": [5, 1], "an": [5, 1], "each": [5, 1], "every": [5, 1],
            "several": [5, 3], "many": [5, 5]
        }
        possible = False
        for i, token in enumerate(tmp_args):
            if token.lower() in keywords:
                possible = True
        ret = []
        if possible:
            for i, token in enumerate(tmp_args):
                # if token.lower() in keywords or token.lower() in keywords_num:
                if token.lower() in keywords:
                    ret.append((token, i))
        return ret

    def process(self):
        sentence_map = {}
        next_prev_sentence_map = {}
        for line in self.lines:
            groups = line.split("\t")
            if groups[0] not in sentence_map:
                sentence_map[groups[0]] = []
                next_prev_sentence_map[groups[0]] = []
            sentence_map[groups[0]].append((int(groups[4]), int(groups[5])))
            prev_sent = groups[1]
            if prev_sent == "NONE":
                prev_sent = "[UNK]"
            next_sent = groups[2]
            if next_sent == "NONE":
                next_sent = "[UNK]"
            next_prev_sentence_map[groups[0]] = (prev_sent, next_sent)
        for sentence in sentence_map:
            tmp_args = sentence_map[sentence]
            tokens = sentence.split()
            tmp_keys = []
            for tmp_start, tmp_end in tmp_args:
                all_tmp_keys = self.check_keywords(tokens[tmp_start:tmp_end])
                all_tmp_keys = [(x[0], x[1] + tmp_start) for x in all_tmp_keys]
                tmp_keys += all_tmp_keys
            if len(tmp_keys) == 0:
                continue
            _, select_idx = random.choice(tmp_keys)
            all_tmp_pos = [str(x[1]) for x in tmp_keys]
            all_tmp_pos = list(set(all_tmp_pos))
            select_other_sent = random.choice([0, 1])
            self.f_out.write(sentence + "\t" + next_prev_sentence_map[sentence][select_other_sent] + "\t" + str(select_other_sent) + "\t" + str(select_idx) + "\t" + " ".join(all_tmp_pos) + "\n")


class Randomizer:
    def __init__(self, fname, ratio=1.0):
        lines = [x.strip() for x in open(fname).readlines()]
        print(len(lines))
        random.shuffle(lines)
        f_out = open(fname, "w")
        write_len = (int(len(lines) * ratio))
        print(write_len)
        for l in lines[:write_len]:
            f_out.write(l + "\n")
        # f_out_train = open("samples/wikipedia_tmparg_full/train.formatted.txt", "w")
        # f_out_test = open("samples/wikipedia_tmparg_full/test.formatted.txt", "w")
        # for line in lines[:10000000]:
        #     f_out_train.write(line + "\n")
        # for line in lines[10000000:]:
        #     f_out_test.write(line + "\n")


def visualize():
    lines = [x.strip() for x in open("samples/gigaword/twosent/train.formatted.txt").readlines()]
    cate_instances = {}
    for line in lines:
        t = line.split("\t")[-1]
        if t not in cate_instances:
            cate_instances[t] = []
        cate_instances[t].append(line)

    for t in cate_instances:
        random.shuffle(cate_instances[t])
        for line in cate_instances[t][:100000000000]:
            tokens = line.split("\t")[0].split()
            tokens[int(line.split("\t")[1])] = "[[" + tokens[int(line.split("\t")[1])] + "]]"
            print(" ".join(tokens) + '\t' + line.split("\t")[1] + "\t" + line.split("\t")[-2] + "\t" + t)


# p = AdditionalPatternProcessor()
# p = TmpArgProcessor()
p = TmpArgDimensionFilter()
# f = TmpArgFilter()
# visualize()
# r = Randomizer("samples/wikipedia_joint/test.formatted.txt", 0.3)
