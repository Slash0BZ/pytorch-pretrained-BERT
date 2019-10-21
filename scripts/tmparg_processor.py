import jsonlines
import os
import random
from word2number import w2n
import copy


class TmpArgProcessor:

    def __init__(self):
        self.output_path = "samples/wikipedia/tmparg_collection_srl_skeleton_with_prev_and_next.txt"
        self.f_out = open(self.output_path, "w")
        self.root_path = "samples/wikipedia"
        self.context_table = {}
        self.build_context_table()
        self.process()

    def build_context_table(self):
        lines = [x.strip() for x in open("samples/wikipedia/raw_collection_contextsent_tokenized.txt").readlines()]
        print("Loaded all lines")
        for i, line in enumerate(lines):
            if i % 1000000 == 0:
                print("Added " + str(i) + " sentences to context table.")
            key = line.split("\t")[1].replace(" ", "")
            if len(line.split("\t")) < 3:
                continue
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
                        """CHANGING TO SRL ONLY!"""
                        srl_tokens, srl_verb_pos, srl_tmp_start, srl_tmp_end = self.get_stripped(obj['words'], verb['tags'], verb_pos, tmp_start, tmp_end)
                        # if srl_tmp_end - srl_tmp_start != tmp_end - tmp_start:
                        #     print(obj['words'])
                        #     print(verb['tags'])
                        #     print(str(verb_pos) + "\t" + str(tmp_start) + "\t" + str(tmp_end))
                        #     print()
                        #     print(srl_tokens)
                        #     print(str(srl_verb_pos) + "\t" + str(srl_tmp_start) + "\t" + str(srl_tmp_end))
                        #     print("--------------------")
                        self.f_out.write(" ".join(srl_tokens) + "\t" + left_sent + "\t" + right_sent + "\t" + str(srl_verb_pos) + "\t" + str(srl_tmp_start) + "\t" + str(srl_tmp_end) + "\n")
                        # self.f_out.write(" ".join(obj['words']) + "\t" + left_sent + "\t" + right_sent + "\t" + str(verb_pos) + "\t" + str(tmp_start) + "\t" + str(tmp_end) + "\n")

    def process(self):
        for dirName, subdirList, fileList in os.walk(self.root_path):
            for fname in fileList:
                if fname.startswith("srl"):
                    self.process_single_file(self.root_path + "/" + fname)
                    print("Finished " + fname)


class TmpArgDimensionFilter:
    def __init__(self):
        self.file_path = "samples/wikipedia/tmparg_collection_srl_skeleton_with_prev_and_next.txt"
        self.lines = [x.strip() for x in open(self.file_path).readlines()]
        self.rand_file_path = "samples/wikipedia/raw_collection_randomsent_contextsent_tokenized.txt"
        self.rand_lines = [x.strip() for x in open(self.rand_file_path).readlines()]
        self.output_path = "samples/wikipedia_joint_singlesent_srl"
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
        self.process()

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
        if tmparg_tokens[0] in ["for", "over"] and "second time" not in " ".join(tmparg_tokens):
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
        for i, token in enumerate(tmparg_tokens[quantity_stop-5:quantity_stop]):
            if token == "every" or token == "once" or token == "per" or token == "each":
                num /= 1.0
                valid = True
            if token == "twice":
                num /= 2.0
                valid = True
            if token == "times":
                div = self.quantity([tmparg_tokens[quantity_stop-5:quantity_stop][i-1]])
                if div is not None and div > 0.0:
                    num /= div
                    valid = True
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
        for t in tmparg_tokens:
            if t in keywords:
                return t, keywords[t][0]
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
        if tmparg_tokens[0] in ["after", "later"] and tmparg_tokens[-1] in self.value_map:
            return str(num) + " " + unit
        if tmparg_tokens[-1] == "later":
            return str(num) + " " + unit

        return "NO_ORDERING_FOUND"

    def process(self):
        duration_strings = set()
        frequency_strings = set()
        typical_strings = set()
        ordering_strings = set()
        for line in self.lines:
            group = line.split("\t")
            sent = group[0]
            tokens_lower = sent.lower().split()
            prev_sent = group[1].lower()
            next_sent = group[2].lower()
            verb_pos = int(group[3])
            # if verb_pos == -1:
            #     print(tokens_lower)
            #     print(verb_pos)
            #     print(tmp_start)
            #     print(tmp_end)
            tmp_start = int(group[4])
            tmp_end = int(group[5])

            duration_check = self.check_duration_sentences(tokens_lower[tmp_start:tmp_end])
            frequency_check = self.check_frequency_sentences(tokens_lower[tmp_start:tmp_end])
            typical_check, typ_group = self.check_typical_sentences(tokens_lower[tmp_start:tmp_end])
            ordering_check = self.check_ordering_sentences(tokens_lower, tmp_start, tmp_end)
            no_tmp_token = []
            new_verb_pos = -1
            for i, t in enumerate(tokens_lower):
                if tmp_end > i >= tmp_start:
                    continue
                no_tmp_token.append(t)
                if i == verb_pos:
                    new_verb_pos = len(no_tmp_token) - 1
            if new_verb_pos < 0:
                continue
            """CHANGED THIS!!!"""
            # new_verb_pos += 2 + len(prev_sent.split())
            new_verb_pos += 1

            if duration_check != "NO_UNIT_FOUND" and duration_check != "FOUND_UNIT_BUT_NOT_DURATION":
                # duration_strings.add("[CLS] " + prev_sent + " [SEP] " + " ".join(no_tmp_token) + " [SEP]\t" + str(new_verb_pos) + "\t" + duration_check + "\t" + "DUR" + "\n")
                duration_strings.add("[CLS] " + " ".join(no_tmp_token) + " [SEP]\t" + str(new_verb_pos) + "\t" + duration_check + "\t" + "DUR" + "\n")
            if frequency_check != "NO_UNIT_FOUND" and frequency_check != "FOUND_UNIT_BUT_NOT_FREQUENCY":
                # frequency_strings.add("[CLS] " + prev_sent + " [SEP] " + " ".join(no_tmp_token) + " [SEP]\t" + str(new_verb_pos) + "\t" + frequency_check + "\t" + "FREQ" + "\n")
                frequency_strings.add("[CLS] " +  " ".join(no_tmp_token) + " [SEP]\t" + str(new_verb_pos) + "\t" + frequency_check + "\t" + "FREQ" + "\n")
            if typical_check != "NO_TYPICAL_FOUND":
                """SAMPLING!"""
                r = random.random()
                if r < 0.1:
                    # typical_strings.add("[CLS] " + prev_sent + " [SEP] " + " ".join(no_tmp_token) + " [SEP]\t" + str(new_verb_pos) + "\t" + typical_check + "\t" + "TYP" + "\n")
                    typical_strings.add("[CLS] " + " ".join(no_tmp_token) + " [SEP]\t" + str(new_verb_pos) + "\t" + typical_check + "\t" + "TYP" + "\n")
            if ordering_check != "NO_ORDERING_FOUND":
                ordering_strings.add("[CLS] " + prev_sent + " [SEP] " + " ".join(no_tmp_token) + " [SEP]\t" + str(0) + "\t" + "yes" + "\t" + "ORD" + "\n")

        ordering_strings_neg = set()
        while len(ordering_strings_neg) < len(ordering_strings):
            rand_line = random.choice(self.rand_lines)
            if len(rand_line.split("\t")) < 3:
                continue
            start_idx = random.choice([0, 1])
            prev = rand_line.split("\t")[start_idx].lower()
            next = rand_line.split("\t")[start_idx+1].lower()
            if prev != "NONE" and next != "NONE":
                ordering_strings_neg.add("[CLS] " + prev + " [SEP] " + next + "[SEP] \t" + str(0) + "\t" + "no" + "\t" + "ORD" + "\n")

        duration_strings = list(duration_strings)
        frequency_strings = list(frequency_strings)
        typical_strings = list(typical_strings)
        ordering_strings = list(ordering_strings)
        ordering_strings_neg = list(ordering_strings_neg)
        random.shuffle(duration_strings)
        random.shuffle(frequency_strings)
        random.shuffle(typical_strings)
        random.shuffle(ordering_strings)
        random.shuffle(ordering_strings_neg)

        splits = [0.89, 0.9, 1.0]
        splits_name = ["train", "dev", "test"]
        data_lists = [duration_strings, frequency_strings, typical_strings, ordering_strings, ordering_strings_neg]
        prev_split = 0.0
        for i, s in enumerate(splits):
            f_out = open(self.output_path + "/" + splits_name[i] + ".formatted.txt", "w")
            for data_list in data_lists:
                start_idx = int(len(data_list) * prev_split)
                end_idx = int(len(data_list) * s)
                for item in data_list[start_idx:end_idx]:
                    f_out.write(item)
            prev_split = s


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
    def __init__(self):
        lines = [x.strip() for x in open("samples/wikipedia_joint/dev.formatted.txt").readlines()]
        random.shuffle(lines)
        f_out = open("samples/wikipedia_joint/dev.formatted.txt", "w")
        for l in lines:
            f_out.write(l + "\n")
        # f_out_train = open("samples/wikipedia_tmparg_full/train.formatted.txt", "w")
        # f_out_test = open("samples/wikipedia_tmparg_full/test.formatted.txt", "w")
        # for line in lines[:10000000]:
        #     f_out_train.write(line + "\n")
        # for line in lines[10000000:]:
        #     f_out_test.write(line + "\n")


def visualize():
    lines = [x.strip() for x in open("samples/wikipedia/tmparg_index.txt").readlines()]
    for line in lines[:100]:
        tokens = line.split("\t")[0].split()
        tokens[int(line.split("\t")[1])] = "[[" + tokens[int(line.split("\t")[1])] + "]]"
        print(" ".join(tokens))


p = TmpArgProcessor()
p = TmpArgDimensionFilter()
# f = TmpArgFilter()
# visualize()
# r = Randomizer()
