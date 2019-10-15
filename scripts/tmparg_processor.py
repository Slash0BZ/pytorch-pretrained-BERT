import jsonlines
import os
import random


class TmpArgProcessor:

    def __init__(self):
        self.output_path = "samples/wikipedia/tmparg_collection.txt"
        self.f_out = open(self.output_path, "w")
        self.root_path = "samples/wikipedia"
        self.process()

    def get_tmp_arg_range(self, tags):
        start = -1
        end = -1
        for i, t in enumerate(tags):
            if t == "B-ARGM-TMP":
                start = i
                end = i + 1
            if t == "I-ARGM-TMP":
                end += 1
        return start, end

    def process_single_file(self, file_name):
        lines = [x.strip() for x in open(file_name).readlines()]
        reader = jsonlines.Reader(lines)
        for obj_list in reader:
            for obj in obj_list:
                for verb in obj['verbs']:
                    tmp_start, tmp_end = self.get_tmp_arg_range(verb['tags'])
                    if tmp_start > -1:
                        self.f_out.write(" ".join(obj['words']) + "\t" + str(tmp_start) + "\t" + str(tmp_end) + "\n")

    def process(self):
        for dirName, subdirList, fileList in os.walk(self.root_path):
            for fname in fileList:
                self.process_single_file(self.root_path + "/" + fname)


class TmpArgFilter:

    def __init__(self):
        self.file_path = "samples/wikipedia/tmparg_collection.txt"
        self.lines = [x.strip() for x in open(self.file_path).readlines()]
        self.output_path = "samples/wikipedia/tmparg_index.txt"
        self.f_out = open(self.output_path, "w")
        self.process()

    def check_keywords(self, tmp_args):
        keywords = [
            # Durational
            "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years", "decade", "decades", "century", "centuries",
            # Time of the day
            "dawn", "morning", "noon", "afternoon", "evening", "dusk", "night", "midnight",
            "dawns", "mornings", "noons", "afternoons", "evenings", "dusks", "nights", "midnights",
            # Time of the week
            "monday", "tuesday", "wednesday", 'thursday', "friday", "saturday", "sunday",
            "mondays", "tuesdays", "wednesdays", 'thursdays', "fridays", "saturdays", "sundays",
            # Months
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
            "januarys", "januaries", "februarys", "februaries", "marches", "marchs", "aprils", "mays", "junes", "julys", "julies", "augusts", "septembers", "octobers", "novembers", "decembers",
            # Seasons
            "spring", "summer", "autumn", "fall", "winter",
            "springs", "summers", "autumns", "falls", "winters",
        ]
        for i, token in enumerate(tmp_args):
            if token.lower() in keywords:
                return token, i
        return None, -1

    def process(self):
        sentence_map = {}
        for line in self.lines:
            groups = line.split("\t")
            if groups[0] not in sentence_map:
                sentence_map[groups[0]] = []
            sentence_map[groups[0]].append((int(groups[1]), int(groups[2])))
        for sentence in sentence_map:
            tmp_args = sentence_map[sentence]
            tokens = sentence.split()
            window = len(tokens)
            select_idx = -1
            for tmp_start, tmp_end in tmp_args:
                tmp_key, idx = self.check_keywords(tokens[tmp_start:tmp_end])
                idx += tmp_start
                if tmp_key is not None:
                    cur_window = tmp_end - tmp_start
                    if cur_window < window:
                        window = cur_window
                        select_idx = idx
            if select_idx > -1:
                self.f_out.write(sentence + "\t" + str(select_idx) + "\n")


class Randomizer:
    def __init__(self):
        lines = [x.strip() for x in open("samples/wikipedia/tmparg_index.txt").readlines()]
        random.shuffle(lines)
        f_out_train = open("samples/wikipedia_tmparg_full/train.formatted.txt", "w")
        f_out_test = open("samples/wikipedia_tmparg_full/test.formatted.txt", "w")
        for line in lines[:7500000]:
            f_out_train.write(line + "\n")
        for line in lines[7500000:]:
            f_out_test.write(line + "\n")


randomizer = Randomizer()

