import jsonlines
from ccg_nlpy import local_pipeline
import re
import os
from word2number import w2n


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
            if self.check_patterns(tokens_lower):
                ret.append(sent["tokens"])
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

    def check_patterns_keyword(self, tokens):
        key_words = [
            # time of the dar
            "dawn", "morning", "noon", "afternoon", "evening", "dusk", "night", "midnight",
            # time of the week
            "monday", "tuesday", "wednesday", 'thursday', "friday", "saturday", "sunday",
            # time of the year
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
            # time of the year
            "spring", "summer", "autumn", "winter",
        ]
        for t in tokens:
            if t.lower() in key_words:
                return True
        return False

    def check_patterns_timepoint(self, tokens):
        for j, t in enumerate(tokens):
            t_lower = t.lower()
            for i in range(1, len(t) - 1):
                if t_lower[i] == ':' and t_lower[i - 1].isdigit() and t_lower[i + 1].isdigit():
                    return True
            if t == ":" and j > 0 and GigawordExtractor.quantity(tokens[j - 1]) is not None:
                return True
            if t_lower in ["am", "pm", "a.m.", "p.m.", "a.m", "p.m", "clock", "oclock", "o'clock"]:
                return True
            if j > 0:
                concat = ""
                for jj in range(j - 1, j + 2):
                    concat += tokens[jj]
                if concat.lower() in ["a.m" or "p.m"]:
                    return True
        return False

    def check_patterns(self, tokens):
        return self.check_patterns_keyword(tokens) or self.check_patterns_timepoint(tokens)

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


class SecondFilter:

    def __init__(self):
        self.source = [
            "samples/typical/all_eng_srl.jsonl",
            "samples/typical/all_eng_srl_2.jsonl",
        ]
        self.ret = []
        self.process()

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

    def check_patterns_keyword(self, tokens):
        key_words_day = [
            # time of the day
            "dawn", "morning", "noon", "afternoon", "evening", "dusk", "night", "midnight",
        ]
        key_words_week = [
            # time of the week
            "monday", "tuesday", "wednesday", 'thursday', "friday", "saturday", "sunday",
        ]
        key_words_month = [
            # time of the year
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        ]
        key_words_season = [
            # time of the year
            "spring", "summer", "autumn", "winter",
        ]
        for t in tokens:
            if t.lower() in key_words_day:
                return t.lower(), "TYP_DAY"
            if t.lower() in key_words_week:
                return t.lower(), "TYP_WK"
            if t.lower() in key_words_month:
                return t.lower(), "TYP_MON"
            if t.lower() in key_words_season:
                return t.lower(), "TYP_SEA"
        return None

    def get_stripped(self, tokens, tags, orig_verb_pos):
        new_tokens = []
        new_verb_pos = -1
        for i in range(0, len(tokens)):
            if tags[i] != "O" and "ARGM-TMP" not in tags[i]:
                # if "ARGM-TMP" in tags[i]:
                new_tokens.append(tokens[i])
            if i == orig_verb_pos:
                new_verb_pos = len(new_tokens) - 1
        return new_tokens, new_verb_pos

    def get_stripped_tmp_only(self, tokens, tags, orig_verb_pos, label):
        new_tokens = []
        new_verb_pos = -1
        for i in range(0, len(tokens)):
            if "ARGM-TMP" not in tags[i]:
                new_tokens.append(tokens[i])
            else:
                valid = True
                for j in range(i, len(tokens)):
                    if "ARGM-TMP" not in tags[j]:
                        break
                    if tokens[j].lower() == label.lower():
                        valid = False
                        break
                for j in range(i, -1, -1):
                    if "ARGM-TMP" not in tags[j]:
                        break
                    if tokens[j].lower() == label.lower():
                        valid = False
                        break
                if valid:
                    new_tokens.append(tokens[i])
            if i == orig_verb_pos:
                new_verb_pos = len(new_tokens) - 1
        return new_tokens, new_verb_pos

    def get_verb_position(self, tags):
        for i, t in enumerate(tags):
            if t == "B-V":
                return i
        return -1

    def process(self):
        lines = []
        for s in self.source:
            lines += [x.strip() for x in open(s).readlines()]
        reader = jsonlines.Reader(lines)
        label_map = {}
        for obj_list in reader:
            for obj in obj_list:
                verb_obj_select = None
                label = ""
                t = ""
                for verb in obj['verbs']:
                    argtmp_start, argtmp_end = self.get_tmp_range(verb['tags'])
                    tmp_tokens = obj['words'][argtmp_start:argtmp_end]
                    if self.check_patterns_keyword(tmp_tokens) is not None:
                        label, t = self.check_patterns_keyword(tmp_tokens)
                        verb_obj_select = verb
                        break
                if verb_obj_select is not None:
                    # stripped_tokens, verb_pos = self.get_stripped(obj['words'], verb_obj_select['tags'], self.get_verb_position(verb_obj_select['tags']))
                    stripped_tokens, verb_pos = self.get_stripped_tmp_only(obj['words'], verb_obj_select['tags'], self.get_verb_position(verb_obj_select['tags']), label)
                    self.ret.append([stripped_tokens, verb_pos, label, t])
                    if label not in label_map:
                        label_map[label] = 0
                    label_map[label] += 1
        print(label_map)

    def save_to_file(self, output_file):
        import random
        seen = set()
        unique_list = []
        for g in self.ret:
            key = " ".join(g[0]) + str(g[1]) + g[2]
            if key not in seen:
                unique_list.append(g)
            seen.add(key)

        random.shuffle(unique_list)
        f_out = open(output_file, "w")
        for i in range(0, len(unique_list) - 1, 2):
            cur = unique_list[i]
            f_out.write(" ".join(cur[0]) + "\t" + str(cur[1]) + "\t" + cur[2] + "\t" + " ".join(unique_list[i+1][0]) + "\t" + str(unique_list[i+1][1]) + "\t" + unique_list[i+1][2] + "\tTYPICAL\n")


# extractor = GigawordExtractor()
# extractor.process_path("/Volumes/SSD/gigaword/data/all", "samples/typical/all_raw.txt")


filter = SecondFilter()
filter.save_to_file("samples/joint_fullsent/typical.srl.pair.all.txt")