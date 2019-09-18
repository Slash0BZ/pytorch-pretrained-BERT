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
            "morning",
            "afternoon",
            "night",
            "evening",
            "noon",
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
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


# extractor = GigawordExtractor()
# extractor.process_path("/home/xyzhou/Documents/gigaword/afp_eng", "samples/typical/afp_eng_raw.txt")
