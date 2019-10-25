import re
import os
from bs4 import BeautifulSoup
from spacy.lang.en import English
import random
import json


class GigawordExtractor:
    def __init__(self):
        self.nlp = English()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.tokenizer = self.nlp.Defaults.create_tokenizer(self.nlp)
        self.typ_words = [
            ""
        ]
        self.typ_keys = [
            "dawn", "morning", "noon", "afternoon", "evening", "dusk", "night", "midnight",
            "monday", "tuesday", "wednesday", 'thursday', "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
            "spring", "summer", "autumn", "winter",
        ]

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
            "decade",
            "decades",
            "century",
            "centuries",
        ]

        self.ordering_keys = [
            "before", "after", "later", "ago "
        ]

        self.additional_keys = [
            "age ", "fall "
        ]

    def strip_html_labels(self, str):
        soup = BeautifulSoup(str)
        return soup.get_text()

    def get_content(self, file_name):
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
            if '<P>' in document:
                paragraphs = re.findall(r'<P>(.+?)</P>', document)
            else:
                paragraphs = re.findall(r'<TEXT>(.+?)</TEXT>', document)
            ret.append(" ".join(paragraphs))
        return ret

    def read_file(self, file_name):
        articles = self.get_content(file_name)
        ret = []
        for article in articles:
            doc = self.nlp(article)
            for i, sent in enumerate(doc.sents):
                sent = str(sent)
                ap = False
                # for key in self.duration_keys:
                #     if key in sent.lower():
                #         ap = True
                # for key in self.typ_keys:
                #     if key in sent.lower():
                #         ap = True
                # for key in self.additional_keys:
                #     if key in sent.lower():
                #         ap = True
                # for key in self.ordering_keys:
                #     if key in sent.lower():
                #         ap = True
                r = random.random()
                if r < 0.01:
                    ap = True
                if ap:
                    prev_sent = "NONE"
                    if i > 0:
                        prev_sent = str(list(doc.sents)[i - 1])
                    next_sent = "NONE"
                    if i < len(list(doc.sents)) - 1:
                        next_sent = str(list(doc.sents)[i + 1])
                    ret.append((prev_sent, sent, next_sent))
        return ret

    def process_path(self, path, out_path=None):
        f_out = None
        if out_path is not None:
            f_out = open(out_path, "w")
        all_file_paths = set()
        for root, dirs, files in os.walk(path):
            for file in files:
                all_file_paths.add(path + "/" + file)
        all_file_paths = list(all_file_paths)
        for path in all_file_paths:
            sents = self.read_file(path)
            print("Done processing " + str(path))
            for p, c, n in sents:
                f_out.write(p + "\t" + c + "\t" + n + "\n")
                f_out.flush()


class WikipediaExtractor:
    def __init__(self):
        self.nlp = English()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.tokenizer = self.nlp.Defaults.create_tokenizer(self.nlp)
        self.typ_words = [
            ""
        ]
        self.typ_keys = [
            "dawn", "morning", "noon", "afternoon", "evening", "dusk", "night", "midnight",
            "monday", "tuesday", "wednesday", 'thursday', "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
            "spring", "summer", "autumn", "winter",
        ]

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
            "decade",
            "decades",
            "century",
            "centuries",
        ]

        self.ordering_keys = [
            "before", "after", "later", "ago "
        ]

        self.additional_keys = [
            "age ", "fall "
        ]

    def strip_html_labels(self, str):
        soup = BeautifulSoup(str)
        return soup.get_text()

    def read_file(self, file_name):
        lines = [x.strip() for x in open(file_name).readlines()]
        ret = []
        articles = []
        cur_article = ""
        for line in lines:
            if line.startswith("<doc"):
                if cur_article != "":
                    articles.append(cur_article)
                    cur_article = ""
                continue
            content = self.strip_html_labels(line)
            cur_article += content + " "
        if cur_article != "":
            articles.append(cur_article)

        for article in articles:
            doc = self.nlp(article)
            for i, sent in enumerate(doc.sents):
                sent = str(sent)
                ap = False
                # for key in self.duration_keys:
                #     if key in sent.lower():
                #         ap = True
                # for key in self.typ_keys:
                #     if key in sent.lower():
                #         ap = True
                # for key in self.additional_keys:
                #     if key in sent.lower():
                #         ap = True
                # for key in self.ordering_keys:
                #     if key in sent.lower():
                #         ap = True
                """NOTE: RANDOM SKIPPING!"""
                r = random.random()
                if r < 0.01:
                    ap = True

                if ap:
                    prev_sent = "NONE"
                    if i > 0:
                        prev_sent = str(list(doc.sents)[i - 1])
                    next_sent = "NONE"
                    if i < len(list(doc.sents)) - 1:
                        next_sent = str(list(doc.sents)[i + 1])
                    ret.append((prev_sent, sent, next_sent))
        return ret

    def process_path(self, path, out_path=None):
        f_out = None
        if out_path is not None:
            f_out = open(out_path, "w")
        all_file_paths = set()
        for root, dirs, files in os.walk(path):
            for dir in dirs:
                for _, _, real_files in os.walk(path + "/" + dir):
                    for file in real_files:
                        all_file_paths.add(path + "/" + dir + "/" + file)
        all_file_paths = list(all_file_paths)
        for path in all_file_paths:
            sents = self.read_file(path)
            print("Done processing " + str(path))
            for p, c, n in sents:
                f_out.write(p + "\t" + c + "\t" + n + "\n")
                f_out.flush()


class TokenizationProcessor:

    def __init__(self):
        self.lines = [x.strip() for x in open("samples/gigaword/raw_collection_randsent_contextsent.txt").readlines()]
        self.nlp = English()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.f_out = open("samples/gigaword/raw_collection_randsent_contextsent_tokenized.txt", "w")
        self.process()

    def process(self):
        for line in self.lines:
            sents = []
            for sent in line.split("\t"):
                doc = self.nlp(sent)
                tokens = []
                for t in doc:
                    tokens.append(str(t))
                if len(tokens) == 0:
                    sents.append("NONE")
                    continue
                sents.append(" ".join(tokens))
            self.f_out.write("\t".join(sents) + "\n")


class Randomizer:
    def __init__(self):
        self.lines = [x.strip() for x in open("samples/wikipedia_randsent/test.formatted.txt").readlines()]
        random.shuffle(self.lines)
        f_out = open("samples/wikipedia_randsent/test.formatted.txt", "w")
        for l in self.lines[:100000]:
            f_out.write(l + "\n")


class MultiLingualLinker:
    def __init__(self):
        self.idmap_source = {
            "fr": "/shared/wikipedia/processed/idmap/fr2entitles",
        }
        self.idmap = {}
        self.build_idmap(['fr'])
        self.target_source = {
            "en": "samples/linking/en",
            "fr": "samples/linking/fr"
        }
        self.target_list = {}
        self.build_target_list(['en', 'fr'])
        self.outputs = {}
        self.source_path = {
            "en": "/shared/wikipedia/processed/enlink_in_pages",
            "fr": "/shared/wikipedia/processed/frlink_in_pages",
        }
        self.build_outputs(['en', 'fr'])

    def build_idmap(self, langs):
        for lang in langs:
            if lang not in self.idmap:
                self.idmap[lang] = {}
            lines = [x.strip() for x in open(self.idmap_source[lang]).readlines()]
            for line in lines:
                group = line.split("\t")
                if len(group) < 2:
                    continue
                self.idmap[lang][group[0]] = group[1]

    def build_target_list(self, langs):
        for lang in langs:
            if lang not in self.target_list:
                self.target_list[lang] = set()
            lines = [x.strip() for x in open(self.target_source[lang]).readlines()]
            for line in lines:
                self.target_list[lang].add(line)

    def build_outputs(self, langs):
        for lang in langs:
            if lang not in self.outputs:
                self.outputs[lang] = []
            all_file_paths = set()
            s_path = self.source_path[lang]
            for root, dirs, files in os.walk(s_path):
                for dir in dirs:
                    for _, _, real_files in os.walk(s_path + "/" + dir):
                        for file in real_files:
                            if not file.endswith("json"):
                                continue
                            all_file_paths.add(s_path + "/" + dir + "/" + file)
            all_file_paths = list(all_file_paths)
            for file_path in all_file_paths:
                with open(file_path) as json_file:
                    data = json.load(json_file)
                for obj in data:
                    text = obj['text']
                    spans = obj['linked_spans']
                    for span in spans:
                        start = span['start']
                        end = span['end']
                        title = span['label']
                        if lang != "en":
                            if title not in self.idmap[lang]:
                                continue
                            title = self.idmap[lang][title]
                        if title in self.target_list[lang]:
                            self.outputs[lang].append({
                                "text": text,
                                "start": start,
                                "end": end,
                                "title": title,
                            })

    def save(self, output_path):
        for lang in self.outputs:
            print(lang + " counts: " + str(len(self.outputs[lang])) + "\n")
        with open(output_path, 'w') as f:
            json.dump(self.outputs, f)


# extractor = GigawordExtractor()
# extractor.process_path("samples/gigaword/all", "samples/gigaword/raw_collection_randsent_contextsent.txt")
# extractor = WikipediaExtractor()
# extractor.process_path("/shared/wikipedia/processed/enwiki_with_links", "samples/wikipedia/raw_collection_randomsent_contextsent.txt")
p = TokenizationProcessor()
# r = Randomizer()
# m = MultiLingualLinker()
# m.save("samples/linking/outputs.json")
