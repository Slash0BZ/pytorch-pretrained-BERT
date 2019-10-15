from elasticsearch import Elasticsearch
import re
import os
import random
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

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
        self.nlp = English()
        self.tokenizer = Tokenizer(self.nlp.vocab)

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
        prob = random.random()
        if prob < 0.1:
            ret.append(p)
        # for sent in sent_view:
        #     tokens_lower = sent['tokens']
        #     if len(tokens_lower.split()) < 10:
        #         continue
        #     prob = random.random()
        #     if prob > 0.1:
        #         continue
        #     ret.append(tokens_lower)
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
                        prob = random.random()
                        if prob > 0.1:
                            continue
                        try:
                            cur_list = self.process_paragraph(p)
                            if f_out is not None:
                                for c in cur_list:
                                    f_out.write(c + "\n")
                        except Exception as e:
                            print(e)


def produce_random_sentence():
    extractor = GigawordExtractor()
    extractor.process_path("/Volumes/SSD/gigaword/data/all", "samples/ir_raw_text.txt")


def get_top_perspectives(evidence):
    es = Elasticsearch(['http://macniece.seas.upenn.edu'], port=4010)
    res = es.search(index="random_sentence_0", body={"query": {"match": {"text": evidence}}}, size=5)
    output = []
    for doc in res['hits']['hits']:
        score = doc['_score']
        print(doc['_score'])
        print(doc['_source']['text'])
        perspective_text = doc
        output.append((perspective_text, id, score))
    return output


def create_index():
    es = Elasticsearch(['http://macniece.seas.upenn.edu'], port=4010)
    index_name = "random_sentence_0"
    # es.indices.create(index_name)
    f = open("samples/ir_raw_text.txt")
    lines = [x.strip() for x in f.readlines()][717:]
    data = []
    for line in lines:
        data.append({'text': line})
    for idx, doc in enumerate(data):
        if idx % 1000 == 0:
            print("Processing id: {}".format(idx))
        es.index(index=index_name, doc_type='text', id=idx, body=doc)


# produce_random_sentence()
# create_index()
get_top_perspectives("John read the word .")
