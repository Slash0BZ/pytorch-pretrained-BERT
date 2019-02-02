import os
import re

from word2number import w2n
from ccg_nlpy import local_pipeline


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

    @staticmethod
    def num(s):
        try:
            return float(s)
        except:
            if s == 'a' or s == 'an':
                return 1.0
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
                    for i in range(0, idx - num_position):
                        modified_tokens.pop()
                    modified_tokens.append(str(value))
                    modified_tokens.append(token)
                    appended = True
            if not appended:
                modified_tokens.append(token)
        return ParagraphConverter.combine_timex(modified_tokens)

    def process_paragraph(self, paragraph):
        ret = []
        paragraph = paragraph.lower()
        ta = self.pipeline.doc(paragraph)
        sent_view = ta.get_view("SENTENCE")
        for sent in sent_view:
            processed_tokens = self.convert_sentence_with_unit(sent['tokens'].split())
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


extractor = GigawordExtractor("/Users/xuanyuzhou/Downloads/tmp/2doc")
extractor.output_to_file("samples/gigaword_big.txt")






