from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from word2number import w2n
import jsonlines
import time


class FirstProcessor:

    def __init__(self):
        self.file_list = [
            "samples/duration/afp_eng.txt",
            "samples/duration/apw_eng.txt",
            "samples/duration/nyt_eng.txt",
        ]
        self.ret = []
        for fname in self.file_list:
            self.process_file(fname)
        f_out = open("samples/frequency/step_1_output.txt", "w")
        for l in self.ret:
            f_out.write(l + "\n")

    def process_file(self, fname):
        key_words = [
            "times",
            "time",
            "once",
            "twice",
            "every",
        ]
        lines = [x.strip() for x in open(fname).readlines()]
        for line in lines:
            tokens = line.split()
            for t in tokens:
                if t.lower() in key_words:
                    self.ret.append(line)


class PretrainedModel:
    """
    A pretrained model is determined by both an archive file
    (representing the trained model)
    and a choice of predictor.
    """
    def __init__(self, archive_file: str, predictor_name: str) -> None:
        self.archive_file = archive_file
        self.predictor_name = predictor_name

    def predictor(self) -> Predictor:
        archive = load_archive(self.archive_file)
        return Predictor.from_archive(archive, self.predictor_name)


class Tokenizer:

    def split_words(self, sentence):
        return sentence.split()


class AllenSRL:

    def __init__(self, output_path):
        model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz',
                                'semantic-role-labeling')
        self.predictor = model.predictor()
        self.predictor._model = self.predictor._model.cuda()
        self.output_path = output_path

    def predict_batch(self, sentences):
        f_out = jsonlines.open(self.output_path, "w")
        counter = 0
        start_time = time.time()
        batch_size = 128
        for i in range(0, len(sentences), batch_size):
            input_map = []
            for j in range(0, batch_size):
                input_map.append({"sentence": sentences[i+j]})
            prediction = self.predictor.predict_batch_json(input_map)
            f_out.write(prediction)
            counter += 1
            if counter % 10 == 0:
                print("Average Time: " + str((time.time() - start_time) / (10.0 * float(batch_size))))
                start_time = time.time()

    def predict_single(self, instances):
        f_out = jsonlines.open(self.output_path, "w")
        counter = 0
        start_time = time.time()
        for instance in instances:
            prediction = self.predictor.predict_tokenized(instance.split())
            f_out.write(prediction)
            counter += 1
            if counter % 10 == 0:
                print("Average Time: " + str((time.time() - start_time) / 10.0))
                start_time = time.time()

    def predict_file(self, path):
        with open(path) as f:
            lines = [x.strip() for x in f.readlines()]
        new_lines = []
        for l in lines:
            if len(l.split()) < 150:
                new_lines.append(l)
        self.predict_batch(new_lines)


class SecondProcessor:
    def __init__(self):
        self.input_file = "samples/frequency/step_1_srl.jsonl"
        self.ret = []
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
            "decades"
            "century",
            "centuries",
        ]
        self.freq_conversion = {
            "once": 1.0,
            "twice": 2.0,
        }
        self.process()

    def get_trivial_floats(self, s):
        try:
            if s == "a" or s == "an":
                return 1.0
            n = float(s)
            return n
        except:
            return None

    def quantity(self, s):
        try:
            if self.get_trivial_floats(s) is not None:
                return self.get_trivial_floats(s)
            cur = w2n.word_to_num(s)
            if cur is not None:
                return float(cur)
            return None
        except:
            return None

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

    def validate_argtmp(self, tmp_tokens):
        unit = ""
        num = -1
        unit_pos = -1
        for i, t in enumerate(tmp_tokens):
            if t.lower() in self.duration_keys:
                if i == 0:
                    continue
                num = self.quantity(tmp_tokens[i - 1])
                if num is None:
                    continue
                unit = t.lower()
                unit_pos = i
                break
        if unit == "":
            return None
        for i, t in enumerate(tmp_tokens):
            t = t.lower()
            if t == "every" and abs(i - unit_pos) < 5:
                return num, unit
            if t in self.freq_conversion and abs(i - unit_pos) < 5:
                return num / self.freq_conversion[t], unit
            if t == "times" and i > 0 and abs(i - unit_pos) < 5:
                q = self.quantity(tmp_tokens[i - 1].lower())
                if q is not None:
                    return num / float(q), unit
        return None

    def get_stripped(self, tokens, tags, orig_verb_pos):
        new_tokens = []
        new_verb_pos = -1
        for i in range(0, len(tokens)):
            if tags[i] != "O" and "ARGM-TMP" not in tags[i]:
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
        lines = [x.strip() for x in open(self.input_file).readlines()]
        reader = jsonlines.Reader(lines)
        for obj_list in reader:
            for obj in obj_list:
                verb_obj_select = None
                num = -1
                unit = ""
                for verb in obj['verbs']:
                    argtmp_start, argtmp_end = self.get_tmp_range(verb['tags'])
                    tmp_tokens = obj['words'][argtmp_start:argtmp_end]
                    validation = self.validate_argtmp(tmp_tokens)
                    if validation is not None:
                        verb_obj_select = verb
                        num = validation[0]
                        unit = validation[1]
                        break
                if verb_obj_select is not None:
                    stripped_tokens, verb_pos = self.get_stripped(obj['words'], verb_obj_select['tags'], self.get_verb_position(verb_obj_select['tags']))
                    self.ret.append([stripped_tokens, verb_pos, str(num) + " " + unit])
                    # print(" ".join(stripped_tokens) + "\t" + str(verb_pos))
                    # print(str(num) + " " + unit)
                    # print()

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
            f_out.write(" ".join(cur[0]) + "\t" + str(cur[1]) + "\t" + cur[2] + "\t" + " ".join(unique_list[i+1][0]) + "\t" + str(unique_list[i+1][1]) + "\t" + unique_list[i+1][2] + "\tFREQ\n")


# srl = AllenSRL("samples/typical/afp_eng_raw_srl_2.jsonl")
# srl.predict_file("samples/typical/afp_eng_raw.txt")
# second_processor = SecondProcessor()
# second_processor.save_to_file("samples/pretrain_combine/freq.srl.pair.all.txt")

def insert_duration_data(limit, output_file):
    import random
    lines = [x.strip() for x in open("samples/UD_English_finetune/train.srl.pair.formatted.txt").readlines()]
    random.shuffle(lines)
    f_out = open(output_file, "w")
    for line in lines[:limit]:
        f_out.write("\t".join(line.split("\t")[:-1]) + "\tDUR\n")


insert_duration_data(15937, "samples/pretrain_combine/dur.srl.pair.all.txt")
