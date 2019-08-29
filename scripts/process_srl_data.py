import jsonlines


class SRLConverter:

    def __init__(self, input_file, ref_file):
        self.input_file = input_file
        self.ref_file = ref_file

    def get_verb_position(self, tags):
        for i, t in enumerate(tags):
            if t == "B-V":
                return i
        return -1

    def get_stripped(self, tokens, tags, orig_verb_pos):
        new_tokens = []
        new_verb_pos = -1
        for i in range(0, len(tokens)):
            if tags[i] != "O" and "ARGM-TMP" not in tags[i]:
                new_tokens.append(tokens[i])
            if i == orig_verb_pos:
                new_verb_pos = len(new_tokens) - 1
        return new_tokens, new_verb_pos

    def process(self, out_file):
        input_lines = [x.strip() for x in open(self.input_file).readlines()]
        ref_lines = [x.strip() for x in open(self.ref_file).readlines()]
        ref_objects = {}
        ref_reader = jsonlines.Reader(ref_lines)
        f_out = open(out_file, "w")
        for obj in ref_reader:
            tokens = obj['TOKENS']
            tags = obj['TAGS']
            key = []
            for i in range(0, len(tokens)):
                if "ARGM-TMP" in tags[i]:
                    continue
                key.append(tokens[i])
            key = "".join(key)
            ref_objects[key] = obj

        for line in input_lines:
            sentence = line.split("\t")[0]
            key = "".join(sentence.split())
            if key not in ref_objects:
                print(sentence)
                continue
            ref_obj = ref_objects[key]
            verb_pos = int(line.split("\t")[1])

            short_tokens, new_verb_pos = self.get_stripped(ref_obj['TOKENS'], ref_obj['TAGS'], verb_pos)
            f_out.write(" ".join(short_tokens) + "\t" + str(new_verb_pos) + "\t" + line.split("\t")[2] +
                        "\t-1\t-1\t-1\t-1\t-1\t-1\n")


# converter = SRLConverter("samples/UD_English_finetune/test.formatted.txt",
#                          "samples/duration/duration_srl_succeed.jsonl")
# converter.process("samples/UD_English_finetune/test.srl.formatted.txt")


class UDST_SRL_Converter:

    def __init__(self, input_file, ref_file):
        self.input_file = input_file
        self.ref_file = ref_file
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

    def get_verb_position(self, tags):
        for i, t in enumerate(tags):
            if t == "B-V":
                return i
        return -1

    def get_stripped(self, tokens, tags, orig_verb_pos):
        new_tokens = []
        new_verb_pos = -1
        for i in range(0, len(tokens)):
            if tags[i] != "O":
                new_tokens.append(tokens[i])
            if i == orig_verb_pos:
                new_verb_pos = len(new_tokens) - 1
        return new_tokens, new_verb_pos

    def get_seconds(self, exp):

        unit = ""
        if exp.startswith("PT") and exp[-1] == "M":
            unit = "minute"
        if exp[-1] == "M" and exp[1] != "T":
            unit = "month"
        if exp[-1] == "Y":
            unit = "year"
        if exp[-1] == "D":
            unit = "day"
        if exp[-1] == "W":
            unit = "week"
        if exp[-1] == "H":
            unit = "hour"
        if exp[-1] == "S":
            unit = "second"
        if exp.startswith("PT"):
            exp = exp[2:]
        else:
            exp = exp[1:]
        if unit == "" or exp == "NULL":
            return None
        exp = exp[:-1]
        label_num = float(exp)

        return label_num * self.convert_map[unit]

    def process(self, out_file):
        import math
        input_lines = [x.strip() for x in open(self.input_file).readlines()]
        ref_lines = [x.strip() for x in open(self.ref_file).readlines()]
        ref_objects = {}
        ref_reader = jsonlines.Reader(ref_lines)
        f_out = open(out_file, "w")
        for obj in ref_reader:
            tokens = obj['words']
            key = "".join(tokens)
            ref_objects[key] = obj

        for line in input_lines:
            sentence = line.split("\t")[0]
            key = "".join(sentence.split())
            if key not in ref_objects:
                print(sentence)
                continue
            ref_obj = ref_objects[key]
            verb_pos = int(line.split("\t")[1])

            for verb in ref_obj["verbs"]:
                if self.get_verb_position(verb['tags']) == verb_pos:
                    short_tokens, new_verb_pos = self.get_stripped(ref_obj['words'], verb['tags'], verb_pos)
                    label = line.split("\t")[2]
                    if " " not in label:
                        group = line.split("\t")
                        lower = self.get_seconds(group[2])
                        upper = self.get_seconds(group[3])
                        lower_e = math.log(lower)
                        upper_e = math.log(upper)

                        if (lower_e + upper_e) / 2.0 >= 11.367:
                            label = "1 years"
                        else:
                            label = "1 hours"
                    f_out.write(" ".join(short_tokens) + "\t" + str(new_verb_pos) + "\t" + label +
                                "\t-1\t-1\t-1\t-1\t-1\t-1\n")
                    break


converter = UDST_SRL_Converter("samples/duration/timebank_svo.txt",
                               "samples/duration/timebank_all_srl.jsonl")
converter.process("samples/duration/timebank_verb_srl.txt")
