import jsonlines
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from torch import nn
from word2number import w2n
import torch
import os
import numpy as np
from scipy.stats import norm
from scripts import tmparg_processor


class TemporalModelJointEval(BertPreTrainedModel):
    def __init__(self, config, num_labels, num_typical_labels):
        super(TemporalModelJointEval, self).__init__(config)
        self.num_labels = num_labels
        self.num_typical_labels = num_typical_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.log_softmax = nn.LogSoftmax(-1)
        self.softmax = nn.Softmax(-1)

        self.dur_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.freq_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.typical_classifier = nn.Linear(config.hidden_size, self.num_typical_labels)

        self.apply(self.init_bert_weights)

    def get_single_inference(self, input_ids, token_type_ids, attention_mask, target_ids):
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        target_all_output = seq_output.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, seq_output.size(2)))

        pooled_output = self.dropout(target_all_output)

        dur_logits = self.softmax(self.dur_classifier(pooled_output)).view(-1, self.num_labels)
        freq_logits = self.softmax(self.freq_classifier(pooled_output)).view(-1, self.num_labels)
        typical_logits = self.softmax(self.typical_classifier(pooled_output)).view(-1, self.num_typical_labels)

        return freq_logits, dur_logits, typical_logits

    def forward(self, input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a):
        return self.get_single_inference(input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a)


class Annotator:

    def __init__(self, bert_model, num_labels):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))
        self.model = TemporalModelJointEval.from_pretrained(bert_model,
                                                        cache_dir=cache_dir,
                                                        num_labels=9,
                                                        num_typical_labels=31).to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.max_seq_length = 128

    def annotate(self, sentence, verb_pos):
        tokens = sentence.lower().split()
        for i, _ in enumerate(tokens):
            if tokens[i].lower() not in self.tokenizer.vocab:
                tokens[i] = "[UNK]"

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.device)
        target_idx = torch.tensor([verb_pos + 1], dtype=torch.long).to(self.device)

        with torch.no_grad():
            freq_logits, dur_logits, typ_logits = self.model(
                input_ids, segment_ids, input_mask, target_idx
            )

        return freq_logits.detach().cpu().numpy(), dur_logits.detach().cpu().numpy(), typ_logits.detach().cpu().numpy()


def get_verb_pos(tags):
    for i, t in enumerate(tags):
        if t == "B-V":
            return i
    return -1


def get_stripped(tokens, tags, orig_verb_pos):
    new_tokens = []
    new_verb_pos = -1
    for i in range(0, len(tokens)):
        if tags[i] != "O" and "ARGM-TMP" not in tags[i]:
            new_tokens.append(tokens[i])
        if i == orig_verb_pos:
            new_verb_pos = len(new_tokens) - 1
    return new_tokens, new_verb_pos


def process_dev_question():
    lines = [x.strip() for x in open("samples/unsupervised_mctaco/dev_questions_srl.jsonl").readlines()]
    reader = jsonlines.Reader(lines)
    f_out = open("samples/unsupervised_mctaco/dev_questions_mapping.txt", "w")
    for obj_list in reader:
        for obj in obj_list:
            max_verb_pos = -1
            max_verb = None
            for verb in obj['verbs']:
                verb_pos = get_verb_pos(verb['tags'])
                if verb_pos > max_verb_pos:
                    max_verb_pos = verb_pos
                    max_verb = verb
            if max_verb is None:
                continue
            print(max_verb)
            stripped_tokens, verb_pos = get_stripped(obj['words'], max_verb['tags'], max_verb_pos)
            f_out.write(" ".join(obj['words']) + "\t" + " ".join(stripped_tokens) + "\t" + str(verb_pos) + "\n")


def get_predictions():
    annotator = Annotator("bert_joint_epoch_1", 31)
    f_out = open("samples/unsupervised_mctaco/dev_questions_predictions.txt", "w")
    lines = [x.strip() for x in open("samples/unsupervised_mctaco/dev_questions_mapping.txt")]
    for line in lines:
        srl_form = line.split("\t")[1]
        verb_pos = int(line.split("\t")[2])
        freq, dur, typ = annotator.annotate(srl_form, verb_pos)
        freq = list(freq[0])
        dur = list(dur[0])
        typ = list(typ[0])
        outs = [str(x) for x in freq + dur + typ]
        f_out.write(line.split("\t")[0] + "\t" + " ".join(outs) + "\n")


def get_question_type(question):
    if "how long" in question.lower() or "how many years" in question.lower():
        return "Event Duration"
    if "how often" in question.lower():
        return "Frequency"
    if "how many times" in question.lower():
        return "Frequency"
    if "what age" in question.lower() or "when " in question.lower() or "what day" in question.lower() or "what time" in question.lower() or "what year" in question.lower():
        return "Typical Time"
    if "which year" in question.lower():
        return "Typical Time"
    return "unknown"


def get_trivial_floats(s):
    try:
        if s == "a" or s == "an":
            return 1.0
        n = float(s)
        return n
    except:
        return None


def quantity(s):
    try:
        if get_trivial_floats(s) is not None:
            return get_trivial_floats(s)
        cur = w2n.word_to_num(s)
        if cur is not None:
            return float(cur)
        return None
    except:
        return None


def normalize_timex(v_input, u):
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
        if seconds / convert_map[v] < 0.5:
            break
        prev_unit = v
    if prev_unit == "seconds" and seconds > 60.0:
        prev_unit = "centuries"
    new_val = int(seconds / convert_map[prev_unit])

    return prev_unit, new_val


def decide_duration(answer, tokenizer):
    logit_labels = {
        "seconds": 0,
        "minutes": 1,
        "hours": 2,
        "days": 3,
        "weeks": 4,
        "months": 5,
        "years": 6,
        "decades": 7,
        "centuries": 8,
    }
    m = {
        "second": "seconds",
        "minute": "minutes",
        "hour": "hours",
        "day": "days",
        "week": "weeks",
        "month": "months",
        "year": "years",
        "decade": "decades",
        "century": "centuries",
    }
    others = {
        "night": "8.0 hours",
        "generation": "1.0 decades",
        "morning": "4.0 hours",
        "lives": "5.0 centuries",
        "nanosecond": "1.0 seconds"
    }
    advance_map = {
        "seconds": 60.0,
        "minutes": 60.0,
        "hours": 24.0,
        "days": 7.0,
        "weeks": 30.0,
        "months": 12.0,
        "years": 10.0,
        "decades": 10.0,
        "centuries": 10.0,
    }
    tokens = answer.lower().split()
    label = "NONE"
    expression = None
    num_pos = -1
    for i, t in enumerate(tokens):
        if t in logit_labels:
            label = t
            num_pos = i - 1
        if t in m:
            label = m[t]
            num_pos = i - 1
        if t in others:
            expression = others[t]
    if (label == "NONE" or num_pos == -1) and expression is None:
        return None
    if expression is None:
        number = quantity(tokens[num_pos])
        if number is None:
            number = 2.0
    else:
        number = float(expression.split()[0])
        label = expression.split()[1]
    normed_u, normed_v = normalize_timex(number, label)

    return "[unused501] " + tokenizer.ids_to_tokens[logit_labels[normed_u] + 1]


def decide_frequency(answer, tokenizer, filter):
    logit_labels = {
        "seconds": 0,
        "minutes": 1,
        "hours": 2,
        "days": 3,
        "weeks": 4,
        "months": 5,
        "years": 6,
        "decades": 7,
        "centuries": 8,
    }
    check = filter.check_frequency_sentences(answer.split())
    l = [43, 44, 45, 46, 47, 48, 49, 50, 51]
    if check in ["SKIP_DURATION", "NO_UNIT_FOUND", "FOUND_UNIT_BUT_NOT_FREQUENCY"]:
        return None
    normed_u, _ = normalize_timex(float(check.split()[0]), check.split()[1])

    return "[unused502] " + tokenizer.ids_to_tokens[l[logit_labels[normed_u]]]


def decide_typical(answer, tokenizer, filter):
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
    vocab_indices = {
        1: [10, 11, 12, 13, 14, 15, 16, 17],
        2: [18, 19, 20, 21, 22, 23, 24],
        3: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
        4: [37, 38, 39, 40],
    }
    check = filter.check_typical_sentences(answer.split())
    unit, group = check
    if unit == "NO_TYPICAL_FOUND":
        return None
    return "[unused503] " + tokenizer.ids_to_tokens[vocab_indices[group][keywords[unit][1]]]


def get_answer_predictions():
    logit_lines = [x.strip() for x in open("samples/unsupervised_mctaco/dev_questions_predictions.txt").readlines()]
    logit_map = {}
    for line in logit_lines:
        key = line.split("\t")[0].replace(" ", "").lower()
        logits = [float(x) for x in line.split("\t")[1].split()]
        logit_map[key] = {
            "freq": logits[0:9],
            "dur": logits[9:18],
            "typ": logits[18:]
        }
    source_lines = [x.strip() for x in open("samples/unsupervised_mctaco/dev.tsv").readlines()]
    output_labels = []
    for line in source_lines:
        groups = line.split("\t")
        sentence = groups[0]
        question = groups[1]
        answer = groups[2]
        gold_question_type = groups[-1]
        """ONLY DURATION"""
        if gold_question_type != "Event Duration":
            output_labels.append("no")
            continue
        question_key = question.replace(" ", "").lower()
        logits = logit_map[question_key]
        question_type = get_question_type(question)
        if question_type == "unknown":
            """NOTE: ALWAYS NO!"""
            output_labels.append("no")
        elif question_type == "duration":
            output_labels.append(decide_duration(answer, logits['dur']))
        elif question_type == "frequency":
            output_labels.append(decide_frequency(answer, logits['freq']))
        elif question_type == "typical":
            output_labels.append(decide_typical(answer, logits['typ']))
        else:
            output_labels.append("no")

    f_out = open("tmp_outputs.txt", "w")
    for o in output_labels:
        f_out.write(o + "\n")


def print_argmax_outputs():
    logit_lines = [x.strip() for x in open("samples/unsupervised_mctaco/dev_questions_predictions.txt").readlines()]
    labels_typical = [
        # time of the dar
        "dawn", "morning", "noon", "afternoon", "evening", "dusk", "night", "midnight",
        # time of the week
        "monday", "tuesday", "wednesday", 'thursday', "friday", "saturday", "sunday",
        # time of the year
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
        "december",
        # time of the year
        "spring", "summer", "autumn", "winter",
    ]

    labels_unit = ["seconds", "minutes", "hours", "days", "weeks", "months", "years", "decades", "centuries"]

    f_out = open("output.txt", "w")
    for line in logit_lines:
        logits = [float(x) for x in line.split("\t")[1].split()]
        freq = int(np.argmax(logits[0:9]))
        dur = int(np.argmax(logits[9:18]))
        typ = logits[18:]
        typ_day = int(np.argmax(typ[0:8]))
        typ_week = int(np.argmax(typ[8:15]))
        typ_year = int(np.argmax(typ[15:27]))
        typ_season = int(np.argmax(typ[27:31]))
        out_content = [labels_unit[freq], labels_unit[dur],
                       labels_typical[0:8][typ_day], labels_typical[8:15][typ_week], labels_typical[15:27][typ_year], labels_typical[27:31][typ_season]]
        f_out.write("\t".join(out_content) + "\n")


def transform_files():
    source_lines = [x.strip() for x in open("samples/unsupervised_mctaco/test.tsv").readlines()]
    total = 0.0
    correct = 0.0
    filter = tmparg_processor.TmpArgDimensionFilter()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    f_out = open("samples/split_30_70_marked/test.txt", "w")
    for line in source_lines:
        groups = line.split("\t")
        sentence = groups[0]
        question = groups[1]
        answer = groups[2]
        question_key = question.replace(" ", "").lower()
        gold_question_type = groups[-1]
        question_type = get_question_type(question)
        additional_string = ""
        if question_type == "Event Duration":
            additional_string = decide_duration(answer, tokenizer)
        if question_type == "Frequency":
            additional_string = decide_frequency(answer, tokenizer, filter)
        if question_type == "Typical Time":
            additional_string = decide_typical(answer, tokenizer, filter)
        if additional_string is None:
            additional_string = ""
        answer += " " + additional_string
        f_out.write(sentence + "\t" + question + "\t" + answer + "\t" + groups[-2] + "\n")



    #     if gold_question_type in ["Event Duration", "Frequency", "Typical Time"]:
    #         type_total += 1.0
    #         if question_type == gold_question_type:
    #             type_correct += 1.0
    #         else:
    #             print(question)
    #             print(question_type)
    #             print(gold_question_type)
    # print(type_correct / type_total)

transform_files()
