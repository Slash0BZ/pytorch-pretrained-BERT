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
    if "how long" in question.lower():
        return "duration"
    if "how often" in question.lower():
        return "frequency"
    if "when " in question.lower() or "what day" in question.lower() or "what time" in question.lower():
        return "typical"
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


def decide_duration(answer, logits):
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
        return "no"
    if expression is None:
        number = quantity(tokens[num_pos])
        if number is None:
            number = 2.0
    else:
        number = float(expression.split()[0])
        label = expression.split()[1]
    normed_u, normed_v = normalize_timex(number, label)
    # art_data = []
    # for l in range(0, len(logits)):
    #     art_data += [l] * int(logits[l] / 0.001)
    # fit_norm_mu, fit_norm_sig = norm.fit(np.array(art_data))
    # value = logit_labels[normed_u] + (normed_v / advance_map[normed_u])
    # prob = norm.pdf(value, loc=fit_norm_mu, scale=fit_norm_sig)
    #
    # prob = logits[logit_labels[normed_u]]
    if normed_u == "centuries":
        prob = logits[logit_labels[normed_u]]
    else:
        ratio = (normed_v / advance_map[normed_u])
        prob = (1.0 - ratio) * logits[logit_labels[normed_u]] + ratio * logits[logit_labels[normed_u] + 1]

    if prob > 0.2:
        return "yes"
    return "no"


def decide_frequency(answer, logits):
    return "no"


def decide_typical(answer, logits):
    return "no"


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


get_answer_predictions()
