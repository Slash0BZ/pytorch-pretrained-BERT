from pytorch_pretrained_bert import BertTokenizer
import random
import spacy

keywords = {
    "second": [0, 0],
    "minute": [0, 1],
    "hour": [0, 2],
    "day": [0, 3],
    "week": [0, 4],
    "month": [0, 5],
    "year": [0, 6],
    "decade": [0, 7],
    "century": [0, 8],
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
    "spring": [4, 0],
    "summer": [4, 1],
    "fall": [4, 2],
    "winter": [4, 3],
    "after": [5, 0],
    "before": [5, 1],
    "while": [5, 2],
    "during": [5, 2],
    "when": [5, 3],
    "second_1": [6, 0],
    "minute_1": [6, 1],
    "hour_1": [6, 2],
    "day_1": [6, 3],
    "week_1": [6, 4],
    "month_1": [6, 5],
    "year_1": [6, 6],
    "decade_1": [6, 7],
    "century_1": [6, 8],
    "second_2": [7, 0],
    "minute_2": [7, 1],
    "hour_2": [7, 2],
    "day_2": [7, 3],
    "week_2": [7, 4],
    "month_2": [7, 5],
    "year_2": [7, 6],
    "decade_2": [7, 7],
    "century_2": [7, 8],
}

vocab_indices = {
    0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    1: [10, 11, 12, 13, 14, 15, 16, 17],
    2: [18, 19, 20, 21, 22, 23, 24],
    3: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    4: [37, 38, 39, 40],
    5: [41, 42, 61, 62],
    6: [43, 44, 45, 46, 47, 48, 49, 50, 51],
    7: [52, 53, 54, 55, 56, 57, 58, 59, 60],
}
reverse_map = {}
reverse_dim_map = {}

for key in keywords:
    group, index = keywords[key]
    vocab_index = vocab_indices[group][index]
    if "_" in key:
        key = key.split("_")[0]
    reverse_map[vocab_index] = key

dim_map = {
    0: "Duration",
    1: "TimeOfDay",
    2: "TimeOfWeek",
    3: "Month",
    4: "Season",
    5: "Ordering",
    6: "Frequency",
    7: "Boundary"
}
reverse_naming_map = {}
for key in dim_map:
    reverse_naming_map[dim_map[key]] = key

for key in vocab_indices:
    for val in vocab_indices[key]:
        reverse_dim_map[val] = key



def parse_line(line, tokenizer):
    group = line.split("\t")
    tokens = group[0].split()
    masks = [int(x) for x in group[4].split()]
    target_id = int(group[1])
    soft_labels = [int(x) for x in group[2].split()]
    soft_label_values = [float(x) for x in group[3].split()]
    sep_count = 0
    key_tokens = []
    for i, token in enumerate(tokens):
        if token == "[SEP]":
            sep_count += 1
        if sep_count == 1:
            if masks[i] > -1:
                actual_token = tokenizer.ids_to_tokens[masks[i]]
                key_tokens.append(actual_token)
            else:
                key_tokens.append(token)
    has_mask = False
    for t in key_tokens:
        if t == "[MASK]":
            has_mask = True

    max_index = -1
    max_val = -1.0
    for i, label in enumerate(soft_labels):
        if soft_label_values[i] > max_val:
            max_val = soft_label_values[i]
            max_index = label

    count = 0
    val_token = "[MASK]"
    for i, token in enumerate(tokens):
        if token == "[unused500]":
            count += 1
        if count == 2:
            val_token = tokens[i + 2]
            break

    if target_id > 0 or val_token == "[MASK]":
        assert max_index > 0
        val_token = tokenizer.ids_to_tokens[max_index]

    assert val_token != "[MASK]"

    val_id = max_index

    value = reverse_map[val_id]
    dimension = dim_map[reverse_dim_map[val_id]]

    for i, t in enumerate(key_tokens):
        if t not in tokenizer.vocab:
            key_tokens[i] = "[MASK]"

    return " ".join(key_tokens), dimension, value, line


def produce_lines():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    training_lines = [x.strip() for x in open("samples/gigaword/tmp_seq_multi_sent_allmask/train.formatted.txt")]

    # seen_set = set()
    # for line in training_lines:
    #     sentence, _, _, _ = parse_line(line, tokenizer)
    #     seen_set.add(sentence)
    #
    # test_lines = [x.strip() for x in open("samples/gigaword/lm_format_allmask_full.txt").readlines()]

    seen_count = 0
    selection_map = {}
    f_out = open("samples/gigaword/tmp_seq_multi_sent_allmask/train.formatted.txt", "w")
    for line in training_lines:
        sentence, dimension, value, orig_line = parse_line(line, tokenizer)
        if dimension not in selection_map:
            selection_map[dimension] = 0
        selection_map[dimension] += 1
        threshold = 1000000
        if dimension in ["TimeOfDay", "TimeOfWeek", "Month", "Season"]:
            threshold = 250000
        if selection_map[dimension] > threshold:
            continue
        f_out.write(line + "\n")
        # if sentence in seen_set:
        #     seen_count += 1
        #     continue
        # key = dimension + " " + value
        # if key not in selection_map:
        #     selection_map[key] = []

        # selection_map[key].append([sentence, dimension, value, orig_line])

    # f_out = open("samples/gigaword/to_annotate.txt", "w")
    # for key in selection_map:
    #     instances = selection_map[key]
    #     random.shuffle(instances)
    #     for s, d, v, orig_line in instances[:500]:
    #         f_out.write("\t".join([s, d, v, orig_line]) + "\n")


def format_lines():
    lines = [x.strip() for x in open("samples/gigaword/to_annotate.txt").readlines()]
    f_out = open("samples/gigaword/to_annotate_formatted.txt", "w")
    count_map = {}
    total_inst_per_group = 10000000
    random.shuffle(lines)
    for line in lines:
        group = line.split("\t")
        dimension = group[1]
        dim_number = reverse_naming_map[dimension]
        addition = ""
        if dim_number == 6:
            addition = "_1"
        if dim_number == 7:
            addition = "_2"

        _, value_number = keywords[group[2] + addition]

        if dim_number == 7 or dim_number == 5:
            continue
        num_of_labels_in_group = len(vocab_indices[dim_number])
        avg_label = int(float(total_inst_per_group) / float(num_of_labels_in_group))

        if dim_number not in count_map:
            count_map[dim_number] = {}

        if value_number not in count_map[dim_number]:
            count_map[dim_number][value_number] = 0

        mask_count = 0.0
        for t in group[0].split():
            if t == "[MASK]":
                mask_count += 1.0
        if mask_count / float(len(group[0].split())) > 0.1:
            continue

        if count_map[dim_number][value_number] > avg_label:
            continue

        if group[0][0] == "\"" or group[0][0] == "'":
            continue

        label = group[2]
        dimension = ""
        other_labels = []
        for idx in vocab_indices[dim_number]:
            other_label_surface = reverse_map[idx]
            if "_" in other_label_surface:
                other_label_surface = other_label_surface.split("_")[0]
            if other_label_surface == label:
                continue
            other_labels.append(other_label_surface)

        other_label_str = "[" + "/".join(other_labels) + "]"
        if dim_number == 0:
            label = "1 " + label
            dimension = "typical duration"
            other_label_str = "1 [" + "/".join(other_labels) + "]"
        if dim_number == 1:
            dimension = "typical occurring time of the day"
        if dim_number == 2:
            dimension = "typical occurring day of the week"
        if dim_number == 3:
            dimension = "typical occurring month"
        if dim_number == 4:
            dimension = "typical occurring season"
        if dim_number == 6:
            dimension = "typical frequency"
            label = "every " + label
            other_label_str = "every [" + "/".join(other_labels) + "]"

        orig_tokens = group[3].split()
        additor = 0
        for i, t in enumerate(orig_tokens):
            if t == "[SEP]":
                additor = i + 1
                break

        verb_tag = False
        verb_surface = ""
        tokens = []
        for i, t in enumerate(group[0].split()[1:]):
            if t == "[unused500]":
                verb_tag = True
                continue
            if verb_tag:
                verb_surface = t
                tokens.append("<font color='red'>" + t + "</font>")
                verb_tag = False
            else:
                if t == "[MASK]":
                    tokens.append(orig_tokens[i + additor])
                else:
                    tokens.append(t)

        has_mask = False
        for t in tokens:
            if t == "[MASK]":
                has_mask = True
                break
        if has_mask:
            continue

        if verb_surface == "[MASK]" or verb_surface == "" or verb_surface == " " or verb_surface == "the":
            continue
        if tokens[0] == '"':
            continue

        # Conditional Check Passed
        count_map[dim_number][value_number] += 1

        ret_tokens = []
        for t in tokens:
            if t.startswith("##"):
                ret_tokens[-1] = ret_tokens[-1] + t.replace("##", "")
            else:
                ret_tokens.append(t)

        f_out.write("\t".join([" ".join(ret_tokens).strip(), verb_surface, dimension, label, other_label_str]) + "\n")


def prepare_udst_annotation():
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = nlp.tokenizer.tokens_from_list
    lines = [x.strip() for x in open("samples/UD_English_untouched/test.formatted.txt").readlines()]
    sent_map = {}
    f_out = open("samples/intrinsic/udst_to_annotate.tsv", "w")
    value_map = {
        "seconds": 0.0,
        "minutes": 1.0,
        "hours": 2.0,
        "days": 3.0,
        "weeks": 4.0,
        "months": 5.0,
        "years": 6.0,
        "decades": 7.0,
        "centuries": 8.0
    }
    for line in lines:
        sentence = line.split("\t")[0].lower()
        verb_pos = int(line.split("\t")[1])
        key = sentence + " " + str(verb_pos)
        label = line.split("\t")[2]
        if "forever" in label or "instantaneous" in label:
            continue
        if key not in sent_map:
            sent_map[key] = []
        sent_map[key].append(value_map[label.split()[1]])
    label_map = {}
    prepared_lines = set()
    for line in lines:
        sentence = line.split("\t")[0].lower()
        verb_pos = int(line.split("\t")[1])
        key = sentence + " " + str(verb_pos)
        diff_sum = 0.0
        diff_count = 0.0
        if key not in sent_map:
            continue
        for i in range(0, len(sent_map[key])):
            for j in range(i, len(sent_map[key])):
                diff_sum += abs(sent_map[key][i] - sent_map[key][j])
                diff_count += 1.0
        if diff_count > 0.0:
            if diff_sum / diff_count > 2.0:
                continue

        label = line.split("\t")[2]
        if "forever" in label or "instantaneous" in label:
            continue
        if label not in label_map:
            label_map[label] = 0
        doc = nlp(sentence.split())
        counter = 0
        cont = False
        for tok in doc:
            if counter == verb_pos:
                if tok.pos_ != "VERB":
                    cont = True
                    break
            counter += 1
        if cont:
            continue

        if label_map[label] >= 100:
            continue
        tokens = sentence.split()
        verb_form = tokens[verb_pos]
        tokens[verb_pos] = "<font color='red'>" + tokens[verb_pos] + "</font>"
        other_labels = []
        for k in value_map:
            if k not in label:
             other_labels.append(k)
        other_label_str = "1 [" + "/".join(other_labels) + "]"
        old_len = len(prepared_lines)
        prepared_lines.add(" ".join(tokens) + "\t" + verb_form + "\t" + "typical duration" + "\t" + label + "\t" + other_label_str + "\n")
        new_len = len(prepared_lines)
        if new_len > old_len:
            label_map[label] += 1

    prepared_lines = list(prepared_lines)
    for l in prepared_lines:
        f_out.write(l)


def word_piece_tokenize(tokens, verb_pos, tokenizer):
    if verb_pos < 0:
        return None, -1

    ret_tokens = []
    ret_verb_pos = -1
    for i, token in enumerate(tokens):
        token = token.lower()
        if i == verb_pos:
            ret_verb_pos = len(ret_tokens)
        sub_tokens = tokenizer.tokenize(token)
        ret_tokens.extend(sub_tokens)

    return ret_tokens, ret_verb_pos

def get_soft_labels(orig_label):
    markers = ["_dur", "_freq", "_bnd"]
    keywords = {
        "second" + markers[0]: [0, 0],
        "seconds" + markers[0]: [0, 0],
        "minute" + markers[0]: [0, 1],
        "minutes" + markers[0]: [0, 1],
        "hour" + markers[0]: [0, 2],
        "hours" + markers[0]: [0, 2],
        "day" + markers[0]: [0, 3],
        "days" + markers[0]: [0, 3],
        "week" + markers[0]: [0, 4],
        "weeks" + markers[0]: [0, 4],
        "month" + markers[0]: [0, 5],
        "months" + markers[0]: [0, 5],
        "year" + markers[0]: [0, 6],
        "years" + markers[0]: [0, 6],
        "decade" + markers[0]: [0, 7],
        "decades" + markers[0]: [0, 7],
        "century" + markers[0]: [0, 8],
        "centuries" + markers[0]: [0, 8],
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
        "falls": [4, 2],
        "autumns": [4, 2],
        "winters": [4, 3],
        "spring": [4, 0],
        "summer": [4, 1],
        "autumn": [4, 2],
        "fall": [4, 2],
        "winter": [4, 3],
        "after": [5, 0],
        "before": [5, 1],
        "while": [5, 2],
        "during": [5, 2],
        "when": [5, 3],
        "second" + markers[1]: [6, 0],
        "seconds" + markers[1]: [6, 0],
        "minute" + markers[1]: [6, 1],
        "minutes" + markers[1]: [6, 1],
        "hour" + markers[1]: [6, 2],
        "hours" + markers[1]: [6, 2],
        "day" + markers[1]: [6, 3],
        "days" + markers[1]: [6, 3],
        "week" + markers[1]: [6, 4],
        "weeks" + markers[1]: [6, 4],
        "month" + markers[1]: [6, 5],
        "months" + markers[1]: [6, 5],
        "year" + markers[1]: [6, 6],
        "years" + markers[1]: [6, 6],
        "decade" + markers[1]: [6, 7],
        "decades" + markers[1]: [6, 7],
        "century" + markers[1]: [6, 8],
        "centuries" + markers[1]: [6, 8],
        "second" + markers[2]: [7, 0],
        "seconds" + markers[2]: [7, 0],
        "minute" + markers[2]: [7, 1],
        "minutes" + markers[2]: [7, 1],
        "hour" + markers[2]: [7, 2],
        "hours" + markers[2]: [7, 2],
        "day" + markers[2]: [7, 3],
        "days" + markers[2]: [7, 3],
        "week" + markers[2]: [7, 4],
        "weeks" + markers[2]: [7, 4],
        "month" + markers[2]: [7, 5],
        "months" + markers[2]: [7, 5],
        "year" + markers[2]: [7, 6],
        "years" + markers[2]: [7, 6],
        "decade" + markers[2]: [7, 7],
        "decades" + markers[2]: [7, 7],
        "century" + markers[2]: [7, 8],
        "centuries" + markers[2]: [7, 8],
    }

    return keywords[orig_label][1]


def format_model_lines(sentence, dimension, value, verb_pos=None):
    sentence = sentence.replace("<font color='red'>", "**##")
    sentence = sentence.replace("</font>", "")

    tokens = sentence.split()
    if verb_pos is None:
        verb_pos = -1
        for i, t in enumerate(tokens):
            if t.startswith("**##"):
                verb_pos = i
                tokens[i] = tokens[i].replace("**##", "")
                break

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, do_basic_tokenize=False)
    tokens, verb_pos = word_piece_tokenize(tokens, verb_pos, tokenizer)
    tokens.insert(verb_pos, "[unused500]")

    key_tok = None
    candidates = []
    dim = -1
    if dimension == "typical duration":
        key_tok = "[unused501]"
        candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        value = value.split()[1] + "_dur"
        dim = 0
    if dimension == "typical frequency":
        key_tok = "[unused502]"
        candidates = [43, 44, 45, 46, 47, 48, 49, 50, 51]
        value = value.split()[1] + "_freq"
        dim = 1
    if dimension == "typical occurring time of the day":
        key_tok = "[unused503]"
        candidates = [10, 11, 12, 13, 14, 15, 16, 17]
        dim = 2
    if dimension == "typical occurring day of the week":
        key_tok = "[unused503]"
        candidates = [18, 19, 20, 21, 22, 23, 24]
        dim = 3
    if dimension == "typical occurring month":
        key_tok = "[unused503]"
        candidates = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        dim = 4
    if dimension == "typical occurring season":
        key_tok = "[unused503]"
        candidates = [37, 38, 39, 40]
        dim = 5

    label = get_soft_labels(value)

    seq = "[CLS] " + " ".join(tokens) + " [SEP] [unused500] " + key_tok + " [MASK] [SEP]"

    f_out = open("samples/intrinsic/model.dur.txt", "a+")
    candidates = [str(x) for x in candidates]

    f_out.write(seq + "\t" + str(len(seq.split()) - 2) + "\t" + str(label) + "\t" + " ".join(candidates) + "\t" + str(dim) + "\n")


def format_bert_lines(sentence, dimension, value, verb_pos=None):
    sentence = sentence.replace("<font color='red'>", "**##")
    sentence = sentence.replace("</font>", "")

    tokens = sentence.split()
    if verb_pos is None:
        verb_pos = -1
        for i, t in enumerate(tokens):
            if t.startswith("**##"):
                verb_pos = i
                tokens[i] = tokens[i].replace("**##", "")
                break

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, do_basic_tokenize=False)
    tokens, verb_pos = word_piece_tokenize(tokens, verb_pos, tokenizer)

    key_tok = None
    candidates = []
    dim = -1
    if dimension == "typical duration":
        key_tok = "for 1"
        candidates = [2117, 3371, 3198, 2154, 2733, 3204, 2095, 5476, 2301]
        value = value.split()[1] + "_dur"
        dim = 0
    if dimension == "typical frequency":
        key_tok = "every"
        candidates = [2117, 3371, 3198, 2154, 2733, 3204, 2095, 5476, 2301]
        value = value.split()[1] + "_freq"
        dim = 1
    if dimension == "typical occurring time of the day":
        key_tok = "in the"
        candidates = [6440, 2851, 11501, 5027, 3944, 18406, 2305, 7090]
        dim = 2
    if dimension == "typical occurring day of the week":
        key_tok = "on"
        candidates = [6928, 9857, 9317, 9432, 5958, 5095, 4465]
        dim = 3
    if dimension == "typical occurring month":
        key_tok = "in"
        candidates = [2254, 2337, 2233, 2258, 2089, 2238, 2251, 2257, 2244, 2255, 2281, 2285]
        dim = 4
    if dimension == "typical occurring season":
        key_tok = "in the"
        candidates = [3500, 2621, 7114, 3467]
        dim = 5

    label = get_soft_labels(value)

    new_tokens = []
    target = -1
    for i in range(0, len(tokens)):
        new_tokens.append(tokens[i])
        if i == verb_pos:
            new_tokens.append(key_tok)
            new_tokens.append("[MASK]")
            target = len(new_tokens)

    if dimension == "typical duration" or dimension == "typical occurring time of the day" or dimension == "typical occurring season":
        target += 1

    seq = "[CLS] " + " ".join(new_tokens) + " [SEP]"

    f_out = open("samples/intrinsic/bert.dur.txt", "a+")
    candidates = [str(x) for x in candidates]

    f_out.write(seq + "\t" + str(target) + "\t" + str(label) + "\t" + " ".join(candidates) + "\t" + str(dim) + "\n")


def format_visualization(sentence, dimension, value, verb_pos=None):
    sentence = sentence.replace("<font color='red'>", "**##")
    sentence = sentence.replace("</font>", "")

    if dimension != "typical duration":
        return

    tokens = sentence.split()
    if verb_pos is None:
        verb_pos = -1
        for i, t in enumerate(tokens):
            if t.startswith("**##"):
                verb_pos = i
                tokens[i] = tokens[i].replace("**##", "")
                break

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, do_basic_tokenize=False)
    tokens, verb_pos = word_piece_tokenize(tokens, verb_pos, tokenizer)

    f_out = open("samples/intrinsic/test.formatted.txt", "a+")

    f_out.write(" ".join(tokens) + "\t" + str(verb_pos) + "\t[unused520]\t0\t" + str(get_soft_labels(value.split()[1] + "_dur")) + "\n")


def filter_lines():
    # lines = [x.strip() for x in open("/Users/xuanyuzhou/Downloads/realnewsall.csv").readlines()[0:]]
    lines = [x.strip() for x in open("samples/gigaword/to_annotate_formatted.txt").readlines()[0:]]

    sent_map = {}
    for line in lines:
        # group = line[1:-1].split('","')
        # sentence = group[27]
        # likelihood = group[35]
        # dimension = group[29]
        # label = group[30]
        # validation = group[34]
        group = line.split("\t")
        sentence = group[0]
        likelihood = "true"
        validation = "true"
        dimension = group[2]
        label = group[3]
        if likelihood == "true":
            likelihood = 1
        else:
            likelihood = 0
        if validation == "true":
            validation = 1
        else:
            validation = 0
        sentence = sentence + "****" + label
        if "forever" in label or "inst" in label:
            continue
        if sentence not in sent_map:
            sent_map[sentence] = [0, 0, "", ""]

        sent_map[sentence][0] = sent_map[sentence][0] + validation
        sent_map[sentence][1] = sent_map[sentence][1] + likelihood
        sent_map[sentence][2] = dimension
        sent_map[sentence][3] = label

    counter = 0
    for sent in sent_map:
        if sent_map[sent][1] >= 0:
            counter += 1
            if sent_map[sent][2] != "typical duration" or "day" not in sent_map[sent][3]:
                continue
            format_model_lines(sent.split("****")[0], sent_map[sent][2], sent_map[sent][3])
            format_bert_lines(sent.split("****")[0], sent_map[sent][2], sent_map[sent][3])
            # format_visualization(sent.split("****")[0], sent_map[sent][2], sent_map[sent][3])
            print(sent)
            print(sent_map[sent][3])
    print(counter)
    print(len(sent_map))


def filter_udst():
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = nlp.tokenizer.tokens_from_list
    lines = [x.strip() for x in open("samples/UD_English_finetune_filtered/test.formatted.txt").readlines()]
    label_map = {}
    for line in lines:
        sentence = line.split("\t")[0].lower()
        verb_pos = int(line.split("\t")[1])
        doc = nlp(sentence.split())
        counter = 0
        cont = False
        for tok in doc:
            if counter == verb_pos:
                if tok.pos_ != "VERB":
                    cont = True
                    break
            counter += 1
        if cont:
            continue
        label = line.split("\t")[2]
        if "forever" in label or "instantaneous" in label:
            continue
        dimension = "typical duration"
        if label not in label_map:
            label_map[label] = 0
        if label_map[label] > 50:
            continue
        label_map[label] += 1

        format_model_lines(sentence, dimension, label, verb_pos=verb_pos)
        format_bert_lines(sentence, dimension, label, verb_pos=verb_pos)

def produce_to_annotate_lines():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    training_lines = [x.strip() for x in open("samples/gigaword/lm_format_realnews.txt")]
    selection_map = {}
    for line in training_lines:
        sentence, dimension, value, orig_line = parse_line(line, tokenizer)
        key = dimension + " " + value
        if key not in selection_map:
            selection_map[key] = []

        selection_map[key].append([sentence, dimension, value, orig_line])

    f_out = open("samples/gigaword/to_annotate.txt", "w")
    for key in selection_map:
        instances = selection_map[key]
        random.shuffle(instances)
        for s, d, v, orig_line in instances:
            f_out.write("\t".join([s, d, v, orig_line]) + "\n")


import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import decomposition
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np

def visualize_distribution():
    lines = [x.strip() for x in open("score_outputs_e2.txt").readlines()]

    prob_sum = [0.0] * 9
    for line in lines:
        scores = [float(x) for x in line.split("\t")]
        scores = [math.exp(x) for x in scores]

        sum_total = 0.0
        for s in scores:
            sum_total += s

        scores = [x / sum_total for x in scores]

        for i in range(0, 9):
            prob_sum[i] += scores[i]

    prob_sum = [x / float(len(lines)) for x in prob_sum]

    prob_sum_e2 = prob_sum

    lines = [x.strip() for x in open("score_outputs.txt").readlines()]

    prob_sum = [0.0] * 9
    for line in lines:
        scores = [float(x) for x in line.split("\t")]
        scores = [math.exp(x) for x in scores]

        sum_total = 0.0
        for s in scores:
            sum_total += s

        scores = [x / sum_total for x in scores]

        for i in range(0, 9):
            prob_sum[i] += scores[i]

    prob_sum = [x / float(len(lines)) for x in prob_sum]

    y = ["seconds", "minutes", "hours", "days", "weeks", "months", "years", "decades", "centuries"]
    y_pos = np.arange(len(y))
    #
    # simulated_e2 = []
    #
    # for i in range(0, 10000):
    #     r = random.random()
    #     prev_boundary = 0.0
    #     next_boundary = 0.0
    #     for j in range(0, 9):
    #         next_boundary += prob_sum_e2[j]
    #         if i > 0:
    #             prev_boundary += prob_sum_e2[j - 1]
    #         if prev_boundary < r <= next_boundary:
    #             simulated_e2.append(j)
    #             break


    plt.plot(prob_sum_e2)
    plt.plot(prob_sum)
    # p1 = sns.distplot(simulated_e2, color="r")
    # p1 = sns.lineplot(prob_sum, y, color="b")
    plt.xticks(y_pos, y)
    print("\t".join([str(x) for x in prob_sum_e2]))
    print("\t".join([str(x) for x in prob_sum]))


# format_lines()
# filter_lines()
# filter_udst()
# produce_to_annotate_lines()
visualize_distribution()







