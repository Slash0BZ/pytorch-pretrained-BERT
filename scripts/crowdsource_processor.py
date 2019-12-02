from pytorch_pretrained_bert import BertTokenizer
import random

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
    5: [41, 42],
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
    training_lines = [x.strip() for x in open("samples/gigaword/tmp_seq_allmask_fixverb/train.formatted.txt")]

    # seen_set = set()
    # for line in training_lines:
    #     sentence, _, _, _ = parse_line(line, tokenizer)
    #     seen_set.add(sentence)
    #
    # test_lines = [x.strip() for x in open("samples/gigaword/lm_format_allmask_full.txt").readlines()]

    seen_count = 0
    selection_map = {}
    for line in test_lines:
        sentence, dimension, value, orig_line = parse_line(line, tokenizer)
        if sentence in seen_set:
            seen_count += 1
            continue
        key = dimension + " " + value
        if key not in selection_map:
            selection_map[key] = []

        selection_map[key].append([sentence, dimension, value, orig_line])
    print(seen_count)

    f_out = open("samples/gigaword/to_annotate.txt", "w")
    for key in selection_map:
        instances = selection_map[key]
        random.shuffle(instances)
        for s, d, v, orig_line in instances[:500]:
            f_out.write("\t".join([s, d, v, orig_line]) + "\n")


def format_lines():
    lines = [x.strip() for x in open("samples/gigaword/to_annotate.txt").readlines()]
    f_out = open("samples/gigaword/to_annotate_formatted.txt", "w")
    count_map = {}
    total_inst_per_group = 100
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

        f_out.write("\t".join([" ".join(tokens).strip(), verb_surface, dimension, label, other_label_str]) + "\n")


def filter_lines():
    lines = [x.strip() for x in open("samples/gigaword/pilot_c.csv").readlines()[1:]]

    sent_map = {}
    for line in lines:
        group = line[1:-1].split('","')
        sentence = group[27]
        likelihood = group[32]
        if likelihood == "true":
            likelihood = True
        else:
            likelihood = False
        validation = group[34]
        if validation == "true":
            validation = True
        else:
            validation = False
        label = group[30]
        if sentence not in sent_map:
            sent_map[sentence] = [True, True, ""]

        sent_map[sentence][0] = sent_map[sentence][0] and validation
        sent_map[sentence][1] = sent_map[sentence][1] and likelihood
        sent_map[sentence][2] = label

    counter = 0
    for sent in sent_map:
        if sent_map[sent][0] and sent_map[sent][1]:
            counter += 1
            print(sent)
            print(sent_map[sent][2])
    print(counter)


filter_lines()










