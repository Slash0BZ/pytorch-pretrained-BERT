import math
import numpy as np


def plural_units(u):
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
    if u in m:
        return m[u]
    return u


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
    seconds = convert_map[plural_units(u)] * float(v_input)
    prev_unit = "seconds"
    for i, v in enumerate(convert_map):
        if seconds / convert_map[v] < 0.5:
            break
        prev_unit = v
    if prev_unit == "seconds" and seconds > 60.0:
        prev_unit = "centuries"
    new_val = int(seconds / convert_map[prev_unit])

    return prev_unit, str(new_val)


def get_unit_map(scores, label_list):
    normalize_map = {'seconds': 1, 'minutes': 1, 'hours': 1, 'days': 1, 'weeks': 1, 'months': 1, 'years': 1, 'decades': 1, "centuries": 1}
    unit_prob_map = {}
    for i, key in enumerate(label_list):
        unit_prob_map[key] = scores[i]
    ksum = 0.0
    for key in unit_prob_map:
        ksum += math.exp(float(unit_prob_map[key])) / normalize_map[key]
    for key in unit_prob_map:
        unit_prob_map[key] = math.exp(float(unit_prob_map[key])) / normalize_map[key] / ksum
    unit_map = sorted(unit_prob_map.items(), key=lambda x: x[1], reverse=True)
    return unit_map


def eval_joint_pair_data(gold_path, predict_logits):
    duration_keys_ordered = [
        "seconds",
        "minutes",
        "hours",
        "days",
        "weeks",
        "months",
        "years",
        "decades",
        "centuries"
    ]
    typical_keys_ordered = [
        "morning",
        "afternoon",
        "night",
        "evening",
        "noon",
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        "spring", "summer", "autumn", "winter",
    ]
    distance_map = {
        "seconds": 0,
        "minutes": 1,
        "hours": 2,
        "days": 3,
        "weeks": 4,
        "months": 5,
        "years": 6,
        "decades": 7,
        "centuries": 8
    }
    gold_lines = [x.strip() for x in open(gold_path).readlines()]
    filtered_gold_lines = []
    for line in gold_lines:
        groups = line.split("\t")
        if groups[-1] != "DUR":
            pass
            # continue
        if groups[2] == "NONE" or groups[-1] not in ["FREQ", "DUR"]:
            filtered_gold_lines.append(line)
            continue
        if len(groups[0].split()) > 120 or len(groups[3].split()) > 120:
            continue
        label_a = groups[2].lower()
        label_a_num = float(label_a.split()[0])
        label_a, _ = normalize_timex(label_a_num, label_a.split()[1].lower())
        label_b = groups[5].lower()
        label_b_num = float(label_b.split()[0])
        label_b, _ = normalize_timex(label_b_num, label_b.split()[1].lower())
        # IMPORTANT!!
        skip_list = ["instantaneous", "forever"]
        if label_a in skip_list or label_b in skip_list:
            continue
        filtered_gold_lines.append(line)
    gold_lines = filtered_gold_lines

    logits = [x.strip() for x in open(predict_logits).readlines()]
    assert(len(gold_lines) == len(logits))

    pair_correct = 0.0
    pair_total = 0.0

    classification_total = {}
    classification_distance = {}
    count_map = {}
    typical_correct = 0
    typical_total = 0
    for i in range(0, len(gold_lines)):
        groups = gold_lines[i].split("\t")
        label_a = groups[2].lower()
        label_b = groups[5].lower()
        inst_type = groups[-1]
        if inst_type not in classification_total:
            classification_total[inst_type] = 0.0
            classification_distance[inst_type] = 0.0

        scores = [float(x) for x in logits[i].split("\t")]
        if label_a == "none":
            pair_total += 1.0
            label_comparison = groups[-1]
            prediction = "LESS"
            if scores[-1] > scores[-2]:
                prediction = "MORE"
            if label_comparison == prediction:
                pair_correct += 1.0
        else:
            if inst_type in ["FREQ", "DUR"]:
                val_a = float(label_a.split()[0])
                unit_a = label_a.split()[1]
                label_a, _ = normalize_timex(val_a, unit_a)
                val_b = float(label_b.split()[0])
                unit_b = label_b.split()[1]
                label_b, _ = normalize_timex(val_b, unit_b)
                if label_a not in count_map:
                    count_map[label_a] = 0
                if label_b not in count_map:
                    count_map[label_b] = 0
                count_map[label_a] += 1
                count_map[label_b] += 1
            if inst_type == "FREQ":
                prediction_map_a = get_unit_map(scores[0:9], duration_keys_ordered)
                prediction_map_b = get_unit_map(scores[39:48], duration_keys_ordered)
            if inst_type == "DUR":
                prediction_map_a = get_unit_map(scores[9:18], duration_keys_ordered)
                prediction_map_b = get_unit_map(scores[48:57], duration_keys_ordered)
            if inst_type == "TYPICAL":
                prediction_a = typical_keys_ordered[np.argmax(np.array(scores[18:39]))]
                prediction_b = typical_keys_ordered[np.argmax(np.array(scores[57:78]))]
                if prediction_a == label_a:
                    typical_correct += 1
                if prediction_b == label_b:
                    typical_correct += 1
                typical_total += 2
                continue
            prediction_a = "ERROR"
            prediction_b = "ERROR"
            for _, (p, _) in enumerate(prediction_map_a):
                prediction_a = p
                break
            for _, (p, _) in enumerate(prediction_map_b):
                prediction_b = p
                break

            cur_dist = float(abs(distance_map[label_a] - distance_map[prediction_a]))
            classification_distance[inst_type] += cur_dist
            classification_total[inst_type] += 1.0

            classification_distance[inst_type] += float(abs(distance_map[label_b] - distance_map[prediction_b]))
            classification_total[inst_type] += 1.0
    for key in classification_distance:
        if key in ["FREQ", "DUR"]:
            print(key)
            print(str(classification_distance[key] / classification_total[key]))
    print(pair_correct / pair_total)
    print("TYPICAL")
    print(str(typical_correct / typical_total))


# eval_joint_pair_data("samples/pretrain_combine/test.formatted.txt", "bert_joint_36k_tmp/bert_logits.txt")
eval_joint_pair_data("samples/conceptnet/test.formatted.txt", "bert_logits.txt")
