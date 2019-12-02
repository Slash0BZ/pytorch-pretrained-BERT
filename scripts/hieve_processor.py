import os
import random


def splitter():

    for i in range(1, 6):
        train_file = open("samples/hieve_processed/split_" + str(i) + "/train.formatted.txt", "w")
        test_file = open("samples/hieve_processed/split_" + str(i) + "/test.formatted.txt", "w")

        root_path = "samples/hieve_processed"
        file_names = []
        for dirName, subdirList, fileList in os.walk(root_path):
            for i, fname in enumerate(fileList):
                if not fname.endswith("tsvx"):
                    continue
                file_names.append(fname)
        random.shuffle(file_names)
        for i, fname in enumerate(file_names):
            if i < int(float(len(file_names)) * 0.8):
                f_out = train_file
            else:
                f_out = test_file
            lines = [x.strip() for x in open(root_path + "/" + fname).readlines()]
            tokens = []
            event_map = {}
            rel_map = {}
            for line in lines:
                func = line.split("\t")[0]
                if func == "Text":
                    tokens = line.split("\t")[1].split()
                elif func == "Event":
                    joint_context = []
                    target_pos = int(line.split("\t")[5])
                    ret_pos = -1
                    for i in range(max(0, int(target_pos) - 20), min(len(tokens), int(target_pos) + 20)):
                        joint_context.append(tokens[i])
                        if i == target_pos:
                            ret_pos = len(joint_context) - 1
                    if ret_pos > -1:
                        event_map[int(line.split("\t")[1])] = (" ".join(joint_context), ret_pos)
                elif func == "Relation":
                    group = line.split("\t")
                    rel_type = group[3]
                    arg_1 = int(group[1])
                    arg_2 = int(group[2])
                    if arg_1 not in rel_map:
                        rel_map[arg_1] = []
                    if arg_2 not in rel_map:
                        rel_map[arg_2] = []
                    rel_map[arg_1].append(arg_2)
                    rel_map[arg_2].append(arg_1)

                    if arg_1 not in event_map or arg_2 not in event_map:
                        continue

                    arg_str_1 = [str(x) for x in event_map[arg_1]]
                    arg_str_2 = [str(x) for x in event_map[arg_2]]

                    if rel_type == "Coref":
                        f_out.write("\t".join(arg_str_1) + "\t" + "\t".join(arg_str_2) + "\t1\t" + fname + "\t" + group[
                            1] + "\t" + group[2] + "\n")
                        f_out.write("\t".join(arg_str_2) + "\t" + "\t".join(arg_str_1) + "\t1\t" + fname + "\t" + group[
                            2] + "\t" + group[1] + "\n")
                    elif rel_type == "SuperSub":
                        f_out.write("\t".join(arg_str_1) + "\t" + "\t".join(arg_str_2) + "\t2\t" + fname + "\t" + group[
                            1] + "\t" + group[2] + "\n")
                        f_out.write("\t".join(arg_str_2) + "\t" + "\t".join(arg_str_1) + "\t3\t" + fname + "\t" + group[
                            2] + "\t" + group[1] + "\n")
            for e in event_map:
                for e_prime in event_map:
                    if e not in rel_map or e_prime not in rel_map[e]:
                        # r = random.random()
                        # if r < 0.5:
                        if e != e_prime:
                            f_out.write("\t".join([str(x) for x in event_map[e]]) + "\t" + "\t".join([str(x) for x in event_map[e_prime]]) + "\t0\t" + fname + "\t" + str(e) + "\t" + str(e_prime) + "\n")


# lines = [x.strip() for x in open("samples/hieve_processed/all.txt").readlines()]
#
# for i in range(1, 6):
#     random.shuffle(lines)
#     train_file = open("samples/hieve_processed/split_" + str(i) + "/train.formatted.txt", "w")
#     test_file = open("samples/hieve_processed/split_" + str(i) + "/test.formatted.txt", "w")
#     pos = int(float(len(lines)) * 0.8)
#     for line in lines[0:pos]:
#         train_file.write(line + "\n")
#     for line in lines[pos:]:
#         test_file.write(line + "\n")


import math
import numpy as np

def evaluator(test_file, prediction_file):
    ref_lines = [x.strip() for x in open(test_file).readlines()]
    logit_lines = [x.strip() for x in open(prediction_file).readlines()]

    prediction_map = {}
    test_instances = []
    for i, line in enumerate(ref_lines):
        fname = line.split("\t")[-3]
        key_1 = int(line.split("\t")[-2])
        key_2 = int(line.split("\t")[-1])
        if key_1 == key_2:
            continue
        logits = [float(x) for x in logit_lines[i].split()]
        s = 0.0
        for num in logits:
            s += math.exp(num)
        logits = [math.exp(x) / s for x in logits]

        comb_key_1 = fname + " " + str(key_1) + " " + str(key_2)
        comb_key_2 = fname + " " + str(key_2) + " " + str(key_1)

        if comb_key_1 not in prediction_map:
            prediction_map[comb_key_1] = [0.0] * 4
        if comb_key_2 not in prediction_map:
            prediction_map[comb_key_2] = [0.0] * 4

        prediction_map[comb_key_1][0] += logits[0]
        prediction_map[comb_key_1][1] += logits[1]
        prediction_map[comb_key_1][2] += logits[2]
        prediction_map[comb_key_1][3] += logits[3]

        prediction_map[comb_key_2][0] += logits[0]
        prediction_map[comb_key_2][1] += logits[1]
        prediction_map[comb_key_2][2] += logits[3]
        prediction_map[comb_key_2][3] += logits[2]

        gold_label = int(line.split("\t")[4])
        if gold_label in [0, 1]:
            if key_1 < key_2:
                test_instances.append([comb_key_1, gold_label])
        if gold_label in [2]:
            test_instances.append([comb_key_1, 2])
        if gold_label in [3]:
            test_instances.append([comb_key_1, 3])

    correct_map = {}
    predicted_map = {}
    labeled_map = {}
    for key, cur_target in test_instances:
        target_label_id = np.argmax(np.array(prediction_map[key]))

        if cur_target == target_label_id:
            if cur_target not in correct_map:
                correct_map[cur_target] = 0.0
            correct_map[cur_target] += 1.0
        if target_label_id not in predicted_map:
            predicted_map[target_label_id] = 0.0
        predicted_map[target_label_id] += 1.0
        if cur_target not in labeled_map:
            labeled_map[cur_target] = 0.0
        labeled_map[cur_target] += 1.0

    for key in correct_map:
        print(str(key))
        print("precision: " + str(correct_map[key] / predicted_map[key]))
        print("recall: " + str(correct_map[key] / labeled_map[key]))


evaluator("samples/hieve_processed/split_2/test.formatted.txt", "bert_outputs.txt")
