import os
import random

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
                    f_out.write("\t".join(arg_str_1) + "\t" + "\t".join(arg_str_2) + "\t1\n")
                    f_out.write("\t".join(arg_str_2) + "\t" + "\t".join(arg_str_1) + "\t1\n")
                elif rel_type == "SuperSub":
                    f_out.write("\t".join(arg_str_1) + "\t" + "\t".join(arg_str_2) + "\t2\n")
                    f_out.write("\t".join(arg_str_2) + "\t" + "\t".join(arg_str_1) + "\t3\n")
        for e in event_map:
            for e_prime in event_map:
                if e not in rel_map or e_prime not in rel_map[e]:
                    r = random.random()
                    if r < 0.5:
                        f_out.write("\t".join([str(x) for x in event_map[e]]) + "\t" + "\t".join([str(x) for x in event_map[e_prime]]) + "\t0\n")


# lines = [x.strip() for x in open("samples/hieve_processed/all.txt").readlines()]

# for i in range(1, 6):
#     random.shuffle(lines)
#     train_file = open("samples/hieve_processed/split_" + str(i) + "/train.formatted.txt", "w")
#     test_file = open("samples/hieve_processed/split_" + str(i) + "/test.formatted.txt", "w")
#     pos = int(float(len(lines)) * 0.8)
#     for line in lines[0:pos]:
#         train_file.write(line + "\n")
#     for line in lines[pos:]:
#         test_file.write(line + "\n")




