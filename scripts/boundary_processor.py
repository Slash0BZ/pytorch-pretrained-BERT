
def transform_boundary_file(input_file, output_file):
    lines = [x.strip() for x in open(input_file).readlines()]
    f_out = open(output_file, "w")
    cur_group = []

    for line in lines:
        if line == "":
            if len(cur_group) > 0:
                group_left = cur_group[0][1]
                group_right = cur_group[-1][2]
                group_comb = "".join([x[0] for x in cur_group])
                group_comb = group_comb.replace("##", "")
                f_out.write(group_comb + "\t" + str(group_left) + "\t" + str(group_right) + "\n")
                cur_group = []
            f_out.write("\n")
            continue

        groups = line.split("\t")
        groups[1] = float(groups[1])
        groups[2] = float(groups[2])
        if groups[0].startswith("##") or len(cur_group) == 0:
            cur_group.append(groups)
        else:
            group_left = cur_group[0][1]
            group_right = cur_group[-1][2]
            group_comb = "".join([x[0] for x in cur_group])
            group_comb = group_comb.replace("##", "")
            f_out.write(group_comb + "\t" + str(group_left) + "\t" + str(group_right) + "\n")
            cur_group = []
            cur_group.append(groups)

def check_for_missing():
    have_lines = [x.strip() for x in open("samples/boundary_results/conll_boundary.txt").readlines()]
    have_sents = set()
    cur_sent = ""
    for hl in have_lines:
        if hl == "":
            if len(cur_sent) > 0:
                have_sents.add(cur_sent)
                cur_sent = ""
        else:
            cur_sent += hl.split("\t")[0].lower()

    gold_lines = [x.strip() for x in open("/Users/xuanyuzhou/Downloads/boundary_CLM_CoNLL_dev.out").readlines()]
    all_sents = set()
    for l in gold_lines:
        ll = l.replace("{{", "")
        ll = ll.replace("}}", "")
        ll = ll.replace("[[", "")
        ll = ll.replace("]]", "")
        ll = ll.replace(" ", "")
        if ll.lower() not in have_sents:
            print(l)


def fill_missing():
    have_lines = [x.strip() for x in open("samples/boundary_results/conll_boundary.txt").readlines()]
    cur_sent = []
    all_sents = []
    for hl in have_lines:
        if hl == "":
            if len(cur_sent) > 0:
                all_sents.append(cur_sent)
                cur_sent = []
        else:
            cur_sent.append(hl)
    if len(cur_sent) > 0:
        all_sents.append(cur_sent)

    gold_lines = [x.strip() for x in open("/Users/xuanyuzhou/Downloads/CLM_Conll_dev.out").readlines()]
    all_gold_sents = []
    cur_sent = []
    for hl in gold_lines:
        if hl == "":
            if len(cur_sent) > 0:
                all_gold_sents.append(cur_sent)
                cur_sent = []
        else:
            cur_sent.append(hl)
    if len(cur_sent) > 0:
        all_gold_sents.append(cur_sent)

    insert_map = {}
    additions = 0
    for i in range(0, len(all_sents)):
        cur_raw_sent = " ".join([x.split("\t")[0] for x in all_sents[i]])
        cur_gold_sent = " ".join([x.split("\t")[0] for x in all_gold_sents[i+additions]])
        if cur_gold_sent != cur_raw_sent:
            insert_map[i] = [x.split("\t")[0] + "\t" + str(0.0) + "\t" + str(0.0) for x in all_gold_sents[i+additions]]
            additions += 1

    f_out = open("samples/boundary_results/conll_boundary_fixed.txt", "w")
    for i in range(0, len(all_sents)):
        if i in insert_map:
            for l in insert_map[i]:
                print(l)
                f_out.write(l + "\n")
            f_out.write("\n")
        for l in all_sents[i]:
            f_out.write(l + "\n")
        f_out.write("\n")


transform_boundary_file("samples/boundary_results/conll.raw.txt", "samples/boundary_results/conll_boundary.txt")
transform_boundary_file("samples/boundary_results/ontonotes.raw.txt", "samples/boundary_results/ontonotes_boundary.txt")
# check_for_missing()
fill_missing()
