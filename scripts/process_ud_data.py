from ccg_nlpy import local_pipeline

def parse_instance(group):
    key = group[0] + "\t" + group[1]
    label = group[2]
    return key, label


def get_average_label(labels):
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
    reverse_list = [
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
    val_sum = 0.0
    total = 0.0
    for l in labels:
        val_sum += distance_map[l.split()[1]]
        total += 1.0
    closest_val = round(val_sum / total)
    return "1 " + reverse_list[closest_val]


def convert_test_labels(path, out_path):
    lines = [x.strip() for x in open(path).readlines()]
    instance_map = {}
    for line in lines:
        groups = line.split("\t")
        key, label = parse_instance(groups[:3])
        if key not in instance_map:
            instance_map[key] = []
        instance_map[key].append(label)
        key, label = parse_instance(groups[3:6])
        if key not in instance_map:
            instance_map[key] = []
        instance_map[key].append(label)

    f_out = open(out_path, "w")
    new_instances = []
    for key in instance_map:
        new_label = get_average_label(instance_map[key])
        new_instances.append(key + "\t" + new_label)

    for i in range(0, len(new_instances) - 1, 2):
        f_out.write(new_instances[i] + "\t" + new_instances[i + 1] + "\tNONE\n")


# convert_test_labels("samples/UD_English_SRL_9label/test.formatted.txt", "samples/UD_English_SRL_9label_avg/test.formatted.txt")

def convert_mctaco_data(path, out_path):
    lines = [x.strip() for x in open(path).readlines()]
    pipeline = local_pipeline.LocalPipeline()
    f_out = open(out_path, "w")
    for line in lines:
        sent = line.split("\t")[1]
        doc = pipeline.doc(sent)
        list_of_verbs = []
        for i, token_group in enumerate(list(doc.get_pos)):
            if token_group['label'].startswith("VB"):
                list_of_verbs.append(token_group['tokens'])
        if len(list_of_verbs) == 1:
            f_out.write(line + "\t" + list_of_verbs[0] + "\n")
        elif len(list_of_verbs) > 1:
            if list_of_verbs[0].lower() in ['did', 'does', 'do', 'had', 'have', 'has', 'was', 'were', 'is', 'happened',
                                            'happens', 'happen', 'be', 'is', 'are']:
                f_out.write(line + "\t" + list_of_verbs[1] + "\n")
            else:
                f_out.write(line + "\t" + list_of_verbs[0] + "\n")
        else:
            f_out.write(line + "\tNONE\n")


convert_mctaco_data("samples/split_30_70_good/dev.txt", "samples/split_30_70_good_verb/dev.txt")
convert_mctaco_data("samples/split_30_70_good/test.txt", "samples/split_30_70_good_verb/test.txt")

