from ccg_nlpy import local_pipeline
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

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

class PretrainedModel:
    """
    A pretrained model is determined by both an archive file
    (representing the trained model)
    and a choice of predictor.
    """
    def __init__(self, archive_file: str, predictor_name: str) -> None:
        self.archive_file = archive_file
        self.predictor_name = predictor_name

    def predictor(self) -> Predictor:
        archive = load_archive(self.archive_file)
        return Predictor.from_archive(archive, self.predictor_name)


class AllenSRL:

    def __init__(self):
        model = PretrainedModel('../event-duration-visualization/srl-model-2018.05.25.tar.gz',
                                'semantic-role-labeling')
        self.predictor = model.predictor()

    def predict_batch(self, sentences):
        for sentence in sentences:
            prediction = self.predictor.predict(sentence)
            print(prediction)

    def predict_single(self, sentence):
        return self.predictor.predict_tokenized(sentence.split())

    def predict_file(self, path):
        with open(path) as f:
            lines = [x.strip() for x in f.readlines()]
        self.predict_batch(lines)

    def get_stripped(self, tokens, tags, orig_verb_pos):
        new_tokens = []
        new_verb_pos = -1
        for i in range(0, len(tokens)):
            if tags[i] != "O":
                new_tokens.append(tokens[i])
            if i == orig_verb_pos:
                new_verb_pos = len(new_tokens) - 1
        return new_tokens, new_verb_pos

    def get_verb_form(self, tokens, tags):
        for i, t in enumerate(tags):
            if t.endswith("-V"):
                return i, tokens[i]
        return -1, "NONE"


def convert_mctaco_data(path, out_path):
    lines = [x.strip() for x in open(path).readlines()]
    pipeline = local_pipeline.LocalPipeline()
    f_out = open(out_path, "w")
    srl = AllenSRL()
    for line in lines:
        sent = line.split("\t")[1]
        doc = pipeline.doc(sent)
        list_of_verbs = []
        tokenized_words = []
        for i, token_group in enumerate(list(doc.get_pos)):
            tokenized_words.append(token_group['tokens'])
            if token_group['label'].startswith("VB"):
                list_of_verbs.append(token_group['tokens'])
        selected_verb = ""
        if len(list_of_verbs) == 1:
            selected_verb = list_of_verbs[0]
        elif len(list_of_verbs) > 1:
            if list_of_verbs[0].lower() in ['did', 'does', 'do', 'had', 'have', 'has', 'was', 'were', 'is', 'happened',
                                            'happens', 'happen', 'be', 'is', 'are']:
                selected_verb = list_of_verbs[1]
            else:
                selected_verb = list_of_verbs[0]
        srl_result = srl.predict_single(" ".join(tokenized_words))
        tokens = ["NONE"]
        verb_pos = -1
        for verb in srl_result['verbs']:
            tags = verb['tags']
            tokens_raw = srl_result['words']
            if srl.get_verb_form(tokens_raw, tags)[1].lower() == selected_verb:
                tokens, verb_pos = srl.get_stripped(srl_result["words"], tags, srl.get_verb_form(tokens_raw, tags)[0])
        f_out.write(line + "\t" + " ".join(tokens) + "\t" + str(verb_pos) + "\n")




# convert_mctaco_data("samples/split_30_70_good/dev.txt", "samples/split_30_70_good_verb_srl/dev.txt")
# convert_mctaco_data("samples/split_30_70_good/test.txt", "samples/split_30_70_good_verb_srl/test.txt")

