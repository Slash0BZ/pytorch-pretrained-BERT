import torch
from torch import nn
import os
import random
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertLayer, BertOnlyMLMHead
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from torch.nn import CrossEntropyLoss
import numpy as np

class TemporalModelJoint(BertPreTrainedModel):
    def __init__(self, config, num_labels, num_typical_labels):
        super(TemporalModelJoint, self).__init__(config)
        self.num_labels = num_labels
        self.num_typical_labels = num_typical_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.log_softmax = nn.LogSoftmax(-1)
        self.softmax = nn.Softmax(-1)

        self.dur_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.freq_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.typical_classifier = nn.Linear(config.hidden_size, self.num_typical_labels)

        self.concat_size = self.num_labels * 2 + self.num_typical_labels

        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

        self.lm_loss_fn = CrossEntropyLoss(ignore_index=-1)

    def get_single_inference(self, input_ids, token_type_ids, attention_mask, target_ids):
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        target_all_output = seq_output.gather(1, target_ids.view(-1, 1).unsqueeze(2).repeat(1, 1, seq_output.size(2)))

        pooled_output = self.dropout(target_all_output)

        dur_logits = self.softmax(self.dur_classifier(pooled_output)).view(-1, self.num_labels)
        freq_logits = self.softmax(self.freq_classifier(pooled_output)).view(-1, self.num_labels)
        typical_logits = self.softmax(self.typical_classifier(pooled_output)).view(-1, self.num_typical_labels)

        cls_logits = self.cls(seq_output)

        return freq_logits, dur_logits, typical_logits, cls_logits

    def compute_lm_loss(self, cls_output, labels):
        if labels is None:
            return None
        lm_loss = self.lm_loss_fn(cls_output.view(-1, 30522), labels.view(-1))
        return lm_loss

    def forward(self, input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a):
        freq_a, dur_a, typ_a, cls_a = self.get_single_inference(input_ids_a, token_type_ids_a, attention_mask_a, target_ids_a)

        return freq_a, dur_a, typ_a


class Annotator:

    def __init__(self, bert_model, num_labels):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))

        self.model = TemporalModelJoint.from_pretrained(bert_model,
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

        print(tokens[verb_pos + 1])

        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.device)
        target_idx = torch.tensor([verb_pos + 1], dtype=torch.long).to(self.device)

        with torch.no_grad():
            freq_logits, dur_logits, typ_logits = self.model(
                input_ids, segment_ids, input_mask, target_idx
            )

        return freq_logits.detach().cpu().numpy(), dur_logits.detach().cpu().numpy(), typ_logits.detach().cpu().numpy()


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


def process_random_input():
    annotator_joint = Annotator("../event-duration-visualization/bert_joint_epoch_1", 31)
    srl = AllenSRL()

    lines = [x.strip() for x in open("samples/UD_English_finetune/test.formatted.txt").readlines()]
    random.shuffle(lines)
    lines = lines[:200]

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
    for line in lines:
        text = line.split("\t")[0]
        pred = int(line.split("\t")[1])
        srl_result = srl.predict_single(text)
        tokens = []
        for verb in srl_result['verbs']:
            tags = verb['tags']
            if tags[int(pred)].endswith("-V"):
                tokens, verb_pos = srl.get_stripped(srl_result["words"], tags, int(pred))

        freq, dur, typ = annotator_joint.annotate(" ".join(tokens), verb_pos)
        freq = int(np.argmax(freq[0]))
        dur = int(np.argmax(dur[0]))
        typ_day = int(np.argmax(typ[0][0:8]))
        typ_week = int(np.argmax(typ[0][8:15]))
        typ_year = int(np.argmax(typ[0][15:27]))
        typ_season = int(np.argmax(typ[0][27:31]))
        orig_tokens = text.split()
        orig_tokens[pred] = "[" + orig_tokens[pred] + "]"
        out_content = [" ".join(orig_tokens), labels_unit[freq], labels_unit[dur],
                       labels_typical[0:8][typ_day], labels_typical[8:15][typ_week], labels_typical[15:27][typ_year], labels_typical[27:31][typ_season]]
        f_out.write("\t".join(out_content) + "\n")


process_random_input()
