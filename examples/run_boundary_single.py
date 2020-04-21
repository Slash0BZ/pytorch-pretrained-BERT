from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from torch.nn import CrossEntropyLoss
import torch
import os
import math


class BertBoundary(BertPreTrainedModel):
    def __init__(self, config, num_labels=4):
        super(BertBoundary, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.apply(self.init_bert_weights)
        self.classifier = nn.Linear(config.hidden_size, 4)

        self.lm_loss_fn = CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, token_type_ids, attention_mask, lm_labels=None):
        seq_output, target_all_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(seq_output)
        logits = self.classifier(pooled_output)

        if lm_labels is None:
            return logits
        else:
            loss = self.lm_loss_fn(logits.view(-1, 4), lm_labels.view(-1))
            return loss


class BertBoundaryRunner:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))
        self.bert_model = "models/bert_boundary"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=False)
        self.model = BertBoundary.from_pretrained(self.bert_model, cache_dir=cache_dir).to(self.device)
        self.model.eval()

        self.max_seq_length = 128

    def softmax(self, a):
        s = 0.0
        aa = [math.exp(x) for x in a]
        for aaa in aa:
            s += aaa
        ret = [x / s for x in aa]
        return ret

    def run_tokens(self, tokens):

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

        with torch.no_grad():
            logits = self.model(
                input_ids, segment_ids, input_mask, None
            )
        logits = logits.view(1, 128, 4).cpu().numpy()[0]

        ls_current = []
        rs_current = []
        for token_logits in logits:
            softmaxed = self.softmax(token_logits)
            ls_current.append(softmaxed[1] + softmaxed[3])
            rs_current.append(softmaxed[2] + softmaxed[3])

        return ls_current[1:-1][:len(tokens)], rs_current[1:-1][:len(tokens)]

runner = BertBoundaryRunner()
ls, rs = runner.run_tokens("Chris went to New Zealand today , and she went to the supermarket .".split())
