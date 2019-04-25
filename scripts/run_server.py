import torch
import os
import numpy as np
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForTemporalClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer


class Annotator:

    def __init__(self, bert_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))

        self.model = BertForTemporalClassification.from_pretrained(bert_model,
                                                          cache_dir=cache_dir,
                                                          num_labels=1).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.max_seq_length = 128

    def annotate(self, sentence, verb_pos, arg0_start, arg0_end, arg1_start, arg1_end):
        tokens = sentence.lower().split()

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        arg0_start += 1
        arg0_end += 1
        arg1_start += 1
        arg1_end += 1
        segment_ids = [0] * len(tokens)
        subj_mask = [0] * len(tokens)
        obj_mask = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        for i in range(arg0_start, arg0_end):
            subj_mask[i] = 1
        for i in range(arg1_start, arg1_end):
            obj_mask[i] = 1

        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        subj_mask += padding
        obj_mask += padding

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(subj_mask) == self.max_seq_length
        assert len(obj_mask) == self.max_seq_length

        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.device)
        target_idx = torch.tensor([verb_pos], dtype=torch.long).to(self.device)
        subj_masks = torch.tensor([subj_mask], dtype=torch.long).to(self.device)
        obj_masks = torch.tensor([obj_mask], dtype=torch.long).to(self.device)

        with torch.no_grad():
            pi, mu, sigma = self.model(
                input_ids, segment_ids, input_mask, labels=None, target_idx=target_idx, subj_mask=subj_masks, obj_mask=obj_masks
            )

        ret_list = []
        prev = None
        tmp_map = {}
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        divident = 2592000.0
        for i in range(0, 1000):
            # val = float(math.exp(float(i))) / divident
            val = float(i) * 2592.0 / divident
            cdf = torch.sum(m.cdf(val) * pi, dim=1).detach().cpu().numpy()
            cdf_copy = torch.sum(m.cdf(val) * pi, dim=1).detach().cpu().numpy()
            if prev is not None:
                cdf = np.subtract(cdf, prev)
            for j, c in enumerate(cdf):
                if j not in tmp_map:
                    tmp_map[j] = []
                tmp_map[j].append(c)
            prev = cdf_copy
        if 0 not in tmp_map:
            return ret_list

        return tmp_map[0]


if __name__ == "__main__":
    annotator = Annotator("models/bert-base-temporalverb")
    result = annotator.annotate("I went to China .", 1, 0, 1, 3, 4)
    print(result)