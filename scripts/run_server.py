import torch
import os
import numpy as np
from scipy.stats import norm
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForTemporalClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer


class Annotator:

    def __init__(self, bert_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))

        self.model = BertForTemporalClassification.from_pretrained(bert_model,
                                                                   cache_dir=cache_dir,
                                                                   num_labels=1).to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.max_seq_length = 128

    def get_cdf(self, upper, lower, mu, sigma, pi):
        mu = mu.detach().cpu().numpy()
        sigma = sigma.detach().cpu().numpy()
        pi = pi.detach().cpu().numpy()
        # For log models
        mu = np.exp(mu) / 290304000.0
        sigma = np.exp(sigma) / 290304000.0
        # end
        accu = 0.0
        for i, _ in enumerate(mu):
            n = norm(mu[i], sigma[i])
            accu += pi[i] * (n.cdf(upper) - n.cdf(lower))
        return accu

    def get_mean(self, mu, pi):
        mu = mu.detach().cpu().numpy()
        pi = pi.detach().cpu().numpy()
        acc = 0.0
        for i, _ in enumerate(mu):
            acc += pi[i] * mu[i]
        return acc

    def get_segment_boundaries(self):
        convert_map = [
            1.0,
            60.0 * 0.5,
            60.0 * 60.0 * 0.5,
            24.0 * 60.0 * 60.0 * 0.5,
            7.0 * 24.0 * 60.0 * 60.0 * 0.5,
            30.0 * 24.0 * 60.0 * 60.0 * 0.5,
            365.0 * 24.0 * 60.0 * 60.0 * 0.5,
            100.0 * 365.0 * 24.0 * 60.0 * 60.0 * 0.5,
            290304000.0,
        ]
        return convert_map

    def annotate(self, sentence, verb_pos, arg0_start, arg0_end, arg1_start, arg1_end, arg2_start, arg2_end):
        tokens = sentence.lower().split()
        for i, _ in enumerate(tokens):
            if tokens[i].lower() not in self.tokenizer.vocab:
                tokens[i] = "[UNK]"

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        arg0_start += 1
        arg0_end += 1
        arg1_start += 1
        arg1_end += 1
        segment_ids = [0] * len(tokens)
        subj_mask = [0] * len(tokens)
        obj_mask = [0] * len(tokens)
        arg2_mask = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        for i in range(arg0_start, arg0_end):
            subj_mask[i] = 1
        for i in range(arg1_start, arg1_end):
            obj_mask[i] = 1
            # subj_mask[i] = 1
        for i in range(arg2_start, arg2_end):
            arg2_mask[i] = 1
            # subj_mask[i] = 1
        subj_mask[verb_pos] = 1

        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        subj_mask += padding
        obj_mask += padding
        arg2_mask += padding

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
        arg2_masks = torch.tensor([arg2_mask], dtype=torch.long).to(self.device)

        with torch.no_grad():
            pi, mu, sigma = self.model(
                input_ids, segment_ids, input_mask, labels=None, target_idx=target_idx, subj_mask=subj_masks,
                obj_mask=obj_masks, arg3_mask=arg2_masks
            )
        print(pi)
        print(mu)
        divident = 290304000.0
        print(tokens)
        print("Mean: " + str(self.get_mean(mu[0], pi[0])))

        ret_cdfs = []
        boundaries = self.get_segment_boundaries()
        for i in range(0, len(boundaries) - 1):
            ret_cdfs.append(self.get_cdf(boundaries[i], boundaries[i + 1], mu[0], sigma[0], pi[0]))

        return ret_cdfs


if __name__ == "__main__":
    annotator = Annotator("models/bert-base-temporalverb")
    result = annotator.annotate("I went to China .", 1, 0, 1, 3, 4)
    print(result)