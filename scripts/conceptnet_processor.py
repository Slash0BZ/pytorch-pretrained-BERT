from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer
import numpy as np


def word_piece_tokenize(tokens, verb_pos, tokenizer):
    if verb_pos < 0:
        return None, -1

    ret_tokens = []
    ret_verb_pos = -1
    for i, token in enumerate(tokens):
        token = token.lower()
        if i == verb_pos:
            ret_verb_pos = len(ret_tokens)
        sub_tokens = tokenizer.tokenize(token)
        ret_tokens.extend(sub_tokens)

    return ret_tokens, ret_verb_pos


def process_conceptnet(in_file, out_file):
    lines = [x.strip() for x in open(in_file).readlines()]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, do_basic_tokenize=False)
    f_out = open(out_file, "w")
    for line in lines:
        groups = line.split("\t")
        label = 0
        if groups[-1] == "MORE":
            label = 1
        tokens_a = groups[0].split()
        verb_a = int(groups[1])
        tokens_a, verb_a = word_piece_tokenize(tokens_a, verb_a, tokenizer)
        tokens_b = groups[3].split()
        verb_b = int(groups[4])
        tokens_b, verb_b = word_piece_tokenize(tokens_b, verb_b, tokenizer)
        f_out.write(" ".join(tokens_a) + "\t" + str(verb_a) + "\t" + " ".join(tokens_b) + "\t" + str(verb_b) + "\t" + str(label) + "\n")


# process_conceptnet("samples/conceptnet_2k/train.formatted.txt", "samples/conceptnet_seq/train.formatted.txt")
# process_conceptnet("samples/conceptnet_2k/test.formatted.txt", "samples/conceptnet_seq/test.formatted.txt")

import math
def evaluate(gold_file, prediction_file):
    ref_lines = [x.strip() for x in open(gold_file).readlines()]
    logit_lines = [x.strip() for x in open(prediction_file).readlines()]

    s_predicted = 0
    l_predicted = 0
    s_labeled = 0
    l_labeled = 0
    s_correct = 0
    l_correct = 0
    total = 0
    correct = 0
    for i, line in enumerate(ref_lines):
        logits = [float(x) for x in logit_lines[i].split()]
        s = 0.0
        for num in logits:
            s += math.exp(num)
        logits = [math.exp(x) / s for x in logits]
        predicted_label = int(np.argmax(np.array(logits)))
        actual_label = int(line.split("\t")[4])
        total += 1

        if predicted_label == 0:
            s_predicted += 1
        else:
            l_predicted += 1

        if actual_label == 0:
            s_labeled += 1
        else:
            l_labeled += 1
        if actual_label == predicted_label:
            correct += 1
            if actual_label == 0:
                s_correct += 1
            else:
                l_correct += 1

    s_predicted = float(s_predicted)
    l_predicted = float(l_predicted)
    s_labeled = float(s_labeled)
    l_labeled = float(l_labeled)
    s_correct = float(s_correct)
    l_correct = float(l_correct)
    correct = float(correct)
    total = float(total)

    print("Acc.: " + str(float(correct) / float(total)))
    p = s_correct / s_predicted
    r = s_correct / s_labeled
    f = 2 * p * r / (p + r)
    print("Less than a day: " + str(f))
    p = l_correct / l_predicted
    r = l_correct / l_labeled
    f = 2 * p * r / (p + r)
    print("Longer than a day: " + str(f))


evaluate("samples/conceptnet_seq/test.formatted.txt", "bert_outputs.txt")
