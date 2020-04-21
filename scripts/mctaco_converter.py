from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer

def word_piece_tokenize(tokens, tokenizer):
    ret_tokens = []
    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.tokenize(token)
        ret_tokens.extend(sub_tokens)

    return ret_tokens

def format_instance(context, question, answer, label, tokenizer):
    tokens_context = word_piece_tokenize(context.split(), tokenizer)
    tokens_question = word_piece_tokenize(question.split(), tokenizer)
    tokens_answer = word_piece_tokenize(answer.split(), tokenizer)

    label_val = 0
    if label == "no":
        label_val = 1

    tokens = ["[CLS]"] + tokens_context + ["[SEP]"] + tokens_question + ["[SEP]", "[unused510]"] + tokens_answer + \
             ["[SEP]", "[unused510]", "[unused511]", "[MASK]", "[SEP]"]

    recover_pos = len(tokens) - 2

    label_vec = [63, 64, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
    label_vals = [0.0] * 12
    label_vals[label_val] = 1.0

    its = [" ".join(tokens), str(recover_pos), " ".join([str(x) for x in label_vec]), " ".join([str(x) for x in label_vals]), " ".join([str(x) for x in [-1] * 128])]

    return "\t".join(its)

def format_direct_test(context, question, answer, label, tokenizer):
    tokens_context = word_piece_tokenize(context.split(), tokenizer)
    tokens_question = word_piece_tokenize(question.split(), tokenizer)
    tokens_answer = word_piece_tokenize(answer.split(), tokenizer)

    label_val = 0
    if label == "no":
        label_val = 1

    tokens = ["[CLS]"] + tokens_context + ["[SEP]"] + tokens_question + ["[SEP]", "[unused510]"] + tokens_answer + \
             ["[SEP]", "[unused510]", "[unused511]", "[MASK]", "[SEP]"]
    recover_pos = len(tokens) - 2
    label_vec = [63, 64, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]

    its = [" ".join(tokens), str(recover_pos), str(1), " ".join([str(x) for x in label_vec]), str(0)]
    return "\t".join(its)

def output_file():
    lines = [x.strip() for x in open("../MCTACO/dataset/dev_3783.tsv", "r").readlines()]
    f_out = open("mctaco_dev.txt", "w")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    for i in range(0, 1):
        for l in lines:
            context = l.split("\t")[0]
            question = l.split("\t")[1]
            answer = l.split("\t")[2]
            label = l.split("\t")[3]

            formatted = format_instance(context, question, answer, label, tokenizer)
            # formatted = format_direct_test(context, question, answer, label, tokenizer)
            f_out.write(formatted + "\n")


output_file()
