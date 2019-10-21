import jsonlines
import json

def convert_into_temporalqa_format():
    lines = [x.strip() for x in open("samples/commonsenseqa/train_rand_split.jsonl").readlines()]
    reader = jsonlines.Reader(lines)
    f_out = open("samples/commonsenseqa/train.formatted.txt", "w")
    for obj_list in reader:
        body = obj_list['question']['stem']
        answer_map = {}
        for answer in obj_list['question']['choices']:
            answer_map[answer['label']] = answer['text']

        for key in answer_map:
            if key == obj_list['answerKey']:
                f_out.write("NONE\t" + body + "\t" + answer_map[key] + "\tyes\n")
            else:
                f_out.write("NONE\t" + body + "\t" + answer_map[key] + "\tno\n")


convert_into_temporalqa_format()
