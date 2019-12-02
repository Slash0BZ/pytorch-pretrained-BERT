import jsonlines
import time
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import sys
import os
import argparse
import torch

torch.set_num_threads(1)

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

    def __init__(self, output_path):
        model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz',
                                'semantic-role-labeling')
        self.predictor = model.predictor()
        self.predictor._model = self.predictor._model.cuda()
        self.output_path = output_path

    def predict_batch(self, sentences):
        f_out = jsonlines.open(self.output_path, "w")
        start_time = time.time()
        batch_size = 128
        input_maps = []
        for i in range(0, len(sentences) - batch_size, batch_size):
            input_map = []
            input_size = 0
            for j in range(0, batch_size):
                input_map.append({"sentence": sentences[i+j]})
                input_size += len(sentences[i+j].split())
                if input_size > 3000:
                    continue
            input_maps.append(input_map)
        write_accumulate = 0
        finished_batch = []
        total_lines_wrote = 0
        for input_map in input_maps:
            prediction = self.predictor.predict_batch_json(input_map)
            total_lines_wrote += len(input_map)
            finished_batch.append(prediction)
            write_accumulate += 1
            if write_accumulate == 50:
                for f in finished_batch:
                    f_out.write(f)
                print("Average Time: " + str((time.time() - start_time) / (50.0 * float(batch_size))))
                print("Total Sentences Processed: " + str(total_lines_wrote))
                start_time = time.time()
                write_accumulate = 0
                finished_batch = []

    def predict_single(self, instances):
        f_out = jsonlines.open(self.output_path, "w")
        counter = 0
        start_time = time.time()
        for instance in instances:
            prediction = self.predictor.predict_tokenized(instance.split())
            f_out.write(prediction)
            counter += 1
            if counter % 10 == 0:
                print("Average Time: " + str((time.time() - start_time) / 10.0))
                start_time = time.time()

    def predict_file(self, path):
        with open(path) as f:
            lines = [x.strip() for x in f.readlines()]
        new_lines = []
        for l in lines:
            if len(l.split("\t")) < 3:
                continue
            sent = l.split("\t")[1]
            if len(sent.split()) < 35:
                new_lines.append(sent)
        new_lines = list(set(new_lines))
        self.predict_batch(new_lines)


def produce_rest_files():
    all_lines = [x.strip() for x in open("samples/gigaword/raw_collection_contextsent.txt").readlines()]
    print("loaded all lines.")

    srl_processed_set = []
    for dirName, subdirList, fileList in os.walk("samples/gigaword"):
        for fname in fileList:
            if fname.startswith("srl"):
                lines = [x.strip() for x in open("samples/gigaword/" + fname).readlines()]
                reader = jsonlines.Reader(lines)
                for obj_list in reader:
                    for obj in obj_list:
                        srl_processed_set.append("".join(obj['words']))
                print("processed " + fname)

    srl_processed_set = list(set(srl_processed_set))
    to_process = []
    for i, line in enumerate(all_lines):
        sent = line.split("\t")[1]
        key = sent.replace(" ", "")
        if key not in srl_processed_set:
            to_process.append(line)
        if i % 1000000 == 0:
            print("processed " + str(i) + " lines")
    f_out = open("samples/gigaword/raw_collection_additional_to_srl.txt", "w")
    for line in list(set(to_process)):
        f_out.write(line + "\n")


# produce_rest_files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="INPUT FILE")
    parser.add_argument("output", help="OUTPUT FILE")
    args = parser.parse_args()

    srl = AllenSRL(args.output)
    srl.predict_file(args.input)

