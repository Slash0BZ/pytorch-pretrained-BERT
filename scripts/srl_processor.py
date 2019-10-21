import jsonlines
import time
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import sys
import argparse


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
        counter = 0
        start_time = time.time()
        batch_size = 256
        for i in range(0, len(sentences), batch_size):
            input_map = []
            for j in range(0, batch_size):
                input_map.append({"sentence": sentences[i+j]})
            prediction = self.predictor.predict_batch_json(input_map)
            f_out.write(prediction)
            counter += 1
            if counter % 10 == 0:
                print("Average Time: " + str((time.time() - start_time) / (10.0 * float(batch_size))))
                start_time = time.time()

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
        self.predict_batch(new_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="INPUT FILE")
    parser.add_argument("output", help="OUTPUT FILE")
    args = parser.parse_args()

    srl = AllenSRL(args.output)
    srl.predict_file(args.input)

