import math

class WikiEval:

    def __init__(self, gold_file):
        f = open(gold_file, "r")
        self.gold_numbers = [float(x.strip()) for x in f.readlines()][:13000]
        self.eval_size = len(self.gold_numbers)

    def eval_file(self, file):
        f = open(file, "r")
        predictions = [float(x.strip()) for x in f.readlines()]
        for idx, i in enumerate(predictions):
            if math.isinf(i) or math.isnan(i):
                predictions[idx] = 0.0
        assert len(predictions) >= self.eval_size

        mape = self.calculate_mape(predictions)
        strict_10 = self.calculate_strict(predictions)
        strict_soft = self.calculate_strict_soft(predictions)

        print("=========Evaluation result for " + str(file) + "============")
        print("MAPE: " + str(mape))
        print("Strict Acc.: " + str(strict_10))
        print("Strict Soft: " + str(strict_soft))

    def calculate_mape(self, predictions):
        total = 0.0
        for idx, gold_num in enumerate(self.gold_numbers):
            if gold_num == 0.0:
                total += 0.0
            else:
                total += abs(gold_num - predictions[idx]) / abs(gold_num)

        return total / float(self.eval_size)

    # Count as correct if within 10% difference
    def calculate_strict(self, predictions):
        correct = 0.0
        for idx, gold_num in enumerate(self.gold_numbers):
            if gold_num != 0.0 and abs(gold_num - predictions[idx]) / abs(gold_num) <= 0.2:
                correct += 1.0

        return correct / float(self.eval_size)

    def calculate_strict_soft(self, predictions):
        correct = 0.0
        for idx, gold_num in enumerate(self.gold_numbers):
            if gold_num != 0.0:
                ratio = abs(gold_num - predictions[idx]) / abs(gold_num)
                ratio = 1.0 - ratio
                if ratio < 0.0:
                    ratio = 0.0
                correct += ratio
        return correct / float(self.eval_size)



evaluator = WikiEval("samples/wiki-25/eval-answer.txt")
evaluator.eval_file("samples/wiki-25/prediction/bert.16k.scratch.txt")
evaluator.eval_file("samples/wiki-25/prediction/bert.24k.scratch.txt")
evaluator.eval_file("samples/wiki-25/prediction/bert.29k.scratch.txt")
evaluator.eval_file("samples/wiki-25/prediction/bert.vanilla.29k.scratch.txt")
evaluator.eval_file("samples/wiki-25/prediction/bert.pretrained.txt")
evaluator.eval_file("samples/wiki-25/prediction/bert.10k.txt")
evaluator.eval_file("samples/wiki-25/prediction/bert.20k.txt")
evaluator.eval_file("samples/wiki-25/prediction/bert.27k.txt")
evaluator.eval_file("samples/wiki-25/prediction/median.txt")
