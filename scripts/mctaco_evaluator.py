import argparse
import json

"""
Example Usage: 
python experiments/evaluator.py eval --test_file data/test_9442.tsv --prediction_file experiments/bert/bert.norm.output.txt
"""


class McTacoEvaluator:

    def __init__(self, test_file, output_file, output):
        self.test_file = test_file
        self.output_file = output_file
        self.output = output

    def print_result(self, ref_lines=None, prediction_lines=None):
        if ref_lines is None:
            ref_lines = [x.strip() for x in open(self.test_file).readlines()]
        if prediction_lines is None:
            prediction_lines = [x.strip() for x in open(self.output_file).readlines()]

        result_map = {}
        prediction_count_map = {}
        prediction_map = {}
        gold_count_map = {}
        type_map = {}
        context_map = {}
        answer_map = {}
        q_index_map = {}
        q_content_map = {}
        for i, line in enumerate(ref_lines):
            # q_index = int(line.split("\t")[0])
            q_index = 0
            # line = "\t".join(line.split("\t")[1:])
            key = " ".join(line.split("\t")[0:2])
            q_index_map[key] = q_index
            if q_index not in q_content_map:
                q_content_map[q_index] = []
            q_content_map[q_index].append(key)
            sentence = line.split("\t")[0]
            question = line.split("\t")[1]
            answer = line.split("\t")[2]
            type = line.split("\t")[-1]
            if key not in result_map:
                result_map[key] = []
                type_map[key] = type
                context_map[key] = [sentence, question]
                answer_map[key] = []
                prediction_count_map[key] = 0.0
                gold_count_map[key] = 0.0
                prediction_map[key] = []
            prediction = prediction_lines[i]
            prediction_map[key].append(prediction)
            answer_map[key].append(answer)
            label = line.split("\t")[3]
            if prediction == "yes":
                prediction_count_map[key] += 1.0
            if label == "yes":
                gold_count_map[key] += 1.0
            result_map[key].append(prediction == label)

        # for key in q_content_map:
        #     for i in range(0, len(q_content_map[key])):
        #         for j in range(i, len(q_content_map[key])):
        #             if q_content_map[key][i] != q_content_map[key][j]:
        #                 print(key)

        total = 0.0
        correct = 0.0
        f1 = 0.0
        type_correct = {}
        type_total = {}
        index_perf_map = {}
        for question in result_map:
            val = True
            total += 1.0
            cur_correct = 0.0
            type = type_map[question]
            if type not in type_correct:
                type_correct[type] = 0.0
                type_total[type] = 0.0
            for i, v in enumerate(result_map[question]):
                val = val and v
                if v and prediction_map[question][i] == "yes":
                    cur_correct += 1.0
            q_index = q_index_map[question]
            if val:
                correct += 1.0
                type_correct[type] += 1.0
                index_perf_map[q_index] = True
            else:
                index_perf_map[q_index] = False
            type_total[type] += 1.0
            p = 1.0
            if prediction_count_map[question] > 0.0:
                p = cur_correct / prediction_count_map[question]
            r = 1.0
            if gold_count_map[question] > 0.0:
                r = cur_correct / gold_count_map[question]
            if p + r > 0.0:
                f1 += 2 * p * r / (p + r)

        # print(total)
        order = ["Event Duration", "Event Ordering", "Stationarity", "Frequency", "Typical Time"]
        to_print = [str(100.0 * f1 / total), str(100.0 * correct / total)]
        for o in order:
            to_print.append(str(100.0 * type_correct[o] / type_total[o]))
        print("\t".join(to_print))
        print("Strict Acc.: " + str(correct / total))
        print("Avg F1: " + str(f1 / total))
        for type in type_total:
            print(type + " EM: " + str(type_correct[type] / type_total[type]))
            pass

        if self.output:
            print("Writing results to file: %s" % self.output)
            with open(self.output, "wt", encoding="UTF-8") as output:
                output.write(json.dumps({
                    "em": correct / total,
                    "f1": f1 / total
                }))
        return correct / total

    def print_errors(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="[eval]")
    parser.add_argument("--test_file",
                        required=True,
                        help="path to the csv file with gold labels.")
    parser.add_argument("--prediction_file",
                        required=True,
                        help="path to the line-by-line file containing system predictions.")
    parser.add_argument(
        '--output', '-o',
        help='Output results to this file.')

    args = parser.parse_args()
    if args.command == "eval":
        evaluator = McTacoEvaluator(args.test_file, args.prediction_file, args.output)
        evaluator.print_result()
    else:
        print("Command not found, see --help.")


if __name__ == "__main__":
    main()