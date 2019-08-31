import matplotlib.pyplot as plt


class LossVisualizer:

    def __init__(self, file_path):
        self.file_path = file_path
        self.value_map = {}
        self.load_file()

    def load_file(self):
        lines = [x.strip() for x in open(self.file_path).readlines()]
        for line in lines:
            title = line.split(":")[0]
            if title not in self.value_map:
                self.value_map[title] = []
            raw = line.split(":")[1]
            if "tensor" in raw:
                number = float(raw[8:14].split(",")[0])
            else:
                number = float(raw[1:])
            self.value_map[title].append(number)

    def visualize(self, key):
        plt.plot(self.value_map[key][1:])
        plt.show()


# visualizer = LossVisualizer("bert_classification_only/loss_log.txt")
# visualizer = LossVisualizer("bert_classification_only_freeze/loss_log.txt")
# visualizer = LossVisualizer("bert_classification_only_continue/loss_log.txt")
# visualizer = LossVisualizer("bert_classification/loss_log.txt")
# visualizer = LossVisualizer("bert_comparison/loss_log.txt")
# visualizer = LossVisualizer("bert_comb/loss_log.txt")
# visualizer = LossVisualizer("bert_comb_conceptnet/loss_log.txt")
# visualizer = LossVisualizer("bert_comb_dummy_1e4_40/loss_log.txt")
# visualizer = LossVisualizer("bert_comb_dummy/loss_log.txt")
# visualizer = LossVisualizer("bert_comb_clip_high_e/loss_log.txt")
visualizer = LossVisualizer("loss_log.txt")

# visualizer = LossVisualizer("loss_log.txt")
visualizer.visualize("Total Loss")

