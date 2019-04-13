class AnimalProcessor:

    def __init__(self, path):
        with open(path) as f:
            self.lines = [x.strip() for x in f.readlines()][1:]

    def output_weight_data(self, output_path):
        outputs = []
        for line in self.lines:
            groups = line.split("\t")
            if len(groups) < 2:
                continue
            name = groups[8]
            if groups[-13] == '':
                continue
            weight = int(float(groups[-13]))
            outputs.append(name + " weights " + "[MASK]" + " grams.\t" + str(weight))
        with open(output_path, "w") as f_out:
            for o in outputs:
                f_out.write(o + "\n")


if __name__ == "__main__":
    processor = AnimalProcessor("samples/animal.txt")
    processor.output_weight_data("samples/animal_weight.txt")
