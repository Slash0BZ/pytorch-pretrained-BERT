from pytorch_pretrained_bert.modeling import *


class VerbPhysicsObjectReader:

    def __init__(self, data_path):
        self.data_path = data_path
        self.pairs = []
        self.run()

    def run(self):
        f = open(self.data_path, "r")
        lines = [x.strip() for x in f.readlines()][1:]
        for line in lines:
            groups = line.split(",")
            obj_1 = groups[1]
            obj_2 = groups[2]
            size_agree = int(groups[3])
            size_val = int(groups[4])
            weight_agree = int(groups[5])
            weight_val = int(groups[6])
            strength_agree = int(groups[7])
            strength_val = int(groups[8])
            rigidness_agree = int(groups[9])
            rigidness_val = int(groups[10])
            speed_agree = int(groups[11])
            speed_val = int(groups[12])

            if size_agree == 3 and size_val != -42:
                self.pairs.append([obj_1, obj_2, "size", size_val])
            if weight_agree == 3 and weight_val != -42:
                self.pairs.append([obj_1, obj_2, "weight", weight_val])
            if strength_agree == 3 and strength_val != -42:
                self.pairs.append([obj_1, obj_2, "strength", strength_val])
            if rigidness_agree == 3 and rigidness_val != -42:
                self.pairs.append([obj_1, obj_2, "rigidness", rigidness_val])
            if speed_agree == 3 and speed_val != -42:
                self.pairs.append([obj_1, obj_2, "speed", speed_val])


class Runner:

    def __init__(self):
        self.train_set = VerbPhysicsObjectReader("samples/verbphysics/train-5/train.csv")
        self.dev_set = VerbPhysicsObjectReader("samples/verbphysics/train-5/dev.csv")
        self.test_set = VerbPhysicsObjectReader("samples/verbphysics/train-5/test.csv")

    def output_weight(self):
        f_train = open("samples/verbphysics/dummy/train.txt", "w")
        f_test = open("samples/verbphysics/dummy/test.txt", "w")
        for p in self.train_set.pairs + self.dev_set.pairs:
            if p[2] != "weight":
                continue
            l = p[0] + " weights [MASK] [MASK]."
            l += "\t" + p[1] + " weights [MASK] [MASK].\t"
            if int(p[3]) == -1:
                l += "larger"
            elif int(p[3]) == 0:
                l += "same"
            else:
                l += "smaller"

            f_train.write(l + "\n")

        for p in self.test_set.pairs:
            if p[2] != "weight":
                continue
            l = p[0] + " weights [MASK] [MASK]."
            l += "\t" + p[1] + " weights [MASK] [MASK].\t"
            if int(p[3]) == -1:
                l += "larger"
            elif int(p[3]) == 0:
                l += "same"
            else:
                l += "smaller"

            f_test.write(l + "\n")


r = Runner()
r.output_weight()
