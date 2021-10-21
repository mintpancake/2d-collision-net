import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, sc_file, oc_file, gt_file):
        self.sc = self.load(sc_file)
        self.oc = self.load(oc_file)
        self.pos = torch.Tensor([0, 0])
        self.label = self.load_gt(gt_file)

    def __getitem__(self, index):
        data = [self.sc, self.oc, self.pos]
        return data, self.label

    def __len__(self):
        return 1

    def load(self, file):
        data = []
        f = open(file, 'r')
        line = f.readline()
        while line:
            x, y = line.strip('\n').split(' ')
            x, y = float(x), float(y)
            data.append([x, y])
            line = f.readline()
        f.close()
        return torch.Tensor(data)

    def load_gt(self, file):
        data = []
        f = open(file, 'r')
        line = f.readline()
        while line:
            gt = float(line.strip('\n'))
            data.append(gt)
            line = f.readline()
        f.close()
        return torch.Tensor(data)
