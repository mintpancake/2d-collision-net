import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, data, scene, obj, start, stop):
        self.num = stop-start
        self.scene_pc = self.load(f'data/{data}/scene_pc/{scene}_pc.txt')
        self.pos = []
        self.label = []
        for i in range(start, stop):
            self.pos.append(self.load_pos(f'data/{data}/pos/pos_{i}.txt'))
            self.label.append(self.load_gt(f'data/{data}/gt/gt_{i}.txt'))
        self.scene_pc = torch.Tensor(self.scene_pc)
        self.pos = torch.Tensor(self.pos)
        self.label = torch.Tensor(self.label)

        # self.obj_pc = self.load(f'data/{data}/obj_pc/{obj}_pc.txt')
        # self.obj_pc = torch.Tensor(self.obj_pc)

        self.obj_pc = []
        for i in range(start, stop):
            self.obj_pc.append(
                self.load(f'data/{data}/obj_pc/moved/{obj}_pc_{i}.txt'))
        self.obj_pc = torch.Tensor(self.obj_pc)

    def __getitem__(self, index):
        data = [self.scene_pc, self.obj_pc[index], self.pos[index]]
        # data = [self.scene_pc, self.obj_pc, self.pos[index]]
        return data, self.label[index]

    def __len__(self):
        return self.num

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
        return data

    def load_pos(self, file):
        f = open(file, 'r')
        line = f.readline()
        x, y = line.strip('\n').split(' ')
        x, y = float(x), float(y)
        data = [x, y]
        f.close()
        return data

    def load_gt(self, file):
        data = []
        f = open(file, 'r')
        line = f.readline()
        while line:
            gt = float(line.strip('\n'))
            data.append(gt)
            line = f.readline()
        f.close()
        return data
