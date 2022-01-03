import math

import torch
from torch import nn

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from net import Net
from dataset import Data

model = Net().to('cuda')
DATA = 'data1'
SCENE = 'scene'
OBJ = 'obj'
MODEL_NUMBER = 1000
DATA_SIZE = 13
data_name = DATA
test_data_name = f'{DATA}_test'
device = 'cuda'

test_data = Data(test_data_name, SCENE, OBJ, 0, DATA_SIZE)

test_dataloader = DataLoader(
    test_data, batch_size=DATA_SIZE, shuffle=False, drop_last=False)
for batch, (data, label) in enumerate(test_dataloader):
    sc, oc, pos = data
    sc, oc, pos, label = sc.to(device), oc.to(
        device), pos.to(device), label.to(device)
    res = model(sc, oc, pos)