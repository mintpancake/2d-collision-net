import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import ensure_dir, read_config

from net import Net
from dataset import Data

CFG = read_config()
DATA = CFG['data_name']
SCENE = CFG['scene_name']
OBJ = CFG['obj_name']
MODEL_NUMBER = CFG['test_model_number']
DATA_SIZE = CFG['test_data_size']


def run():
    data_name = DATA
    test_data_name = f'{DATA}_test'

    batch_size = DATA_SIZE

    test_data = Data(test_data_name, SCENE, OBJ, 0, DATA_SIZE)

    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, drop_last=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}!')

    model = Net().to(device)
    model.load_state_dict(torch.load(
        f'models/{data_name}/model_{MODEL_NUMBER}.pth'))

    ensure_dir(f'logs/{test_data_name}/test_error.csv')
    f = open(f'logs/{test_data_name}/test_error.csv', 'w')
    model.eval()
    with torch.no_grad():
        size = len(test_dataloader.dataset)
        batches = math.ceil(size/batch_size)
        for batch, (data, label) in enumerate(test_dataloader):
            sc, oc, pos = data
            sc, oc, pos, label = sc.to(device), oc.to(
                device), pos.to(device), label.to(device)
            curr_batch_size = sc.shape[0]
            pred = model(sc, oc, pos, test=True, data_name=test_data_name)
            pred = pred.squeeze()
            prob = torch.sigmoid(pred)
            prob[prob >= 0.5] = 1
            prob[prob < 0.5] = 0
            print(prob)
            error = torch.sum(torch.abs(prob-label))/curr_batch_size
            print(prob.shape)
            print(f'Number: {curr_batch_size}')
            print(f'Error: {error}')
    f.write(f'{error}\n')
    f.close()
    for i in range(DATA_SIZE):
        ensure_dir(f'data/{test_data_name}/pred/pred_{i}.txt')
        f = open(f'data/{test_data_name}/pred/pred_{i}.txt', 'w')
        for datum in prob[i]:
            f.write(f'{int(datum)}\n')
        f.close()


if __name__ == '__main__':
    run()
