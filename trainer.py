import math

import torch
from torch import nn

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from utils import ensure_dir, read_config

from net import Net
from dataset import Data

CFG = read_config()
DATA = CFG['data_name']
SCENE = CFG['scene_name']
OBJ = CFG['obj_name']
CHECKPOINT = CFG['checkpoint']


def run():
    batch_size = CFG['batch_size']
    learning_rate = CFG['learning_rate']
    epochs = CFG['epochs']

    data_size = CFG['data_size']
    train_size = int(data_size*CFG['tran_test_split'])

    train_data = Data(DATA, SCENE, OBJ, 0, train_size)
    val_data = Data(DATA, SCENE, OBJ, train_size, data_size)

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, drop_last=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}!')

    model = Net().to(device)

    loss_fn = nn.BCEWithLogitsLoss().to(device)  # reduction="none"
    # optimizer = torch.optim.SGD(
    # model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    ensure_dir(f'logs/{DATA}/train_loss.csv')
    f_train_loss = open(f'logs/{DATA}/train_loss.csv', 'w')
    f_train_loss.write('epoch,loss\n')
    ensure_dir(f'logs/{DATA}/train_error.csv')
    f_train_error = open(f'logs/{DATA}/train_error.csv', 'w')
    f_train_error.write('epoch,error\n')
    ensure_dir(f'logs/{DATA}/val_loss.csv')
    f_val_loss = open(f'logs/{DATA}/val_loss.csv', 'w')
    f_val_loss.write('epoch,loss\n')
    ensure_dir(f'logs/{DATA}/val_error.csv')
    f_val_error = open(f'logs/{DATA}/val_error.csv', 'w')
    f_val_error.write('epoch,error\n')
    for t in range(epochs):
        print(f'epoch {t}')

        model.train()
        size = len(train_dataloader.dataset)
        batches = math.ceil(size/batch_size)
        mean_loss = 0
        mean_error = 0
        for batch, (data, label) in enumerate(train_dataloader):
            sc, oc, pos = data
            sc, oc, pos, label = sc.to(device), oc.to(
                device), pos.to(device), label.to(device)
            curr_batch_size = sc.shape[0]
            pred = model(sc, oc, pos)
            pred = pred.squeeze()

            losses = loss_fn(pred, label)
            loss = losses.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # losses.backward(losses.clone().detach())
            # losses.backward()
            # optimizer.step()
            optimizer.zero_grad()
            mean_loss += losses.mean().item()

            prob = torch.sigmoid(pred)
            prob[prob >= 0.5] = 1
            prob[prob < 0.5] = 0
            error = torch.sum(torch.abs(prob-label))/curr_batch_size
            mean_error += error

        mean_loss /= batches
        mean_error /= batches
        f_train_loss.write(f'{t},{mean_loss}\n')
        print(f'train loss: {mean_loss}')
        f_train_error.write(f'{t},{mean_error}\n')
        print(f'train error: {mean_error}')

        model.eval()
        size = len(val_dataloader.dataset)
        batches = math.ceil(size/batch_size)
        mean_loss = 0
        mean_error = 0
        with torch.no_grad():
            for batch, (data, label) in enumerate(val_dataloader):
                sc, oc, pos = data
                sc, oc, pos, label = sc.to(device), oc.to(
                    device), pos.to(device), label.to(device)
                curr_batch_size = sc.shape[0]
                pred = model(sc, oc, pos)
                pred = pred.squeeze()

                losses = loss_fn(pred, label)
                mean_loss += losses.mean().item()

                prob = torch.sigmoid(pred)
                prob[prob >= 0.5] = 1
                prob[prob < 0.5] = 0
                error = torch.sum(torch.abs(prob-label))/curr_batch_size
                mean_error += error

        mean_loss /= batches
        mean_error /= batches
        f_val_loss.write(f'{t},{mean_loss}\n')
        print(f'val loss: {mean_loss}')
        f_val_error.write(f'{t},{mean_error}\n')
        print(f'val error: {mean_error}')

        if t % CHECKPOINT == 0:
            ensure_dir(f'models/{DATA}/model_{t}.pth')
            torch.save(model.state_dict(), f'models/{DATA}/model_{t}.pth')

    f_train_loss.close()
    f_train_error.close()
    f_val_loss.close()
    f_val_error.close()

    print(f'Complete training with {epochs} epochs!')


if __name__ == "__main__":
    run()
