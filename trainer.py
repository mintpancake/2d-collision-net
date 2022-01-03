import math

import torch
from torch import nn

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from net import Net
from dataset import Data

DATA = 'data1'
SCENE = 'scene'
OBJ = 'obj'

if __name__ == '__main__':
    batch_size = 50
    learning_rate = 1e-4
    epochs = 1001

    train_data = Data(DATA, SCENE, OBJ, 0, 800)
    val_data = Data(DATA, SCENE, OBJ, 800, 1000)

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, drop_last=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}!')

    model = Net().to(device)

    loss_fn = nn.BCEWithLogitsLoss().to(device) #reduction="none"
    # optimizer = torch.optim.SGD(
    # model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    f_train_loss = open(f'logs/{DATA}/train_loss.csv', 'w')
    f_train_loss.write('epoch,loss\n')
    f_train_error = open(f'logs/{DATA}/train_error.csv', 'w')
    f_train_error.write('epoch,error\n')
    f_val_loss = open(f'logs/{DATA}/val_loss.csv', 'w')
    f_val_loss.write('epoch,loss\n')
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

        if t % 50 == 0:
            torch.save(model.state_dict(), f'models/{DATA}/model_{t}.pth')

    f_train_loss.close()
    f_train_error.close()
    f_val_loss.close()
    f_val_error.close()

    print(f'Complete training with {epochs} epochs!')
    print('Done!')
