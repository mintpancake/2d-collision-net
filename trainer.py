import torch
from torch import nn
from torch.functional import Tensor
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from net import Net
from dataset import Data

if __name__ == '__main__':
    batch_size = 1
    learning_rate = 1e-3
    epochs = 1000

    train_data = Data('scene_point_cloud.txt',
                      'object_point_cloud.txt', 'gt.txt')

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}!')

    model = Net().to(device)

    loss_fn = nn.BCEWithLogitsLoss(reduction="mean").to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9)
    scaler = GradScaler()

    for t in range(epochs):
        # Training loop
        model.train()
        size = len(train_dataloader.dataset)
        for batch, (data, label) in enumerate(train_dataloader):
            sc, oc, pos = data
            sc, oc, pos, label = sc.to(device), oc.to(
                device), pos.to(device), label.to(device)
            pred = model(sc, oc, pos, device)
            pred = torch.reshape(pred, label.shape)

            # if t == epochs-1:
                

            losses = loss_fn(pred, label)
            # loss = losses.mean()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # losses.backward(losses.clone().detach())
            losses.backward()
            optimizer.step()

            optimizer.zero_grad()

            if t % 10 == 9:
                m = nn.Sigmoid()
                prob=m(pred.reshape([10, 10]))
                print(prob)
                print(torch.mean(pred))
                print(torch.max(prob))
                print(f'epoch: {t + 1}    loss: {losses.mean().item()}')

    torch.save(model.state_dict(), f'model.pth')
    print(f'Complete training with {epochs} epochs!')
    print('Done!')
