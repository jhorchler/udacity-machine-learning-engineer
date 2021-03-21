from groupy.gconv.pytorch_gconv.splitgconv2d import (
    P4ConvZ2, P4ConvP4, SplitGConv2D)
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
from typing import Tuple, Dict
from torch import device
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.data import ImgAugTransform
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import time
import copy
import csv

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #   store everything except ImgAugTransform in one file                   # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class GConvNet(nn.Module):
    # example taken from
    # https://github.com/adambielski/pytorch-gconv-experiments/blob/master/mnist/mnist.py
    def __init__(self, input_channels: int = 1, no_classes: int = 10) -> None:
        super(GConvNet, self).__init__()

        self.in_channels = input_channels
        self.classes = no_classes

        # make_gconv_indices.py forces kernel size to be uneven
        self.gconv1 = P4ConvZ2(self.in_channels, 96, 3)
        self.gconv2 = P4ConvP4(96, 96, 3)
        self.gconv3 = P4ConvP4(96, 192, 3)
        self.gconv4 = P4ConvP4(192, 192, 3)
        self.gconv5 = P4ConvP4(192, 384, 3)
        self.gconv6 = P4ConvP4(384, 384, 3)
        self.fc1 = nn.Linear(1*1*4*384, 1024)
        self.fc2 = nn.Linear(1024, self.classes)

        # initialize weight and bias
        for m in self.modules():
            if isinstance(m, SplitGConv2D):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gconv1(x))
        x = F.relu(self.gconv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.dropout(x, p=0.2)
        x = F.relu(self.gconv3(x))
        x = F.relu(self.gconv4(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.dropout(x, p=0.3)
        x = F.relu(self.gconv5(x))
        x = F.relu(self.gconv6(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.4)
        x = self.fc2(x)
        return x


# CIFAR10  does not have validation set
# creating one was omitted
# hence only train
def train(model: nn.Module, loaders: Dict, dataset_sizes: Dict,
          criterion, optimizer, scheduler, num_epochs: int = 25,
          device: device = 'cpu') -> Tuple:
    since = time.time()
    # track train performance
    train_acc_hist = []
    train_loss_hist = []
    # store best performance
    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('-' * 15)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 15)

        # set model to train mode
        model.train()

        # reset statistics
        running_loss = 0.0
        running_corrects = 0

        for data in tqdm(loaders['train'], desc='Training', unit='batches'):
            x, y = data
            # copy data to GPU if available
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # zero the gradients
            optimizer.zero_grad()

            # forward + loss + backward + optimize
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            # get predictions
            _, preds = torch.max(logits, 1)

            # update statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == y)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        # save statistics
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())
        train_acc_hist.append(epoch_acc.item())
        train_loss_hist.append(epoch_loss)
        scheduler.step()

        time_elapsed = time.time() - epoch_start
        print('Epoch {} completed: {:.0f}m {:.0f}s - Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60, epoch_loss, epoch_acc*100))

    # print summary
    time_elapsed = time.time() - since
    print('-' * 15)
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                         time_elapsed % 60))
    print('Best accuracy: {:4f}'.format(best_acc))

    # load best weights
    model.load_state_dict(best_model)
    return model, train_acc_hist, train_loss_hist


# --- define constants ---
# calculated mean and std on PCam training dataset // see Jupyter Notebook
TRAIN_MEAN = np.array([0.4914, 0.4822, 0.4465])
TRAIN_STD = np.array([0.2023, 0.1994, 0.2010])
# CIFAR10 on Tesla T4 can easily be stored on GPU RAM
LOADER_PARAMETERS = {'batch_size': 128, 'num_workers': 3, 'pin_memory': True}
NN_OUTPUT_SIZE = 10
OPT_LEARNING_RATE = 0.001
LRSCHED_EPOCH_PER_DECAY = 50
LRSCHED_GAMMA = 0.25
NUM_EPOCHS = 500

# --- set device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- define transforms ---
data_transforms = {
    'train': transforms.Compose([
        ImgAugTransform(),
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
    ]),
}

# --- define datasets---
datasets = {
    'train': CIFAR10(root='./data', train=True, download=True,
                     transform=data_transforms['train']),
    'test': CIFAR10(root='./data', train=False, download=True,
                    transform=data_transforms['test']),
}

# --- define dataloaders ---
dataloaders = {
    'train': DataLoader(datasets['train'], batch_size=128, shuffle=True,
                        num_workers=3),
    'test': DataLoader(datasets['test'], batch_size=100, shuffle=False,
                       num_workers=3),
}
ds_sizes = {x: len(datasets[x]) for x in list(datasets.keys())}
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')

# --- set device and send model to it ---
model = GConvNet(input_channels=3, no_classes=NN_OUTPUT_SIZE)
model.to(device, non_blocking=True)

# --- define loss function ---
# both models return without logits - hence calculate CrossEntropyLoss which
# combines nn.LogSoftmax() and nn.NLLLoss()
loss_func = nn.CrossEntropyLoss()

# --- define optimizer ---
optimizer = optim.Adam(model.parameters(), lr=OPT_LEARNING_RATE)

# --- define learning rate adjustment function ---
lr_scheduler = lr_scheduler.StepLR(optimizer, gamma=LRSCHED_GAMMA,
                                   step_size=LRSCHED_EPOCH_PER_DECAY)

# --- train  ---
model, tah, tlh = train(model=model, loaders=dataloaders,
                        dataset_sizes=ds_sizes, criterion=loss_func,
                        optimizer=optimizer, scheduler=lr_scheduler,
                        num_epochs=NUM_EPOCHS, device=device)

# --- test on testset ---
since = time.time()
# start test
model.eval()
predictions = []
running_corrects = 0
with torch.no_grad():
    for data in tqdm(dataloaders['test'], desc='Testing', unit='batches'):
        x, y = data
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        outputs = model(x)
        values, preds = torch.max(outputs.data, 1)
        running_corrects += torch.sum(preds == y)
        predictions.extend(
            list(zip(outputs.data[:, 1].cpu().numpy(),
                     y.data.cpu().numpy())))
# print summary
time_elapsed = time.time() - since
print('-' * 15)
print('Test completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                 time_elapsed % 60))
print('Accuracy on test set: {:4f}'.format(running_corrects.double() /
                                           ds_sizes['test']))

# --- write output files ---
with open('main-training.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'phase', 'accuracy', 'loss'])
    for epoch, epoch_hist in enumerate(
            zip(tah, tlh), 1):
        writer.writerow([epoch, 'Training', *epoch_hist])
with open('main-test.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['prediction', 'label'])
    for pred, label in predictions:
        writer.writerow([pred, label])
