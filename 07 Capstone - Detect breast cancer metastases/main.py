# --- imports ---
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import time
import csv
import os
# -
import utils
import gcnn

# --- define constants ---
# calculated mean and std on PCam training dataset // see Jupyter Notebook
PCAM_TRAIN_MEAN = np.array([0.70075643, 0.538358, 0.69162095])
PCAM_TRAIN_STD = np.array([0.2349793, 0.27740902, 0.21289322])
# extracted images
DATA_DIR = 'data/camelyonpatch'
LOADER_PARAMETERS = {'num_workers': 3, 'pin_memory': True}
# only this batch size prevents out of memory on GPU
LOADER_BS = 64
NN_OUTPUT_SIZE = 2
# start a little higher
OPT_LEARNING_RATE = 0.05
NUM_EPOCHS = 100
LRS_NUM_EPOCH_PER_DECAY = 20
LRS_RATE_DECAY = 0.1

# --- set device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- define transforms ---
data_transforms = {
    'train': transforms.Compose([
        utils.ImgAugTransform(),
        transforms.ToTensor(),
        transforms.Normalize(PCAM_TRAIN_MEAN, PCAM_TRAIN_STD)
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(PCAM_TRAIN_MEAN, PCAM_TRAIN_STD)
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(PCAM_TRAIN_MEAN, PCAM_TRAIN_STD)
    ]),
}

# --- define datasets---
datasets = {phase: datasets.ImageFolder(os.path.join(DATA_DIR, phase),
                                        transform=data_transforms[phase])
                  for phase in ['train', 'valid', 'test']}
ds_sizes = {phase: len(datasets[phase]) for phase in list(datasets.keys())}

# --- define dataloaders ---
dataloaders = {
    'train': DataLoader(datasets['train'], batch_size=LOADER_BS, shuffle=True,
                        **LOADER_PARAMETERS),
    'valid': DataLoader(datasets['valid'], batch_size=LOADER_BS * 2,
                        shuffle=False, **LOADER_PARAMETERS),
    'test': DataLoader(datasets['test'], batch_size=LOADER_BS * 2,
                       shuffle=False, **LOADER_PARAMETERS),
}
class_names = datasets['train'].classes

# --- set device and send model to it ---
model = gcnn.GConvNet(input_channels=3, no_classes=NN_OUTPUT_SIZE)
model.to(device, non_blocking=True)

# --- define loss function ---
# both models return without logits - hence calculate CrossEntropyLoss which
# combines nn.LogSoftmax() and nn.NLLLoss()
# sum is choosen for more accurate loss output as the default calculates
# the mean over the batches
loss_func = nn.CrossEntropyLoss(reduction='sum')

# --- optimizer: use Adam as it was used with CIFAR10 ---
optimizer = optim.Adam(model.parameters(), lr=OPT_LEARNING_RATE)

# --- define learning rate adjustment function ---
lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                         step_size=LRS_NUM_EPOCH_PER_DECAY,
                                         gamma=LRS_RATE_DECAY)

# --- train and validate ---
results = gcnn.train_and_validate(model=model, loaders=dataloaders,
                                  dataset_sizes=ds_sizes, criterion=loss_func,
                                  optimizer=optimizer, scheduler=lr_scheduler,
                                  num_epochs=NUM_EPOCHS, device=device)
model, vah, vlh, tah, tlh, lh, lrh = results

# --- test on testset ---
since = time.time()
# start test; set model to eval mode
model.eval()
predictions = []
running_corrects = 0
# disable gradient history tracking
with torch.no_grad():
    for x, y in tqdm(dataloaders['test'], desc='Testing', unit='batches'):
        x = x.to(device)
        y = y.long().to(device, non_blocking=True)
        # forward
        outputs = model(x)
        # get probabilities and predictions
        values, preds = torch.max(outputs.data, 1)
        # calculate numer of correct predictions and extend list
        running_corrects += torch.sum(preds == y.data).float()
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
    for epoch, epoch_hist in enumerate(zip(vah, vlh), 1):
        writer.writerow([epoch, 'Validation', *epoch_hist])
    for epoch, epoch_hist in enumerate(zip(tah, tlh), 1):
        writer.writerow([epoch, 'Training', *epoch_hist])
with open('main-test.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['prediction', 'label'])
    for pred, label in predictions:
        writer.writerow([pred, label])
with open('main-lr.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['iteration', 'loss', 'lr'])
    for epoch, epoch_hist in enumerate(zip(lh, lrh), 1):
        writer.writerow([epoch, *epoch_hist])

# --- save model ---
# torch.save(model.state_dict(), 'main.pth')
