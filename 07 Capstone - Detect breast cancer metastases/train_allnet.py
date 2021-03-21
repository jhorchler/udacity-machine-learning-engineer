# --- imports ---
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import argparse
import copy
import time
import csv
# -
import utils
import gcnn

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # trains and test CNN and G-CNN                                           # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# --- get arguments ---
parser = argparse.ArgumentParser(description='PyTorch Capstone Project')
# should the model be saved
parser.add_argument('--save-model',
                    action='store_true',
                    default=False,
                    help='should the best model be saved after training')
# get name of model to train
parser.add_argument('--training-name',
                    choices=['gcnn', 'cnn'],
                    required=True,
                    help='model to train and test')
args = parser.parse_args()

# --- define constants ---
# calculated mean and std on PCam training dataset // see Jupyter Notebook
PCAM_TRAIN_MEAN = np.array([0.70075643, 0.538358, 0.69162095])
PCAM_TRAIN_STD = np.array([0.2349793, 0.27740902, 0.21289322])
LOADER_PARAMETERS = {'batch_size': 64, 'num_workers': 3}
NN_OUTPUT_SIZE = 2
OPT_LEARNING_RATE = 1e-3
OPT_MOMENTUM = 0.9
OPT_EPSILON = 1e-7
OPT_WEIGHT_DECAY = 1e-4
CLR_MIN_LR = 1e-3
CLR_MAX_LR = 0.1
NUM_EPOCHS = 24

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
train_x, train_y = utils.get_matrixes(phase='train',
                                      transform=data_transforms['train'])
valid_x, valid_y = utils.get_matrixes(phase='valid',
                                      transform=data_transforms['valid'])
test_x, test_y = utils.get_matrixes(phase='test',
                                    transform=data_transforms['test'])
datasets = {
    'train': utils.PCamDataset(train_x, train_y),
    'valid': utils.PCamDataset(valid_x, valid_y),
    'test': utils.PCamDataset(test_x, test_y),
}

# --- define dataloaders ---
dataloaders = {
    'train': DataLoader(datasets['train'], shuffle=True, **LOADER_PARAMETERS),
    'valid': DataLoader(datasets['valid'], shuffle=False, **LOADER_PARAMETERS),
    'test': DataLoader(datasets['test'], shuffle=False, **LOADER_PARAMETERS),
}
dataset_sizes = {x: len(datasets[x]) for x in list(datasets.keys())}

# --- set device and send model to it ---
# the VM does have two GPUs; one is used for AllGConvNet, the other
# to train AllConvNet - this is to do a second comparison
if args.training_name == 'gcnn':
    device = torch.device('cuda:0')
    model = gcnn.AllGConvNet(input_channels=3, no_classes=NN_OUTPUT_SIZE)
else:
    device = torch.device('cuda:1')
    model = gcnn.AllConvNet(input_channels=3, no_classes=NN_OUTPUT_SIZE)
model.to(device)

# --- define loss function ---
# both models return without logits - hence calculate CrossEntropyLoss which
# combines nn.LogSoftmax() and nn.NLLLoss()
loss_func = nn.CrossEntropyLoss()

# --- define optimizer ---
# will use RMSprop as this was used in the inception training run
optimizer = optim.RMSprop(model.parameters(),
                          lr=OPT_LEARNING_RATE,
                          eps=OPT_EPSILON,
                          momentum=OPT_MOMENTUM,
                          weight_decay=OPT_WEIGHT_DECAY,
                          centered=True)

# --- define learning rate adjustment function ---
lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer,
                                           base_lr=CLR_MIN_LR,
                                           max_lr=CLR_MAX_LR)

# --- train and validate ---
best_model = copy.deepcopy(model.state_dict())
best_acc = 0.0
train_acc_history = []
train_loss_history = []
valid_acc_history = []
valid_loss_history = []
since = time.time()
# start training
for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    print('-' * 15)
    print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
    print('-' * 15)
    # train one epoch and get performance
    train_epoch_acc, train_epoch_loss = gcnn.train_one_epoch(
        model,
        loaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=loss_func,
        optimizer=optimizer,
        device=device)
    # adjust learning rate
    lr_scheduler.step()
    # validate the model for one epoch and get performance
    valid_epoch_acc, valid_epoch_loss = gcnn.validate_model(
        model,
        loaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=loss_func,
        device=device)
    time_elapsed = time.time() - epoch_start
    print('Epoch {} completed in {:.0f}m {:.0f}s'.format(
        epoch + 1, time_elapsed // 60, time_elapsed % 60))
    # store statistics
    train_acc_history.append(train_epoch_acc)
    train_loss_history.append(train_epoch_loss)
    valid_acc_history.append(valid_epoch_acc)
    valid_loss_history.append(valid_epoch_loss)
    # store best performance
    if valid_epoch_acc > best_acc:
        best_acc = valid_epoch_acc
        best_model = copy.deepcopy(model.state_dict())
#
# load best model
model.load_state_dict(best_model)
# print summary
time_elapsed = time.time() - since
print('-' * 15)
print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                     time_elapsed % 60))
print('Best validation Accuracy: {:4f}'.format(best_acc))

# --- test on testset ---
since = time.time()
# start test
model.eval()
predictions = []
running_corrects = 0
# disable gradient history tracking
with torch.no_grad():
    for x, y in tqdm(dataloaders['test'], desc='Testing', unit='batches'):
        # copy variables to GPU
        x = x.to(device)
        y = y.long().reshape(1, -1).squeeze().to(device)
        # get network output
        outputs = model(x)
        # get probabilities and predictions
        values, preds = torch.max(outputs.data, 1)
        # calculate number of correct predicted and extend list
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
                                           dataset_sizes['valid']))

# --- write output files ---
with open('{}.csv'.format(args.training_name), mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'phase', 'accuracy', 'loss'])
    for epoch, epoch_hist in enumerate(
            zip(valid_acc_history, valid_loss_history), 1):
        writer.writerow([epoch, 'Validation', *epoch_hist])
    for epoch, epoch_hist in enumerate(
            zip(train_acc_history, train_loss_history), 1):
        writer.writerow([epoch, 'Training', *epoch_hist])
with open('{}-test.csv'.format(args.training_name), mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['prediction', 'label'])
    for pred, label in predictions:
        writer.writerow([pred, label])

# --- save model ---
if args.save_model:
    torch.save(model.state_dict(), '{}.pth'.format(args.training_name))
