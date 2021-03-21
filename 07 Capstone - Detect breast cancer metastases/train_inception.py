from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import csv
# ----------
import cnn
import utils


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # tests both inception networks
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# --- get model name ---
training_name = sys.argv[1]

# --- define default transformations ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(cnn.INCEPTION_INPUT_SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(cnn.INCEPTION_MEAN, cnn.INCEPTION_STD)
    ]),
    'valid': transforms.Compose([
        transforms.Resize(cnn.INCEPTION_INPUT_SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(cnn.INCEPTION_MEAN, cnn.INCEPTION_STD)
    ]),
}

# --- load default train and validation datasets ---
train_x, train_y = utils.get_matrixes(phase='train',
                                      transform=data_transforms['train'])
valid_x, valid_y = utils.get_matrixes(phase='valid',
                                      transform=data_transforms['valid'])
datasets = {
    'train': utils.PCamDataset(train_x, train_y),
    'valid': utils.PCamDataset(valid_x, valid_y),
}

# --- create the dataloaders ---
dataloaders = {x: DataLoader(datasets[x], **cnn.LOADER_PARAMETERS) for x in
               list(datasets.keys())}
dataset_sizes = {x: len(datasets[x]) for x in list(datasets.keys())}

# one network per GPU
if training_name == 'inception_1':
    device = torch.device('cuda:0')
else:
    device = torch.device('cuda:1')

# --- initialize inception network ----
inception = cnn.initialize_model(cnn.OUTPUT_SIZE, device)

# --- as this is a binary classification problem choose cross entropy loss ---
loss_func = nn.CrossEntropyLoss()

# --- default optimizer: like in inception architecture use RMSprop ---
optimizer = optim.RMSprop(inception.parameters(), lr=cnn.LEARNING_RATE,
                          momentum=cnn.MOMENTUM, eps=cnn.EPSILON,
                          weight_decay=cnn.WEIGHT_DECAY)

# --- define scheduler to decay learning rate ---
scheduler = lr_scheduler.StepLR(optimizer, step_size=cnn.NUM_EPOCH_PER_DECAY,
                                gamma=cnn.GAMMA)

# --- OVERRIDE DEFAULT SETTINGS IF SECOND INCEPTION MODEL ---
if training_name == 'inception_2':
    # change tranforms
    data_transforms['train'] = transforms.Compose([
        transforms.Resize(cnn.INCEPTION_INPUT_SHAPE),
        transforms.RandomRotation(45),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ], p=0.3),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomAffine(degrees=0, shear=45),
        ], p=0.3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3),
            transforms.ColorJitter(contrast=0.3),
            transforms.ColorJitter(saturation=0.3),
            transforms.ColorJitter(hue=0.3),
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(cnn.INCEPTION_MEAN, cnn.INCEPTION_STD)
    ])

    # reload dataset
    train_x, train_y = utils.get_matrixes(phase='train',
                                          transform=data_transforms['train'])
    datasets['train'] = utils.PCamDataset(train_x, train_y)
    dataloaders['train'] = DataLoader(datasets['train'],
                                      **cnn.LOADER_PARAMETERS)
    dataset_sizes['train'] = len(datasets['train'])

    # change optimizer and hence scheduler
    optimizer = optim.SGD(inception.parameters(), lr=cnn.LEARNING_RATE,
                          momentum=cnn.MOMENTUM, weight_decay=cnn.WEIGHT_DECAY,
                          nesterov=True)
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=cnn.NUM_EPOCH_PER_DECAY,
                                    gamma=cnn.GAMMA)

# --- train the network ---
bm, vah, vlh, tah, tlh = cnn.train_and_validate(inception, dataloaders,
                                                dataset_sizes, loss_func,
                                                optimizer, scheduler,
                                                num_epochs=cnn.NUM_EPOCHS,
                                                device=device)

# --- write logfile and save model ---
with open('{}.csv'.format(training_name), mode='w', newline='') as histfile:
    writer = csv.writer(histfile)
    writer.writerow(['epoch', 'phase', 'accuracy', 'loss'])
    for epoch, epoch_hist in enumerate(zip(vah, vlh), 1):
        writer.writerow([epoch, 'validation', *epoch_hist])
    for epoch, epoch_hist in enumerate(zip(tah, tlh), 1):
        writer.writerow([epoch, 'training', *epoch_hist])
#
torch.save(bm.state_dict(), '{}.pth'.format(training_name))
