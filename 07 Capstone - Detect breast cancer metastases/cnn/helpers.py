from torchvision import models
from torch import device
from typing import Dict, Tuple
from tqdm import tqdm
import torch.nn as nn
import time
import copy
import torch


def train_and_validate(model: nn.Module,
                       dataloaders: Dict, dataset_sizes: Dict, criterion,
                       optimizer, scheduler, num_epochs: int = 25,
                       device: device = 'cpu') -> Tuple:
    """Trains a given Inception3 network using training and validation
    data sets.

    Args:
      model (nn.Module): a nn.Module network
      dataloaders (Dict): dictionary holding dataloaders for training and
                          validation datasets
      dataset_sizes (Dict): dictionary holding the sizes of the datasets
                            loaded by `dataloaders`
      criterion: a instance of torch.nn loss function
      optimizer: a instance of torch.optim optimizer function
      scheduler: a instance of torch.optim.lr_scheduler
      num_epochs (int): how many epochs to train (default: 25)
      device (torch.device): the device to run on (default: cpu)

    Returns:
      A tuple holding
        model: the model having the best validation accuracy
        valid_acc_history: a list of all accuracies in validation
        valid_loss_history: a list of all losses in validation
        train_acc_history: a list of all accuracies in training
        train_loss_history: a list of all losses in validation
    """
    since = time.time()
    # track validation performance
    valid_acc_history = []
    valid_loss_history = []
    # track training performance
    train_acc_history = []
    train_loss_history = []
    # track top performance of validation phase
    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('-' * 15)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 15)
        # first train and then validate for each epoch
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            # reset statistics
            running_loss = 0.0
            running_corrects = 0

            for x, y in tqdm(dataloaders[phase], desc=phase, unit='batches'):
                x = x.to(device)
                y = y.long().reshape(1, -1).squeeze().to(device)

                # zero the gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # inception does have auxillary outputs in training mode
                    if phase == 'train':
                        # get the output of the network
                        logits, aux_logits = model(x)
                        # calculate the loss
                        loss = criterion(logits, y) + 0.4 * criterion(
                            aux_logits,
                            y)  # 0.4 is weight for auxillary classifier
                        # calculate gradients
                        loss.backward()
                        # optimize weights
                        optimizer.step()
                    else:
                        logits = model(x)
                        loss = criterion(logits, y)

                    # get predictions
                    _, preds = torch.max(logits, 1)

                # update statistics of the batch
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == y.data)

            # update epoch statistics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # save statistics
            if phase == 'valid':
                valid_acc_history.append(epoch_acc.item())
                valid_loss_history.append(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model.state_dict())
            else:
                train_acc_history.append(epoch_acc.item())
                train_loss_history.append(epoch_loss)

        time_elapsed = time.time() - epoch_start
        print('Epoch {} completed in {:.0f}m {:.0f}s'.format(epoch + 1,
                                                             time_elapsed // 60,
                                                             time_elapsed % 60))

    # print summary
    time_elapsed = time.time() - since
    print('-' * 15)
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                         time_elapsed % 60))
    print('Best validation Accuracy: {:4f}'.format(best_acc))

    # load best weights and return
    model.load_state_dict(best_model)
    return model, valid_acc_history, valid_loss_history, \
           train_acc_history, train_loss_history


def initialize_model(num_classes: int, device: device = 'cpu'):
    """creates a new Inception3 model, freezes the output layers and
    adds new layers having the given number of classes. In addition pushes
    the network to the given device.
    """

    # load pretrained weights from google inception network
    model = models.inception_v3(pretrained=True)

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # reset final layers (inception has two output layers when training)
    # only these layers are trained below
    num_features = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_features, num_classes)
    #
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # send to GPU or CPU
    return model.to(device)
