from torch import device
from typing import Dict, Tuple
from tqdm import tqdm
import torch.nn as nn
import torch
import time
import copy


def train_one_epoch(model: nn.Module, loaders: Dict, dataset_sizes: Dict,
                    criterion, optimizer, device: device) -> Tuple:
    """Trains the given model exactly one epoch.

    Args:
      model (nn.Module): a nn.Module network
      loaders (Dict): dictionary holding dataloaders for training dataset
      dataset_sizes (Dict): dictionary holding the sizes of the datasets
                            loaded by `dataloaders`
      criterion: a instance of torch.nn loss function
      optimizer: a instance of torch.optim optimizer function
      device (torch.device): the device to run on

    Returns:
      A tuple holding
        epoch_acc: accuracy of the epoch
        epoch_loss: the loss of the epoch
    """
    # switch to train mode
    model.train()

    # reset statistics
    running_loss = 0.0
    running_corrects = 0

    # loop over data ..
    for x, y in tqdm(loaders['train'], desc='  Training', unit='batches'):
        # send data to GPU
        x = x.to(device)
        y = y.long().reshape(1, -1).squeeze().to(device)

        # zero the gradients
        optimizer.zero_grad()

        # forward through network
        with torch.set_grad_enabled(True):
            # compute output
            logits = model(x)

            # get predictions
            _, preds = torch.max(logits, 1)

            # compute loss
            loss = criterion(logits, y)

            # compute gradients and optimize learnable parameters
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * x.size(0)
        running_corrects += torch.sum(preds == y.data)

    # epoch stats
    epoch_acc = running_corrects.double() / dataset_sizes['train']
    epoch_loss = running_loss / dataset_sizes['train']
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc*100))
    return epoch_acc.item(), epoch_loss


def validate_model(model: nn.Module, loaders: Dict, dataset_sizes: Dict,
                   criterion, device: device) -> Tuple:
    """Validates the given model using a validation set exactly one epoch.

    Args:
      model (nn.Module): a nn.Module network
      loaders (Dict): dictionary holding dataloaders for validation dataset
      dataset_sizes (Dict): dictionary holding the sizes of the datasets
                            loaded by `dataloaders`
      criterion: a instance of torch.nn loss function
      device (torch.device): the device to run on

    Returns:
      A tuple holding
        epoch_acc: accuracy of the validation epoch
        epoch_loss: the loss of the validation epoch
    """
    # switch to eval mode
    model.eval()

    # reset statistics
    running_loss = 0.0
    running_corrects = 0

    # loop over data ..
    for x, y in tqdm(loaders['valid'], desc='Validation', unit='batches'):
        # send data to GPU
        x = x.to(device)
        y = y.long().reshape(1, -1).squeeze().to(device)

        # forward without tracking gradient history
        with torch.set_grad_enabled(False):
            # compute output
            logits = model(x)

            # get predictions
            _, preds = torch.max(logits, 1)

            # get loss
            loss = criterion(logits, y)

        # statistics
        running_loss += loss.item() * x.size(0)
        running_corrects += torch.sum(preds == y.data)

    # epoch stats
    epoch_acc = running_corrects.double() / dataset_sizes['valid']
    epoch_loss = running_loss / dataset_sizes['valid']
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc*100))
    return epoch_acc.item(), epoch_loss


def train_and_validate(model: nn.Module, loaders: Dict, dataset_sizes: Dict,
                       criterion, optimizer, scheduler, num_epochs: int = 25,
                       device: device = 'cpu') -> Tuple:
    """Trains and validates a given model over the number of epochs given.

    Args:
      model (nn.Module): a nn.Module network
      loaders (Dict): dictionary holding dataloaders for training and
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
        val_acc_hist: a list of all accuracies in validation
        val_loss_hist: a list of all losses in validation
        train_acc_hist: a list of all accuracies in training
        train_loss_hist: a list of all losses in validation
    """
    since = time.time()
    # track validation performance
    val_acc_hist = []
    val_loss_hist = []
    # track training performance
    train_acc_hist = []
    train_loss_hist = []
    # track best validation performance
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
                model.train()
            else:
                model.eval()

            # reset statistics
            running_loss = 0.0
            running_corrects = 0

            for x, y in tqdm(loaders[phase], desc=phase, unit='batches'):
                # copy data to GPU, if GPU is used. For CPU does nothing.
                x = x.to(device, non_blocking=True)
                y = y.long().to(device, non_blocking=True)

                # zero the gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # get network output
                    logits = model(x)

                    # calculate the loss
                    loss = criterion(logits, y)

                    # get predictions
                    _, preds = torch.max(logits, 1)

                    # update parameters if in train mode
                    if phase == 'train':
                        # calculate gradients
                        loss.backward()
                        # update weights
                        optimizer.step()
                        # update learning rate
                        scheduler.step()

                # update statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == y.data).float()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc*100))

            # save statistic history
            if phase == 'valid':
                val_acc_hist.append(epoch_acc.item())
                val_loss_hist.append(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model.state_dict())
            else:
                train_acc_hist.append(epoch_acc.item())
                train_loss_hist.append(epoch_loss)

        time_elapsed = time.time() - epoch_start
        print('Epoch {} completed: {:.0f}m {:.0f}s'.format(epoch + 1,
                                                           time_elapsed // 60,
                                                           time_elapsed % 60))

    # print summary
    time_elapsed = time.time() - since
    print('-' * 15)
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                         time_elapsed % 60))
    print('Best validation Accuracy: {:4f}'.format(best_acc))

    # load best weights
    model.load_state_dict(best_model)
    return model, val_acc_hist, val_loss_hist, train_acc_hist, train_loss_hist
