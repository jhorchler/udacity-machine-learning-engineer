# # # # # # # # # # # # # # # # # # # # # # #
# these steps were executed in that order   #
# in an interactive ipython session and are #
# just recorded here                        #
# # # # # # # # # # # # # # # # # # # # # # #
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch
import os
import os
import gcnn
# extracted images
DATA_DIR = 'data/camelyonpatch'
# load only 1 image per iteration
LOADER_BS = 1
NN_OUTPUT_SIZE = 2
PCAM_TRAIN_MEAN = np.array([0.70075643, 0.538358, 0.69162095])
PCAM_TRAIN_STD = np.array([0.2349793, 0.27740902, 0.21289322])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# transform to Tensor and normalize
train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(PCAM_TRAIN_MEAN, PCAM_TRAIN_STD)])
# load train images
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
train_dataset_length = len(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=LOADER_BS, shuffle=True)
# define model and send to GPU/CPU
model = gcnn.GConvNet(input_channels=3, no_classes=NN_OUTPUT_SIZE)
model.to(device)
# loss function
loss_func = nn.CrossEntropyLoss(reduction='sum')
# define an iterator over the DataLoader
dataiter = iter(train_loader)
# get first batch -> one image
x, y = dataiter.next()
# send variables to GPU
x = x.to(device)
y = y.long().to(device)
# track gradient history
with torch.set_grad_enabled(True):
    # get model output
    logits = model(x)
    # calculate loss
    loss = loss_func(logits, y)
    # get prediction
    _, preds = torch.max(logits, 1)

'''
In [9]: x.size()
Out[9]: torch.Size([1, 3, 96, 96])

In [10]: y.size()
Out[10]: torch.Size([1])

In [11]: logits.size()
Out[11]: torch.Size([1, 2])

In [12]: loss.size()
Out[12]: torch.Size([])

In [13]: x
Out[13]:
tensor([[[[ 0.4891,  0.7061,  0.3222,  ...,  0.7394,  0.5559,  0.6894],
          [-0.2786,  0.6226,  1.0565,  ...,  0.4557,  0.0552,  0.3723],
          [ 0.2221,  0.4224,  1.1400,  ...,  0.2888,  0.3389,  0.8229],
          ...,
          [ 0.4057,  0.4390,  0.8896,  ...,  1.2067,  1.2234, -0.1284],
          [ 0.4390,  0.7394,  0.6059,  ...,  0.3055,  1.2735,  0.9898],
          [ 0.3556,  0.5559,  0.8062,  ..., -1.1130, -0.1284,  1.2735]],

         [[-0.2160,  0.3353,  0.3494,  ..., -0.1171, -0.1877, -0.0040],
          [-0.9087,  0.1939,  0.8866,  ..., -0.3150, -0.6118, -0.3291],
          [-0.4988, -0.0464,  0.8583,  ..., -0.3574, -0.4139, -0.0322],
          ...,
          [ 0.0808,  0.0950,  0.4484,  ...,  0.7735,  0.8018, -0.3008],
          [ 0.1233,  0.3777,  0.2646,  ...,  0.1374,  1.0138,  0.8159],
          [ 0.0808,  0.2505,  0.4343,  ..., -0.9794, -0.0605,  1.2118]],

         [[ 0.1222,  0.7117,  0.6196,  ...,  0.5643,  0.4722,  0.7117],
          [-0.7251,  0.5643,  1.3564,  ...,  0.2880, -0.0804,  0.2880],
          [-0.1909,  0.3249,  1.4117,  ...,  0.1959,  0.1959,  0.7117],
          ...,
          [ 0.3801,  0.4538,  0.9143,  ...,  1.4485,  1.4485, -0.0435],
          [ 0.4722,  0.8038,  0.6564,  ...,  0.5459,  1.4485,  1.1722],
          [ 0.3986,  0.6196,  0.8775,  ..., -1.0567, -0.0804,  1.4485]]]])

In [14]: y
Out[14]: tensor([1])

In [15]: logits
Out[15]: tensor([[ 0.0046, -0.1337]], grad_fn=<AddmmBackward>)

In [16]: loss
Out[16]: tensor(0.7647, grad_fn=<NllLossBackward>)

In [18]: preds
Out[18]: tensor([0])

LABEL = 1
PREDICTION = 0 BECAUSE logits SHOWS HIGHER PREDICTION IN IDX 0

'''
