import numpy as np

# according to source
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/torchvision_models.py#L61
# std and mean are [0.5, 0.5, 0.5]
INCEPTION_STD = np.array([0.5, 0.5, 0.5])
INCEPTION_MEAN = np.array([0.5, 0.5, 0.5])

# as mentioned above the image size is 299x299
INCEPTION_INPUT_SHAPE = 299

# number of classes (normal or tumor)
OUTPUT_SIZE = 2

# define parameters for data loaders
LOADER_PARAMETERS = {'batch_size': 200, 'shuffle': True, 'num_workers': 3}

# set training parameters
NUM_EPOCHS = 24
LEARNING_RATE = 0.1
MOMENTUM = 0.9
EPSILON = 1.0
WEIGHT_DECAY = 0.9
NUM_EPOCH_PER_DECAY = 4
GAMMA = 0.16
