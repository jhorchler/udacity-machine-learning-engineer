from pathlib import Path
from PIL import Image
import numpy as np
import utils

# TODO: add argparse for image_root and output format

# where to store the images
image_root = Path('data/camelyonpatch')
# create directory if it does not exist
image_root.mkdir(parents=True, exist_ok=True)

# create references to HDF5 files
train_x, train_y = utils.get_matrixes(phase='train')
valid_x, valid_y = utils.get_matrixes(phase='valid')
test_x, test_y = utils.get_matrixes(phase='test')
# combine matrixes to a dataset
datasets = {
    'train': utils.PCamDataset(train_x, train_y),
    'valid': utils.PCamDataset(valid_x, valid_y),
    'test': utils.PCamDataset(test_x, test_y),
}

for phase in list(datasets.keys()):
    phase_dir = Path(f'{image_root}/{phase}')
    phase_dir.mkdir(exist_ok=True)
    # will be class 0
    normal_dir = Path(f'{phase_dir}/normal')
    normal_dir.mkdir(exist_ok=True)
    # will be class 1 -> hence positive label
    tumor_dir = Path(f'{phase_dir}/tumor')
    tumor_dir.mkdir(exist_ok=True)
    dataset_len = len(datasets[phase])
    for index in range(dataset_len):
        # get array and label
        img_array, label_array = datasets[phase][index]
        # store percentage of current file
        p = (index / float(dataset_len)) * 100
        print(f'\r{phase}: {index:7d} / {dataset_len} ({p:4.2f} %)', end = '')
        # get the label (0 or 1)
        label = label_array[0][0][0]
        # positive = tumor
        if label:
            Image.fromarray(img_array).save(tumor_dir / f'{index}.png')
        else:
            Image.fromarray(img_array).save(normal_dir / f'{index}.png')
    print()
