from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from imgaug import augmenters as iaa
from typing import Callable, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import h5py
import imgaug


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            # iaa.Sometimes(0.3, iaa.Affine(rotate=(-90, 90), mode='symmetric')),
            iaa.Fliplr(0.25),
            iaa.Flipud(0.25),
            iaa.Sometimes(then_list=[
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
            ], else_list=[
                iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
                iaa.GaussianBlur(sigma=2.0),
            ]),
            iaa.Dropout(p=(0, 0.2), per_channel=0.5),
            iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25,
                                      mode=imgaug.ALL),
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class HDF5Matrix:
    """Represents an HDF5 file.

    Example:

    ```python
      x = HDF5Matrix('input/file.hdf5', 'x')
      model.predict(x)
    ```

    Args:
      datapath (string): path to a HDF5 file
      dataset (string): name of the HDF5 dataset in the hdf5 file
      transform (callable, optional): Optional transform to be applied

    Returns:
      An array-like HDF5 dataset.
    """

    def __init__(self, datapath: str, dataset: str,
                 transform: Callable = None) -> None:
        """Creates a new HDF5Matrix"""

        # check whether the given datapath is actually a file
        p = Path(datapath)
        assert p.is_file()

        # store path to file
        self.h5file = datapath

        # store name of dataset
        self.h5dataset = dataset

        # store reference to callable
        self.transform = transform

    def __getitem__(self, idx: int) -> np.ndarray:
        """Gets idx sample from dataset."""

        # workaround to use num_workers > 1 in dataloader
        # open the file on every iteration .. bad .. ugly ..
        # we would need to build a threadsafe and/or parallel hdf5
        # installation from source
        with h5py.File(self.h5file, 'r', libver='latest',
                       swmr=True) as database:
            assert self.h5dataset in database
            table = database[self.h5dataset]
            item = table[idx]
            if self.transform:
                if type(item).__module__ == np.__name__:
                    im = Image.fromarray(item)
                    im = self.transform(im)
                    return np.array(im)
                else:
                    return self.transform(item)
            else:
                return item

    def __len__(self) -> int:
        """Gets the length of the dataset in datapath."""
        # workaround to use num_workers > 1 in dataloader
        # open the file on every iteration .. bad .. ugly ..
        # we would need to build a threadsafe and/or parallel hdf5
        # installation from source
        with h5py.File(self.h5file, 'r', libver='latest',
                       swmr=True) as database:
            assert self.h5dataset in database
            table = database[self.h5dataset]
            return table.len()

    @property
    def shape(self) -> Tuple:
        """Gets the shape tuple of the dataset dimensions."""
        # workaround to use num_workers > 1 in dataloader
        # open the file on every iteration .. bad .. ugly ..
        # we would need to build a threadsafe and/or parallel hdf5
        # installation from source
        with h5py.File(self.h5file, 'r', libver='latest',
                       swmr=True) as database:
            assert self.h5dataset in database
            table = database[self.h5dataset]
            return table.shape

    @property
    def dtype(self) -> np.dtype:
        """Gets the datatype of the dataset."""
        # workaround to use num_workers > 1 in dataloader
        # open the file on every iteration .. bad .. ugly ..
        # we would need to build a threadsafe and/or parallel hdf5
        # installation from source
        with h5py.File(self.h5file, 'r', libver='latest',
                       swmr=True) as database:
            assert self.h5dataset in database
            table = database[self.h5dataset]
            return table.dtype

    @property
    def ndim(self) -> int:
        """Gets the number of dimensions of the dataset."""
        # workaround to use num_workers > 1 in dataloader
        # open the file on every iteration .. bad .. ugly ..
        # we would need to build a threadsafe and/or parallel hdf5
        # installation from source
        with h5py.File(self.h5file, 'r', libver='latest',
                       swmr=True) as database:
            assert self.h5dataset in database
            table = database[self.h5dataset]
            return table.ndim

    @property
    def size(self) -> int:
        """Gets the number of elements in the dataset."""
        # workaround to use num_workers > 1 in dataloader
        # open the file on every iteration .. bad .. ugly ..
        # we would need to build a threadsafe and/or parallel hdf5
        # installation from source
        with h5py.File(self.h5file, 'r', libver='latest',
                       swmr=True) as database:
            assert self.h5dataset in database
            table = database[self.h5dataset]
            return table.size


class PCamDataset(Dataset):
    """Special class that bundles two HDF5 matrices together.
    It is needed later on during training.

    Args:
      datapath (string): path to a HDF5 file
      dataset (string): name of the HDF5 dataset in the hdf5 file
      transform (callable, optional): Optional transform to be applied

    Returns:
      An PCam dataset.
    """

    def __init__(self, x: HDF5Matrix, y: HDF5Matrix) -> None:
        assert len(x) == len(y), "len of both Matrices must be equal"
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple:
        return self.x[idx], self.y[idx]


def get_matrixes(phase: str, transform: transforms.Compose = None) -> Tuple:
    """Opens two HDF5Matrixes and returns them as tuple."""
    x = HDF5Matrix('data/camelyonpatch_level_2_split_{}_x.h5'.format(phase),
                   'x', transform=transform)
    y = HDF5Matrix('data/camelyonpatch_level_2_split_{}_y.h5'.format(phase),
                   'y')
    return x, y


def get_mean_std(loader: DataLoader) -> Tuple:
    """Calculates mean and standard deviation of a dataset taken from the
    given DataLoader."""
    cnt = 0
    ds_mean = torch.empty(3)
    ds_std = torch.empty(3)
    for data in loader:
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        ds_mean = (cnt * ds_mean + sum_.float()) / (cnt + nb_pixels)
        ds_std = (cnt * ds_std + sum_of_square.float()) / (cnt + nb_pixels)
        cnt += nb_pixels
    return ds_mean.numpy(), torch.sqrt(ds_std - ds_mean ** 2).numpy()
