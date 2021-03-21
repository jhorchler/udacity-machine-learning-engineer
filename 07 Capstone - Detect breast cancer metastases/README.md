# Machine Learning Engineer Nanodegree: Detect breast cancer metastases

This folder holds all files of the Udacity Machine Learning Engineer Nanodegree
**Detect breast cancer metastases**.

## Dataset

First of all the dataset used in the project must be downloaded from
[GoogleDrive](https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB).
A mirror is stored at
[GoogleDrive](https://drive.google.com/drive/folders/1LM3TW5nj5DD4I7ODI4d4BEF4BePEHcTZ?usp=sharing)
as well.

The datafiles must be extracted and placed in a subfolder named `data`, so that
the directory holds the files

    camelyonpatch_level_2_split_test_meta.csv
    camelyonpatch_level_2_split_test_x.h5
    camelyonpatch_level_2_split_test_y.h5
    camelyonpatch_level_2_split_train_mask.h5
    camelyonpatch_level_2_split_train_meta.csv
    camelyonpatch_level_2_split_train_x.h5
    camelyonpatch_level_2_split_train_y.h5
    camelyonpatch_level_2_split_valid_meta.csv
    camelyonpatch_level_2_split_valid_x.h5
    camelyonpatch_level_2_split_valid_y.h5

For comparison with some of the original images of the
[CAMELYON16](https://camelyon16.grand-challenge.org/) dataset these images are
needed as well in the `data` folder:

- [tumor_002.tif](https://drive.google.com/open?id=0BzsdkU4jWx9BQ0loQnhZbE9pY2s)
- [tumor_003.tif](https://drive.google.com/open?id=0BzsdkU4jWx9BdHpsTFcyeXpSc28)
- [normal_001.tif](https://drive.google.com/open?id=0BzsdkU4jWx9BLVNUUzk4dUxHWHM)
- [normal_002.tif](https://drive.google.com/open?id=0BzsdkU4jWx9BRVZqeVotM2VOaGM)

## Software

In this project the following software and libraries are used:

- Python 3.7.3
- GrouPy 0.1.2 (PyTorch implementation from https://github.com/adambielski/GrouPy)
- h5py 2.9.0
- imgaug 0.2.9
- jupyter 1.0.0
- matplotlib 3.1.0
- notebook 5.7.8
- numpy 1.16.4
- openslide-python 1.1.1 (needs openslide library installed, for example using `brew` or `macports` on a Mac)
- pandas 0.24.2
- Pillow 6.0.0
- scikit-learn 0.21.2
- scipy 1.3.0
- seaborn 0.9.0
- torch 1.1.0post2
- torchvision 0.3.0
- tqdm 4.32.2
