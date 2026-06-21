from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from engine.DomainAdaption import Encoder, AttentionDiscriminator, FeatureDiscriminator, Classifier, grad_reverse
from datautils.tools import split_train_test
from collections import OrderedDict


def read_paviaU(data_path, label_path, wave_range):
    data = loadmat(data_path)
    labels = loadmat(label_path)

    data = data['paviaU']
    labels = labels['paviaU_gt_7']
    bands = data.shape[-1]
    wavelength = np.linspace(wave_range[0], wave_range[1], bands)

    return data, labels, wavelength


def read_pavia(data_path, label_path, wave_range):
    data = loadmat(data_path)
    labels = loadmat(label_path)

    data = data['pavia']
    labels = labels['pavia_gt_7']
    bands = data.shape[-1]
    wavelength = np.linspace(wave_range[0], wave_range[1], bands)

    return data, labels, wavelength


class HyperDataset(Dataset):
    def __init__(self, data, label, wave):
        self.data = data
        self.label = label
        self.wave = wave

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].astype(np.float32), self.wave.astype(np.float32), self.label[idx].astype(np.longlong)


if __name__ == '__main__':
    source_data_path = './data/Pavia/paviaU.mat'
    source_label_path = './data/Pavia/paviaU_gt_7.mat'
    batch_size = 10

    source_data, source_labels, source_wavelength = read_paviaU(source_data_path, source_label_path,
                                                                wave_range=[430, 860])

    source_x_train, source_x_test, source_y_train, source_y_test, source_coordinate_train, source_coordinate_test = split_train_test(
        source_data, source_labels, patch_size=5, ratio=0.1)

    target_data_path = './data/Pavia/pavia.mat'
    target_label_path = './data/Pavia/pavia_gt_7.mat'

    target_data, target_labels, target_wavelength = read_pavia(target_data_path, target_label_path,
                                                               wave_range=[430, 860])


    a = 0
