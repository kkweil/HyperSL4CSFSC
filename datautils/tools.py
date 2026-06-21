import os
import random
from abc import ABC

import numpy as np

import math




def split_train_test(data, label, patch_size, ratio, augment_times=1):
    assert patch_size % 2 == 1, 'The window size must be odd.'
    R = int(patch_size / 2)
    label = np.pad(label, R, mode='constant').astype('int')
    data = np.pad(data, (R, R), mode='reflect')[:, :, R:-R]
    coordinate = {i: np.argwhere(label == i + 1) for i in range(0, label.max())}
    for _, v in coordinate.items():
        np.random.shuffle(v)

    if isinstance(ratio, int):
        print(f"Sample the training dataset {ratio} points per class.")
        coordinate_train = []
        coordinate_test = []
        for k, v in coordinate.items():
            if len(v) > 2 * ratio:
                
                coordinate_train.append(v[:ratio].tolist())
                coordinate_test.append(v[ratio:].tolist())
            else:
                coordinate_train.append(v[:ratio].tolist())
                coordinate_test.append(v[int(len(v) / 2):].tolist())

        if augment_times > 1:
            coordinate_train = [sub * augment_times for sub in coordinate_train]
        
        coordinate_train = np.asarray([i for list_ in coordinate_train for i in list_])
        coordinate_test = np.asarray([i for list_ in coordinate_test for i in list_])

        x_train = np.asarray([data[coordinate_train[i][0] - R: coordinate_train[i][0] + R + 1,
                              coordinate_train[i][1] - R: coordinate_train[i][1] + R + 1, :]
                              for i in range(len(coordinate_train))])
        y_train = np.asarray([label[coord[0], coord[1]] - 1 for coord in coordinate_train])

        x_test = np.asarray([data[coordinate_test[i][0] - R: coordinate_test[i][0] + R + 1,
                             coordinate_test[i][1] - R: coordinate_test[i][1] + R + 1, :]
                             for i in range(len(coordinate_test))])
        y_test = np.asarray([label[coord[0], coord[1]] - 1 for coord in coordinate_test])

        return x_train, x_test, y_train, y_test, coordinate_train, coordinate_test

    elif isinstance(ratio, float):
        print(f"Sample the training dataset at a scale of {ratio}")
        coordinate_train = [v[:math.ceil(len(v) * ratio)].tolist() for k, v in coordinate.items()]
        coordinate_train = np.asarray([i for list_ in coordinate_train for i in list_])

        coordinate_test = [v[math.ceil(len(v) * ratio):].tolist() for k, v in coordinate.items()]
        coordinate_test = np.asarray([i for list_ in coordinate_test for i in list_])

        x_train = np.asarray([data[coordinate_train[i][0] - R: coordinate_train[i][0] + R + 1,
                              coordinate_train[i][1] - R: coordinate_train[i][1] + R + 1, :]
                              for i in range(len(coordinate_train))])
        y_train = np.asarray([label[coord[0], coord[1]] - 1 for coord in coordinate_train])

        x_test = np.asarray([data[coordinate_test[i][0] - R: coordinate_test[i][0] + R + 1,
                             coordinate_test[i][1] - R: coordinate_test[i][1] + R + 1, :]
                             for i in range(len(coordinate_test))])
        y_test = np.asarray([label[coord[0], coord[1]] - 1 for coord in coordinate_test])

        return x_train, x_test, y_train, y_test, coordinate_train, coordinate_test

    else:
        raise ValueError('Expect "ratio" for: int or float!')


