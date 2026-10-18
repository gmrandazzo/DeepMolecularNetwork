#!/usr/bin/env python3
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

from datetime import datetime
from math import ceil
import random
import numpy as np
from deepmolecularnetwork.utility.dissimilar_compound_selection import DISC
from deepmolecularnetwork.utility.most_descriptive_compound import MDC


def mdc_train_test_split(dict_target, n_objects=0):
    """
    Most descriptive compound train/test
    selection starting from a dictionary target
    """
    y = list(dict_target.values())
    keys = list(dict_target.keys())
    dmx = [[1.0] * len(y) for _ in range(len(y))]
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            dmx[i][j] = dmx[j][i] = float(np.sqrt((y[i] - y[j]) ** 2))
    csel = MDC(dmx, n_objects)
    test_keys = [keys[indx] for indx in csel.select()]
    train_keys = [key for key in dict_target.keys() if key not in test_keys]
    return train_keys, test_keys


def disc_train_test_split(dict_target, n_objects=0):
    """
    Dissimilarity compound selection
    selection starting from a dictionary target
    """
    y = list(dict_target.values())
    keys = list(dict_target.keys())
    dmx = [[1.0] * len(y) for _ in range(len(y))]
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            dmx[i][j] = dmx[j][i] = float(np.sqrt((y[i] - y[j]) ** 2))
    target_count = len(keys) * 0.2 if n_objects == 0 else n_objects
    csel = DISC(dmx, "max", target_count)
    test_keys = [keys[indx] for indx in csel.select()]
    train_keys = [key for key in dict_target.keys() if key not in test_keys]
    return train_keys, test_keys


def train_test_split(keys_, test_size=0.20, random_state=None):
    """
    Split dataset in training set and test set accounting
    """
    keys = list(keys_)
    keys.sort()
    n_objects = len(keys)
    if random_state is None:
        random.seed(datetime.now().microsecond)
    else:
        random.seed(random_state)
    nsel = int(np.ceil(n_objects * test_size))
    test_keys = random.sample(keys, nsel)
    train_keys = [key for key in keys if key not in test_keys]
    return train_keys, test_keys


def repeated_kfold(keys, n_splits=5, n_repeats=20, random_state=None, test_size=0.2):
    """
    Repeated K-fold cross validation method starting from a dictionary target
    """
    rs = random_state if random_state is not None else 0

    for r in range(n_repeats):
        for subset, test in kfold(keys, n_splits, rs + r):
            train, val = train_test_split(subset, test_size, rs + r)
            yield train, val, test


def kfold(keys, n_splits, random_state=None):
    if random_state is None:
        random.seed(datetime.now().microsecond)
    else:
        random.seed(random_state)
    indexes = list(range(len(keys)))
    random.shuffle(indexes)
    appgroup = []
    maxgobj = int(ceil(len(keys) / float(n_splits)))
    n = 0
    g = 0
    for _ in range(len(keys)):
        if n == maxgobj:
            n = 0
            g += 1
        appgroup.append(g)
        n += 1

    for group_idx in range(n_splits):
        train_keys = []
        test_keys = []
        for k, key_idx in enumerate(indexes):
            if appgroup[k] == group_idx:
                test_keys.append(keys[key_idx])
            else:
                train_keys.append(keys[key_idx])
        yield train_keys, test_keys


def cv_group_read(fcvgroup):
    d = {}
    with open(fcvgroup) as f:
        for g, line in enumerate(f):
            d[g] = list(str.split(line.strip(), ","))
    return d


def static_group_cv(d):
    """
    Static group cross validation
    """
    keys = list(d.keys())
    for i in keys:
        test_keys = d[i]
        train_keys = []
        for j in keys:
            if i != j:
                train_keys.extend(d[j])
        yield (train_keys, test_keys)


def repeated_stratified_cv(d, iterations=100, merge_ngroups=2):
    """
    Repeated Stratified Group Cross Validation.
    d: is a dictionary of object names. Every key is a group and every value
       of a group is a certain number of molecule name
    """
    random_state = datetime.now().microsecond
    random.seed(random_state)
    gid = list(d.keys())
    for _ in range(iterations):
        random.shuffle(gid)
        test_keys = []
        for i in range(merge_ngroups):
            test_keys.extend(d[gid[i]])
        train_keys = []
        for i in range(merge_ngroups, len(gid)):
            train_keys.extend(d[gid[i]])
        yield (train_keys, test_keys)


# Backward compatible CamelCase aliases
MDCTrainTestSplit = mdc_train_test_split
DISCTrainTestSplit = disc_train_test_split
TrainTestSplit = train_test_split
RepeatedKFold = repeated_kfold
KFold = kfold
CVGroupRead = cv_group_read
StaticGroupCV = static_group_cv
RepeatedStratifiedCV = repeated_stratified_cv
