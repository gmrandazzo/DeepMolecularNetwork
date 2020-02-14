#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import sys
import numpy as np
from MDC import MDC
from disc import DISC
import random
from datetime import datetime
import multiprocessing
from math import ceil


def MDCTrainTestSplit(dict_target, n_objects=0):
    """
    Most descriptive compound train/test
    selection starting from a dictionary target
    """
    y = list(dict_target.values())
    keys = list(dict_target.keys())
    dmx = [[1. for j in range(len(y))] for i in range(len(y))]
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            dmx[i][j] = dmx[j][i] = np.sqrt((y[i]-y[j])**2)
    csel = MDC(dmx, n_objects)
    test_keys = []
    for indx in csel.select():
        test_keys.append(keys[indx])
    train_keys = []
    for key in dict_target.keys():
        if key not in test_keys:
            train_keys.append(key)
        else:
            continue
    return train_keys, test_keys


def DISCTrainTestSplit(dict_target, n_objects=0):
    """
    Dissimilarity compound selection
    selection starting from a dictionary target
    """
    y = list(dict_target.values())
    keys = list(dict_target.keys())
    dmx = [[1. for j in range(len(y))] for i in range(len(y))]
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            dmx[i][j] = dmx[j][i] = np.sqrt((y[i]-y[j])**2)
    csel = None
    if n_objects == 0:
        # Get the 20% of the dataset
        csel = DISC(dmx, "max", len(keys)*0.2)
    else:
        csel = DISC(dmx, "max", n_objects)
    test_keys = []
    for indx in csel.select():
        test_keys.append(keys[indx])
    train_keys = []
    for key in dict_target.keys():
        if key not in test_keys:
            train_keys.append(key)
        else:
            continue
    return train_keys, test_keys



def TrainTestSplit(keys, test_size_=0.20, random_state=None):
    """
    Split dataset in training set and test set accounting
    """
    n_objects = len(keys)
    if random_state is None:
        random.seed(datetime.now().microsecond)
    else:
        random.seed(random_state)
    nsel = int(np.ceil(n_objects*test_size_))
    test_keys = random.select(keys, nsel)
    train_keys = []
    for key in keys:
        if key in test_keys:
            continue
        else:
            train_keys.append(key)
    return train_keys, test_keys


def RepeatedKFold_(n_splits, n_repeats, dict_target, random_state=None):
    """
    Repeated K-fold cross validation method starting from a dictionary target
    """
    if random_state is None:
        random.seed(datetime.now().microsecond)
    else:
        random.seed(random_state)
    keys = list(dict_target.keys())
    indexes = [i for i in range(len(keys))]
    maxgobj = int(ceil(len(keys)/float(n_splits)))
    for r in range(n_repeats):
        random.shuffle(indexes)
        appgroup = []
        n = 0
        g = 0
        for i in range(len(keys)):
            if n == maxgobj:
                n = 0
                g += 1
                appgroup.append(g)
            else:
                appgroup.append(g)
                n += 1

        for g in range(n_splits):
            train_keys = []
            test_keys = []
            for k in range(len(keys)):
                if appgroup[k] == g:
                    test_keys.append(keys[indexes[k]])
                else:
                    train_keys.append(keys[indexes[k]])
            yield (train_keys, test_keys)


def RepeatedKFold(n_splits, n_repeats, keys, random_state=None):
    """
    Repeated K-fold cross validation method starting from a dictionary target
    """
    for r in range(n_repeats):
        if random_state is not None:
            yield KFold(n_splits, keys, random_state+r)
        else:
            yield KFold(n_splits, keys, None)

def KFold(n_splits, keys, random_state=None):
    if random_state is None:
        random.seed(datetime.now().microsecond)
    else:
        random.seed(random_state)
    indexes = [i for i in range(len(keys))]
    random.shuffle(indexes)
    appgroup = []
    maxgobj = int(ceil(len(keys)/float(n_splits)))
    n = 0
    g = 0
    for i in range(len(keys)):
        if n == maxgobj:
            n = 0
            g += 1
            appgroup.append(g)
        else:
            appgroup.append(g)
            n += 1

    for g in range(n_splits):
        train_keys = []
        test_keys = []
        for k in range(len(keys)):
            if appgroup[k] == g:
                test_keys.append(keys[indexes[k]])
            else:
                train_keys.append(keys[indexes[k]])
        yield train_keys, test_keys


def CVGroupRead(fcvgroup):
    d = {}
    g = 0
    f = open(fcvgroup, "r")
    for line in f:
        d[g] = list(str.split(line.strip(), ","))
        g += 1
    f.close()
    return d


def StaticGroupCV(d):
    """
    Static group cross validation
    """
    keys = list(d.keys())
    for i in keys:
        test_keys = d[i]
        train_keys = []
        for j in keys:
            if i == j:
                continue
            else:
                train_keys.extend(d[j])
        yield (train_keys, test_keys)


def RepeatedStratifiedCV(d, iterations=100, merge_ngroups=2):
    """
    Repeated Stratified Group Cross Validation.
    d: is a dictionary of object names. Every key is a group and every value
       of a group is a certain number of molecule name
    """
    random_state = datetime.now().microsecond
    random.seed(random_state)
    keys = list(d.keys())
    gid = [i for i in range(len(keys))]
    for it in range(iterations):
        random.shuffle(gid)
        test_keys = []
        for i in range(merge_ngroups):
            test_keys.extend(d[gid[i]])
        train_keys = []
        for i in range(merge_ngroups, len(gid)):
            train_keys.extend(d[gid[i]])
        yield (train_keys, test_keys)
