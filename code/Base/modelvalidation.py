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


def get_random_int(from_, to_, selected_ids):
    """
    Get random int avoiding repetitions.
    If all index were selected return -1
    """
    loop = 0
    while True:
        indx = random.randint(from_, to_-1)
        if indx in selected_ids:
            if loop > 10:
                return -1
            else:
                loop += 1
        else:
            selected_ids.append(indx)
            return indx


def TrainTestKeyWorker(var_lst):
    i_from, i_to, lst, keys = var_lst
    train_keys = []
    test_keys = []
    for i in range(i_from, i_to):
        if i in lst:
           test_keys.append(keys[i])
        else:
            train_keys.append(keys[i])
    return train_keys, test_keys


def TrainTestSplit(dict_target, test_size_=0.20, randomize=False):
    """
    Split dataset in training set and test set accounting
    """
    keys = list(dict_target.keys())
    n_objects = len(keys)
    random_state = 1
    if randomize is True:
        random_state = datetime.now().microsecond
    else:
        random_state = n_objects
    random.seed(random_state)
    nsel = int(np.ceil(n_objects*test_size_))
    test_index = []
    selected_index = []
    for i in range(nsel):
        test_index.append(get_random_int(0, n_objects-1, selected_index))
        # test_index.append(random.randint(0, n_objects-1))

    runlst = []
    nths = multiprocessing.cpu_count()
    di = ceil(n_objects/nths)

    prev = 0
    for i in range(di, n_objects, di):
        runlst.append([prev, i, test_index, keys])
        prev = i
    runlst.append([prev, n_objects, test_index, keys])

    pool = multiprocessing.Pool(nths)
    results = pool.map(TrainTestKeyWorker, runlst)

    train_keys = []
    test_keys = []
    for res in results:
        train_keys.extend(res[0])
        test_keys.extend(res[1])

    """
    for i in range(n_objects):
        if i in test_index:
            test_keys.append(keys[i])
        else:
            train_keys.append(keys[i])
    """
    return train_keys, test_keys


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


def RepeatedKFold(n_splits, n_repeats, dict_target, seed_val=None):
    """
    Repeated K-fold cross validation method starting from a dictionary target
    """
    if seed_val is None:
        random.seed(datetime.now().microsecond)
    else:
        random.seed(seed_val)
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


def KFold(n_splits, dict_target, seed_val=None):
    if seed_val is None:
        random.seed(datetime.now().microsecond)
    else:
        random.seed(seed_val)
    keys = list(dict_target.keys())
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
        yield (train_keys, test_keys)


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
