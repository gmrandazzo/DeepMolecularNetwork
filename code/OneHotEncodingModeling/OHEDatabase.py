#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details
import datetime
import os
import numpy as np
from pathlib import Path
import random


def ReadOneHotEncodingCSV(ohe_csv):
    f = open(ohe_csv, "r")
    m = []
    x = []
    for line in f:
        if "END" in line:
            m.append(x.copy())
            del x[:]
        else:
            x.append(str.split(line.strip(), ","))
    f.close()
    return np.array(m)


def LoadOneHotEncodingCSVDir(ohe_dir):
    p = Path(ohe_dir).glob('**/*')
    X = {}
    for x in p:
        if x.is_file() and ".csv" in str(x):
            key = x.resolve().stem.split(".")[0]
            # Each molecule can have multiple smiles structure
            # representations
            X[key] = ReadOneHotEncodingCSV(x)
        else:
            continue
    # rows = len(list(X.values())[0][0])
    # cols = len(list(X.values())[0][0][0])
    input_shape = list(X.values())[0].shape[1:]
    return X, input_shape


class OHEDatabase(object):
    def __init__(self):
        self.X = None
        self.input_shape = None
        self.ninstances = None
        return

    def saveOHEdb(self, dbpath):
        """
        Save OHE db into a numpy format
        """
        if Path(dbpath).is_dir() is False:
            os.makedirs(dbpath, exist_ok=True)

        for key in self.X.keys():
            fdb = "%s/%s" % (dbpath, key)
            np.save(fdb, self.X[key])

    def loadOHEdb(self, dbpath):
        """
        Load OHE db from a numpy format
        """
        self.X = {}
        for p in Path(dbpath).iterdir():
            if p.is_file() and ".npy" in str(p):
                name = p.resolve().stem.split(".")[0]
                print("Loading %s" % (name))
                arr = np.load(str(p),
                              mmap_mode=None)
                self.X[name] = arr
            else:
                continue
        self.input_shape = list(self.X.values())[0].shape[1:]
        self.ninstances = len(self.X.keys())

    def loadOHEdbFromCSVDir(self, ohe_dir):
        self.X, self.input_shape = LoadOneHotEncodingCSVDir(ohe_dir)

    def getRandomSample(self, a=None):
        sub_X = {}
        if str(type(a)) == "<type 'list'>":
            for name in a:
                if name in self.X.keys():
                    sub_X[name] = self.X[name]
                else:
                    continue
        elif str(type(a)) == "<type 'int'>" or str(type(a)) == "<type 'float'>":
            keys = list(self.X.keys())
            random.seed(datetime.now().microsecond)
            if str(type(a)) == "<type 'int'>":
                sub_keys = random.sample(keys, a)
            else:
                n = self.ninstances * a
                n = int(n)
                sub_keys = random.sample(keys, n)
            for key in sub_keys:
                sub_X[key] = self.X[key]
        else:
            print("Type %s not supported" % (str(type(a))))

        return sub_X
