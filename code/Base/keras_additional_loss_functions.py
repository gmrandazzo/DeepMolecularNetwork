#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

from keras import backend as K
import numpy as np


def rmse(y_true, y_pred):
    """
    Root Mean Square Error
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def score(y_true, y_pred):
    """
    Score function 1
    """
    return K.log(K.mean(K.abs(y_true - y_pred), axis=-1)+1)


def np_score(y_true, y_pred):
    """
    Score function 1
    """
    return np.log(np.mean(np.abs(y_true - y_pred), axis=0))
