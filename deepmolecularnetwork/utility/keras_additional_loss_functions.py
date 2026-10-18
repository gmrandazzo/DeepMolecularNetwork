#!/usr/bin/env python3
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

from deepmolecularnetwork.framework import ops


def rmse(y_true, y_pred):
    """
    Root Mean Square Error
    """
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true), axis=-1))


def score(y_true, y_pred):
    """
    Score function 1
    """
    return ops.log(ops.mean(ops.abs(y_true - y_pred), axis=-1) + 1)
