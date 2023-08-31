#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import numpy as np


def genMask(y_true, mask_value=-9999.):
    """
    Generate a mask to be used in custom loss function
    with missing values in targets
    """
    mask = []
    for y in y_true:
        if int(abs(y-mask_value)) == 0:
            mask.append(0.)
        else:
            mask.append(1.)
    return mask


def MSE(y_true, y_pred, mask_value=-9999.):
    """
    Calculate the MSE
    """
    try:
        mseavg = 0.
        rows = len(y_true)
        cols = len(y_true[0])
        for j in range(cols):
            yt = []
            yp = []
            for i in range(rows):
                yt.append(y_true[i][j])
                yp.append(y_pred[i][j])
            w = np.array(genMask(yt, mask_value))
            a = np.array(yt)*w
            b = np.array(yp)*w
            mseavg += np.square(a - b).mean(axis=0)
        return mseavg/float(cols)
    except TypeError:
        w = np.array(genMask(y_true, mask_value))
        a = np.array(yt)*w
        b = np.array(yp)*w
        return np.square(a - b).mean(axis=0)


def MAE(y_true, y_pred, mask_value=-9999.):
    try:
        maeavg = 0.
        rows = len(y_true)
        cols = len(y_true[0])
        for j in range(cols):
            yt = []
            yp = []
            for i in range(rows):
                yt.append(y_true[i][j])
                yp.append(y_pred[i][j])
            w = np.array(genMask(yt, mask_value))
            a = np.array(yt)*w
            b = np.array(yp)*w
            maeavg += np.abs(a - b).mean(axis=0)
        return maeavg/float(cols)
    except TypeError:
        w = genMask(y_true, mask_value)
        a = np.array(yt)*w
        b = np.array(yp)*w
        return np.abs(a - b).mean(axis=0)


def RSQ(y_true, y_pred, mask_value=-9999.):
    try:
        rsqavg = 0.
        rows = len(y_true)
        cols = len(y_true[0])
        for j in range(cols):
            yt = []
            yp = []
            for i in range(rows):
                yt.append(y_true[i][j])
                yp.append(y_pred[i][j])
            w = np.array(genMask(yt))
            yt = np.array(yt)
            yp = np.array(yp)
            rss = np.sum(np.square(yt*w-yp*w))
            tss = np.sum(np.square(yt*w-np.mean(yt*w)))
            rsqavg += 1. - (rss/tss)
        return rsqavg/float(cols)
    except TypeError:
        w = genMask(y_true)
        rss = np.sum(np.square(y_true*w-y_pred*w))
        tss = np.sum(np.square(y_true*w-np.mean(y_true*w)))
        return 1. - (rss/tss)


def LOGMAE(y_true, y_pred, mask_value=-9999.):
    """
    Log MAE
    """
    return np.log(MAE(np.array(y_true),
                      np.array(y_pred),
                      mask_value))
