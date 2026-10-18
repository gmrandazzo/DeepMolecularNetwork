#!/usr/bin/env python3
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import numpy as np


def genMask(y_true, mask_value=-9999.0):
    """
    Generate a mask to be used in custom loss function
    with missing values in targets
    """
    mask = []
    for y in y_true:
        if int(abs(y - mask_value)) == 0:
            mask.append(0.0)
        else:
            mask.append(1.0)
    return mask


def MSE(y_true, y_pred, mask_value=-9999.0):
    """
    Calculate the Mean Squared Error with optional missing value masking.
    """
    try:
        mseavg = 0.0
        rows = len(y_true)
        cols = len(y_true[0])
        for j in range(cols):
            yt = np.array([y_true[i][j] for i in range(rows)])
            yp = np.array([y_pred[i][j] for i in range(rows)])
            valid = (yt != mask_value) & (~np.isnan(yt))
            if np.any(valid):
                mseavg += np.mean(np.square(yt[valid] - yp[valid]))
        return mseavg / float(cols)
    except (TypeError, IndexError):
        yt = np.array(y_true)
        yp = np.array(y_pred)
        valid = (yt != mask_value) & (~np.isnan(yt))
        if np.any(valid):
            return float(np.mean(np.square(yt[valid] - yp[valid])))
        return 0.0


def MAE(y_true, y_pred, mask_value=-9999.0):
    """
    Calculate the Mean Absolute Error with optional missing value masking.
    """
    try:
        maeavg = 0.0
        rows = len(y_true)
        cols = len(y_true[0])
        for j in range(cols):
            yt = np.array([y_true[i][j] for i in range(rows)])
            yp = np.array([y_pred[i][j] for i in range(rows)])
            valid = (yt != mask_value) & (~np.isnan(yt))
            if np.any(valid):
                maeavg += np.mean(np.abs(yt[valid] - yp[valid]))
        return maeavg / float(cols)
    except (TypeError, IndexError):
        yt = np.array(y_true)
        yp = np.array(y_pred)
        valid = (yt != mask_value) & (~np.isnan(yt))
        if np.any(valid):
            return float(np.mean(np.abs(yt[valid] - yp[valid])))
        return 0.0


def RSQ(y_true, y_pred, mask_value=-9999.0):
    """
    Calculate the R-squared (R2 / Q2) with optional missing value masking.
    """
    try:
        rsqavg = 0.0
        rows = len(y_true)
        cols = len(y_true[0])
        for j in range(cols):
            yt = np.array([y_true[i][j] for i in range(rows)])
            yp = np.array([y_pred[i][j] for i in range(rows)])
            valid = (yt != mask_value) & (~np.isnan(yt))
            if np.any(valid):
                v_yt = yt[valid]
                v_yp = yp[valid]
                rss = np.sum(np.square(v_yt - v_yp))
                tss = np.sum(np.square(v_yt - np.mean(v_yt)))
                if tss != 0:
                    rsqavg += 1.0 - (rss / tss)
        return rsqavg / float(cols)
    except (TypeError, IndexError):
        yt = np.array(y_true)
        yp = np.array(y_pred)
        valid = (yt != mask_value) & (~np.isnan(yt))
        if np.any(valid):
            v_yt = yt[valid]
            v_yp = yp[valid]
            rss = np.sum(np.square(v_yt - v_yp))
            tss = np.sum(np.square(v_yt - np.mean(v_yt)))
            if tss != 0:
                return float(1.0 - (rss / tss))
        return 0.0


def LOGMAE(y_true, y_pred, mask_value=-9999.0):
    """
    Log MAE
    """
    return np.log(MAE(np.array(y_true), np.array(y_pred), mask_value))
