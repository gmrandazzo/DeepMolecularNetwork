#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def exp_pred_plot(y_true, y_pred):
    x = np.array(y_true)
    y = np.array(y_pred)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

    predict_y = intercept + slope * x

    plt.plot(x,y,'go')
    plt.plot(x, predict_y, 'k-')
    plt.legend(('data', 'line-regression r={}'.format(r_value)), 'best')
    plt.autoscale(True)
    plt.grid(True)
    plt.show()
