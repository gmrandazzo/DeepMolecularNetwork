#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import tensorflow
if int(tensorflow.__version__[0]) > 1:
    from tensorflow.keras.layers import Dense, BatchNormalization, Activation
    from tensorflow.keras.layers import Input, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
else:
    from keras.layers import Dense, BatchNormalization, Activation
    from keras.layers import Input, Flatten
    from keras.optimizers import Adam
    from keras.regularizers import l2
    from keras import backend as K
    from keras.models import Model

def dnn_resnet_layer(inputs,
                     nunits=100,
                     activation='relu',
                     batch_normalization=True):
    """DNN-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        nunits (int): number of units
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization

    # Returns
        x (tensor): tensor as input to the next layer
    """
    dnn = Dense(nunits,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=l2(1e-4),
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None)

    x = dnn(inputs)
    if batch_normalization:
        x = BatchNormalization()(x)

    if activation is not None:
        x = Activation(activation)(x)

    return x
