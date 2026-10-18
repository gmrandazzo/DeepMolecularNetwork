#!/usr/bin/env python3
# (c) 2018-2023 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

from deepmolecularnetwork.framework import Activation, BatchNormalization, Dense, l2


def dnn_resnet_layer(inputs, nunits=100, activation="relu", batch_normalization=True):
    """DNN-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        nunits (int): number of units
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization

    # Returns
        x (tensor): tensor as input to the next layer
    """
    dnn = Dense(
        nunits,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=l2(1e-4),
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    )

    x = dnn(inputs)
    if batch_normalization:
        x = BatchNormalization()(x)

    if activation is not None:
        x = Activation(activation)(x)

    return x
