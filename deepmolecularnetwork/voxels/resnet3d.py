#!/usr/bin/env python3
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

"""A vanilla 3D resnet implementation.

Based on Raghavendra Kotikalapudi's 2D implementation
keras-resnet (See https://github.com/raghakot/keras-resnet.)
"""

from deepmolecularnetwork.framework import keras, layers, models, regularizers

K = keras.backend
l2 = regularizers.l2
BatchNormalization = layers.BatchNormalization
add = layers.add
Model = models.Model
Input = layers.Input
Activation = layers.Activation
Dense = layers.Dense
Flatten = layers.Flatten
Conv3D = layers.Conv3D
AveragePooling3D = layers.AveragePooling3D
MaxPooling3D = layers.MaxPooling3D

DIM1_AXIS = 1
DIM2_AXIS = 2
DIM3_AXIS = 3
CHANNEL_AXIS = 4


def _bn_relu(input_tensor):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input_tensor)
    return Activation("relu")(norm)


def _conv_bn_relu_3d(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input_tensor):
        conv = Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
        )(input_tensor)
        return _bn_relu(conv)

    return f


_conv_bn_relu3D = _conv_bn_relu_3d


def _bn_relu_conv3d(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input_tensor):
        activation = _bn_relu(input_tensor)
        return Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
        )(activation)

    return f


def _shortcut3d(input_tensor, residual):
    """3D shortcut to match input and residual and merges them with "sum"."""
    in_shape = getattr(input_tensor, "_keras_shape", input_tensor.shape)
    res_shape = getattr(residual, "_keras_shape", residual.shape)
    stride_dim1 = int(in_shape[DIM1_AXIS]) // int(res_shape[DIM1_AXIS])
    stride_dim2 = int(in_shape[DIM2_AXIS]) // int(res_shape[DIM2_AXIS])
    stride_dim3 = int(in_shape[DIM3_AXIS]) // int(res_shape[DIM3_AXIS])
    equal_channels = res_shape[CHANNEL_AXIS] == in_shape[CHANNEL_AXIS]

    shortcut = input_tensor
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Conv3D(
            filters=int(res_shape[CHANNEL_AXIS]),
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_initializer="he_normal",
            padding="valid",
            kernel_regularizer=l2(1e-4),
        )(input_tensor)
    return add([shortcut, residual])


def _residual_block3d(
    block_function, filters, kernel_regularizer, repetitions, is_first_layer=False
):
    def f(input_tensor):
        x = input_tensor
        for i in range(repetitions):
            strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2, 2)
            x = block_function(
                filters=filters,
                strides=strides,
                kernel_regularizer=kernel_regularizer,
                is_first_block_of_first_layer=(is_first_layer and i == 0),
            )(x)
        return x

    return f


def basic_block(
    filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4), is_first_block_of_first_layer=False
):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""

    def f(input_tensor):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(
                filters=filters,
                kernel_size=(3, 3, 3),
                strides=strides,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=kernel_regularizer,
            )(input_tensor)
        else:
            conv1 = _bn_relu_conv3d(
                filters=filters,
                kernel_size=(3, 3, 3),
                strides=strides,
                kernel_regularizer=kernel_regularizer,
            )(input_tensor)

        residual = _bn_relu_conv3d(
            filters=filters, kernel_size=(3, 3, 3), kernel_regularizer=kernel_regularizer
        )(conv1)
        return _shortcut3d(input_tensor, residual)

    return f


def bottleneck(
    filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4), is_first_block_of_first_layer=False
):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""

    def f(input_tensor):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv3D(
                filters=filters,
                kernel_size=(1, 1, 1),
                strides=strides,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=kernel_regularizer,
            )(input_tensor)
        else:
            conv_1_1 = _bn_relu_conv3d(
                filters=filters,
                kernel_size=(1, 1, 1),
                strides=strides,
                kernel_regularizer=kernel_regularizer,
            )(input_tensor)

        conv_3_3 = _bn_relu_conv3d(
            filters=filters, kernel_size=(3, 3, 3), kernel_regularizer=kernel_regularizer
        )(conv_1_1)
        residual = _bn_relu_conv3d(
            filters=filters * 4, kernel_size=(1, 1, 1), kernel_regularizer=kernel_regularizer
        )(conv_3_3)

        return _shortcut3d(input_tensor, residual)

    return f


def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == "channels_last":
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def _get_block(identifier):
    if isinstance(identifier, str):
        res = globals().get(identifier)
        if not res:
            raise ValueError(f"Invalid {identifier}")
        return res
    return identifier


class Resnet3DBuilder:
    """ResNet3D."""

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, reg_factor):
        """Instantiate a vanilla ResNet3D keras model.

        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use {'basic_block', 'bottlenack_block'}
            repetitions: Repetitions of unit blocks
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """
        _handle_data_format()
        if len(input_shape) != 4:
            raise ValueError(
                "Input shape should be a tuple "
                "(conv_dim1, conv_dim2, conv_dim3, channels) "
                "for tensorflow as backend or "
                "(channels, conv_dim1, conv_dim2, conv_dim3) "
                "for theano as backend"
            )

        block_fn = _get_block(block_fn)
        input_tensor = Input(shape=input_shape)
        # first conv
        conv1 = _conv_bn_relu_3d(
            filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2), kernel_regularizer=l2(reg_factor)
        )(input_tensor)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(conv1)

        # repeat blocks
        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block3d(
                block_fn,
                filters=filters,
                kernel_regularizer=l2(reg_factor),
                repetitions=r,
                is_first_layer=(i == 0),
            )(block)
            filters *= 2

        # last activation
        block_output = _bn_relu(block)

        # average poll and classification
        b_shape = getattr(block, "_keras_shape", block.shape)
        pool2 = AveragePooling3D(
            pool_size=(
                int(b_shape[DIM1_AXIS]),
                int(b_shape[DIM2_AXIS]),
                int(b_shape[DIM3_AXIS]),
            ),
            strides=(1, 1, 1),
        )(block_output)
        flatten1 = Flatten()(pool2)
        dense = Dense(
            units=num_outputs, kernel_initializer="he_normal", kernel_regularizer=l2(reg_factor)
        )(flatten1)

        model = Model(inputs=input_tensor, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 18."""
        return Resnet3DBuilder.build(
            input_shape, num_outputs, basic_block, [2, 2, 2, 2], reg_factor=reg_factor
        )

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 34."""
        return Resnet3DBuilder.build(
            input_shape, num_outputs, basic_block, [3, 4, 6, 3], reg_factor=reg_factor
        )

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 50."""
        return Resnet3DBuilder.build(
            input_shape, num_outputs, bottleneck, [3, 4, 6, 3], reg_factor=reg_factor
        )

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 101."""
        return Resnet3DBuilder.build(
            input_shape, num_outputs, bottleneck, [3, 4, 23, 3], reg_factor=reg_factor
        )

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 152."""
        return Resnet3DBuilder.build(
            input_shape, num_outputs, bottleneck, [3, 8, 36, 3], reg_factor=reg_factor
        )
