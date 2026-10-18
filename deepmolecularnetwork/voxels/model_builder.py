"""
model_builder
(c) 2018-2023 gmrandazzo@gmail.com
This file is part of DeepMolecularNetwork.
You can use,modify, and distribute it under
the terms of the GNU General Public Licenze, version 3.
See the file LICENSE for details
"""

from deepmolecularnetwork.framework import (
    K,
    Model,
    Sequential,
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    LeakyReLU,
    AveragePooling3D,
    Conv3D,
    Flatten,
    MaxPooling3D,
    Concatenate,
    add,
    Layer,
    l2,
    ops,
    optimizers,
)
from deepmolecularnetwork.voxels.resnet3d import Resnet3DBuilder


# Loss function
def rmse(y_true, y_pred):
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true), axis=-1))


def rsquared(y_true, y_pred):
    ss_res = ops.sum(ops.square(y_true - y_pred), axis=-1)
    ss_tot = ops.sum(ops.square(y_true - ops.mean(y_true, axis=-1)), axis=-1)
    return 1 - ss_res / (ss_tot + K.epsilon())


def resnet3d_model(input_shape):
    K.set_image_data_format("channels_first")
    model = Resnet3DBuilder.build_resnet_18(input_shape, 1)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=10**-5), loss="mae", metrics=["mse", "mae"]
    )
    return model


resnet3DModel = resnet3d_model


def model_scirep(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    """
    Architecture from
    https://doi.org/10.1038/s41598-017-17299-w
    """
    model = Sequential()
    if "last" in conv3d_chtype:
        model.add(
            Conv3D(
                nfilters,
                kernel_size=(3, 3, 3),
                input_shape=input_shape,
                data_format="channels_last",
            )
        )
    else:
        model.add(
            Conv3D(
                nfilters,
                kernel_size=(3, 3, 3),
                input_shape=input_shape,
                data_format="channels_first",
            )
        )
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), data_format="channels_first"))

    model.add(
        Conv3D(
            nfilters, kernel_size=(3, 3, 3), input_shape=input_shape, data_format="channels_first"
        )
    )
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), data_format="channels_first"))

    model.add(
        Conv3D(
            nfilters, kernel_size=(3, 3, 3), input_shape=input_shape, data_format="channels_first"
        )
    )
    model.add(LeakyReLU())

    model.add(Flatten())
    for _ in range(ndense_layers):
        model.add(Dense(nunits))
        model.add(LeakyReLU())
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mse", "mae"])
    return model


def resnet_unit(x, filters, pool=False):
    res = x
    if pool:
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        res = Conv3D(filters=filters, kernel_size=[1, 1, 1], strides=(2, 2, 2), padding="same")(res)
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    out = Conv3D(filters=filters, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same")(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv3D(filters=filters, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same")(out)

    out = add([res, out])
    return out


ResNetUnit = resnet_unit


def resnet_model(input_shape):
    voxels = Input(input_shape)
    net = Conv3D(filters=2, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same")(voxels)
    net = resnet_unit(net, 2)
    net = resnet_unit(net, 4, pool=True)
    net = resnet_unit(net, 4)

    net = resnet_unit(net, 8, pool=True)
    net = resnet_unit(net, 8)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = AveragePooling3D(pool_size=(2, 2, 2))(net)
    net = Flatten()(net)
    net = Dense(32)(net)
    net = Activation("relu")(net)
    net = Dense(1)(net)

    model = Model(inputs=voxels, outputs=net)
    model.compile(optimizer=optimizers.Adam(), loss="mae", metrics=["mse", "mae"])

    return model


def build_shapenet10(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    """
    Build ShapeNet10 model.
    Parameter ndense_layers is accepted for signature compatibility.
    """
    _ = ndense_layers
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format("channels_last")
    else:
        K.set_image_data_format("channels_first")

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))

    model.add(
        Conv3D(nfilters, kernel_size=(5, 5, 5), kernel_regularizer=l2(0.001), strides=(2, 2, 2))
    )
    model.add(LeakyReLU())

    model.add(Conv3D(nfilters, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mse", "mae", rmse])
    return model


def build_fcn_model(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    """
    Personal model.
    Parameters ndense_layers, nunits, nfilters are accepted for signature compatibility.
    """
    _ = (ndense_layers, nunits, nfilters)
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format("channels_last")
    else:
        K.set_image_data_format("channels_first")

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(LeakyReLU())
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mse", "mae"])
    return model


class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[1], self.output_dim),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        y = K.dot(x, self.kernel)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def build_custom_model(conv3d_chtype, input_shape, nunits, nfilters):
    """
    Personal model
    """
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format("channels_last")
    else:
        K.set_image_data_format("channels_first")

    voxels = Input(input_shape)
    net = BatchNormalization()(voxels)
    net = Conv3D(filters=nfilters, kernel_size=[3, 3, 3])(net)
    net = MaxPooling3D(pool_size=(2, 2, 2))(net)
    net = LeakyReLU()(net)
    net = Conv3D(filters=nfilters, kernel_size=[3, 3, 3])(net)
    net = MaxPooling3D(pool_size=(2, 2, 2))(net)
    net = LeakyReLU()(net)
    net = Flatten()(net)
    # what is net?
    # print(net)
    net = Dense(nunits)(net)
    net = LeakyReLU()(net)
    net = Dense(1)(net)
    model = Model(inputs=voxels, outputs=net)
    model.compile(optimizer=optimizers.Adam(), loss="mae", metrics=["mse", "mae"])
    return model


def build_model_(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    """
    Personal model.
    Parameter ndense_layers is accepted for signature compatibility.
    """
    _ = ndense_layers
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format("channels_last")
    else:
        K.set_image_data_format("channels_first")

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))

    model.add(Conv3D(nfilters, kernel_size=(2, 2, 2), kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters * 2, kernel_size=(2, 2, 2), kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())

    model.add(Conv3D(nfilters * 4, kernel_size=(2, 2, 2), kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters * 8, kernel_size=(2, 2, 2), kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Flatten())

    model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mse", "mae"])

    return model


def build_model_garbage(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    """
    Personal model.
    Parameter ndense_layers is accepted for signature compatibility.
    """
    _ = ndense_layers
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format("channels_last")
    else:
        K.set_image_data_format("channels_first")

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))

    model.add(Conv3D(nfilters, kernel_size=(2, 2, 2), strides=(1, 1, 1)))
    model.add(Activation("relu"))

    model.add(Conv3D(nfilters, kernel_size=(2, 2, 2), strides=(1, 1, 1)))
    model.add(Activation("relu"))

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters, kernel_size=(2, 2, 2), strides=(1, 1, 1)))
    model.add(Activation("relu"))

    model.add(Conv3D(nfilters, kernel_size=(2, 2, 2), strides=(1, 1, 1)))
    model.add(Activation("relu"))

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters, kernel_size=(2, 2, 2), strides=(1, 1, 1)))
    model.add(Activation("relu"))

    model.add(Flatten())

    model.add(Dense(nunits))
    model.add(Activation("relu"))

    model.add(Dense(nunits))
    model.add(Activation("relu"))

    model.add(Dense(nunits))
    model.add(Activation("relu"))

    model.add(Dense(1))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.00001), loss="mse", metrics=["mse", "mae"]
    )

    return model


def build_model_okg25s25(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    """
    Personal model.
    Parameter ndense_layers is accepted for signature compatibility.
    """
    _ = ndense_layers
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format("channels_last")
    else:
        K.set_image_data_format("channels_first")

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv3D(nfilters, kernel_size=(2, 2, 2), strides=(1, 1, 1)))
    model.add(LeakyReLU())

    model.add(AveragePooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(nfilters * 2, kernel_size=(2, 2, 2), strides=(1, 1, 1)))
    model.add(LeakyReLU())

    model.add(AveragePooling3D(pool_size=(2, 2, 2)))

    model.add(Flatten())

    model.add(Dense(nunits))
    model.add(LeakyReLU())

    model.add(Dense(nunits))
    model.add(LeakyReLU())

    model.add(Dense(nunits))
    model.add(LeakyReLU())

    model.add(Dense(1))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.00001), loss="mse", metrics=["mse", "mae"]
    )

    return model


def build_model_not_well_working(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    """
    Personal model.
    Parameter ndense_layers is accepted for signature compatibility.
    """
    _ = ndense_layers
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format("channels_last")
    else:
        K.set_image_data_format("channels_first")

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))

    model.add(Conv3D(nfilters, kernel_size=(4, 4, 4), strides=(1, 1, 1)))
    model.add(LeakyReLU())

    model.add(Conv3D(nfilters, kernel_size=(3, 3, 3), strides=(1, 1, 1)))
    model.add(LeakyReLU())

    model.add(Conv3D(nfilters * 2, kernel_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(LeakyReLU())

    model.add(Conv3D(nfilters * 2, kernel_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(LeakyReLU())

    model.add(Flatten())

    model.add(Dense(nunits))
    model.add(LeakyReLU())

    model.add(Dense(nunits))
    model.add(LeakyReLU())

    model.add(Dense(1))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.00001), loss="mse", metrics=["mse", "mae"]
    )

    return model


build_model_not_well_Working = build_model_not_well_working


def build_model(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    """
    Personal model.
    Parameter ndense_layers is accepted for signature compatibility.
    """
    _ = ndense_layers
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format("channels_last")
    else:
        K.set_image_data_format("channels_first")

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv3D(nfilters, kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters * 2, kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())

    model.add(MaxPooling3D(pool_size=(3, 3, 3)))

    model.add(Flatten())

    model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())

    model.add(Dense(int(nunits / 2), kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(Dense(1))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.00001), loss="mse", metrics=["mse", "mae"]
    )

    return model


def build_model_orig(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    """
    Personal model.
    Parameter ndense_layers is accepted for signature compatibility.
    """
    _ = ndense_layers
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format("channels_last")
    else:
        K.set_image_data_format("channels_first")

    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))

    model.add(Conv3D(nfilters, kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters * 2, kernel_size=(4, 4, 4)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(nunits))
    model.add(LeakyReLU())
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mse", "mae", rmse])
    return model


def build_2d_data_model(
    conv3d_chtype, input_shape1, input_shape2, ndense_layers=None, nunits=100, nfilters=16
):
    """
    Working with keras 2.2.4+ and Keras 3
    Personal model
    """
    _ = ndense_layers
    alpha = 0.3
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format("channels_last")
    else:
        K.set_image_data_format("channels_first")

    # Model branch 1
    in1 = Input(shape=input_shape1)
    m1 = BatchNormalization()(in1)
    m1 = Conv3D(nfilters, kernel_size=(2, 2, 2), strides=(1, 1, 1))(m1)
    m1 = LeakyReLU(alpha)(m1)
    m1 = AveragePooling3D(pool_size=(2, 2, 2))(m1)

    m1 = Conv3D(nfilters * 2, kernel_size=(2, 2, 2), strides=(1, 1, 1))(m1)
    m1 = LeakyReLU(alpha)(m1)
    m1 = AveragePooling3D(pool_size=(2, 2, 2))(m1)

    m1 = Flatten()(m1)

    # Model branch 2
    in2 = Input(shape=(input_shape2,))
    m2 = Dense(nunits)(in2)
    m2 = LeakyReLU(alpha)(m2)
    m2 = Dense(nunits)(m2)
    m2 = LeakyReLU(alpha)(m2)
    m2 = Dense(nunits)(m2)
    m2 = LeakyReLU(alpha)(m2)

    # Concatenation
    concat = Concatenate()([m1, m2])

    out = Dense(nunits)(concat)
    out = LeakyReLU(alpha)(out)

    out = Dense(nunits)(out)
    out = LeakyReLU(alpha)(out)

    out = Dense(nunits)(out)
    out = LeakyReLU(alpha)(out)

    out = Dense(nunits)(out)
    out = LeakyReLU(alpha)(out)

    out = Dense(1)(out)

    fmodel = Model([in1, in2], out)
    fmodel.compile(
        optimizer=optimizers.Adam(learning_rate=0.00001), loss="mse", metrics=["mse", "mae"]
    )
    return fmodel


def build_2d_data_model_old(
    conv3d_chtype, input_shape1, input_shape2, ndense_layers=None, nunits=100, nfilters=16
):
    """
    Working with keras 2.2.4+ and Keras 3
    Personal model
    """
    _ = ndense_layers
    alpha = 0.3
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format("channels_last")
    else:
        K.set_image_data_format("channels_first")

    # Model branch 1
    in1 = Input(shape=input_shape1)
    m1 = BatchNormalization()(in1)
    m1 = Conv3D(nfilters, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001))(m1)
    m1 = LeakyReLU(alpha)(m1)
    m1 = MaxPooling3D(pool_size=(2, 2, 2))(m1)
    m1 = Conv3D(nfilters * 2, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001))(m1)
    m1 = LeakyReLU(alpha)(m1)
    m1 = MaxPooling3D(pool_size=(3, 3, 3))(m1)
    m1 = Flatten()(m1)

    # Model branch 2
    in2 = Input(shape=(input_shape2,))
    m2 = Dense(2)(in2)
    m2 = LeakyReLU(alpha)(m2)

    # Concatenation
    concat = Concatenate()([m1, m2])

    out = Dense(nunits, kernel_regularizer=l2(0.001))(concat)
    out = LeakyReLU(alpha)(out)
    out = Dense(nunits, kernel_regularizer=l2(0.001))(out)
    out = LeakyReLU(alpha)(out)
    out = Dense(nunits, kernel_regularizer=l2(0.001))(out)
    out = LeakyReLU(alpha)(out)
    out = Dense(nunits, kernel_regularizer=l2(0.001))(out)
    out = LeakyReLU(alpha)(out)
    out = Dense(1)(out)

    fmodel = Model([in1, in2], out)
    fmodel.compile(
        optimizer=optimizers.Adam(learning_rate=0.00001), loss="mse", metrics=["mse", "mae"]
    )
    return fmodel


build_2DData_model = build_2d_data_model
build_2DData_model_old = build_2d_data_model_old
