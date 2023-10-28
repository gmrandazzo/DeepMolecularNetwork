"""
model_builder
(c) 2018-2023 gmrandazzo@gmail.com
This file is part of DeepMolecularNetwork.
You can use,modify, and distribute it under
the terms of the GNU General Public Licenze, version 3.
See the file LICENSE for details
"""

import tensorflow as tf
if int(tf.__version__[0]) > 1:
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import(
        ModelCheckpoint,
        TensorBoard
    )

    from tensorflow.keras.models import(
        Model,
        Sequential
    )

    from tensorflow.keras.layers import(
        add,
        Input,
        Dense,
        Dropout,
        BatchNormalization,
        Activation,
        LeakyReLU,
        Activation,
        AveragePooling3D,
        BatchNormalization,
        Conv3D,
        Cropping3D,
        Flatten,
        MaxPooling3D,
        Conv1D,
        MaxPooling1D,
        Concatenate
    )
    from tensorflow.keras import optimizers
    from tensorflow.keras.layers import Layer
else:
    from keras import backend as K
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import TensorBoard
    import keras
    from keras import backend as K
    from keras.models import(
        Model,
        Input,
        Sequential
    )
    from keras.layers import(
        Dense,
        Dropout,
        BatchNormalization,
        Activation,
        LeakyReLU,
        Activation,
        AveragePooling3D,
        BatchNormalization,
        Conv3D,
        Cropping3D,
        Flatten,
        MaxPooling3D,
        Conv1D,
        MaxPooling1D,
        Concatenate
    )
    from keras.regularizers import l2
    from keras import optimizers
    from keras.engine.topology import Layer

from resnet3d import *


# Loss function
"""
from sklearn.cross_decomposition import PLSRegression
def custom_rmse(y_true, x_features):
    pls2 = PLSRegression(n_components=3)
    pls2.fit(x, y_true)
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
"""

def rmse(y_true, y_pred):
    print(y_pred.shape)
    print(y_true.shape)
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def rsquared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred), axis=-1)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis=-1)), axis=-1)
    return (1 - SS_res/(SS_tot + K.epsilon()))


def resnet3DModel(input_shape):
    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_18(input_shape, 1)
    model.compile(optimizer=optimizers.Adam(lr=10**-5), loss='mae', metrics=['mse', 'mae'])


def model_scirep(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    """
    Architecture from
    https://doi.org/10.1038/s41598-017-17299-w
    """
    model = Sequential()
    if "last" in conv3d_chtype:
        model.add(Conv3D(nfilters,
                         kernel_size=(3, 3, 3),
                         input_shape=input_shape,
                         data_format="channels_last"))
    else:
        model.add(Conv3D(nfilters,
                         kernel_size=(3, 3, 3),
                         input_shape=input_shape,
                         data_format="channels_first"))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
                           data_format="channels_first"))

    model.add(Conv3D(nfilters, kernel_size=(3, 3, 3),
                     input_shape=input_shape,
                     data_format="channels_first"))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
                           data_format="channels_first"))

    model.add(Conv3D(nfilters, kernel_size=(3, 3, 3),
                     input_shape=input_shape,
                     data_format="channels_first"))
    model.add(LeakyReLU())

    model.add(Flatten())
    for i in range(ndense_layers):
        model.add(Dense(nunits))
        model.add(LeakyReLU())
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mse',
                                                                    'mae'])
    return model


def ResNetUnit(x, filters, pool=False):
    res = x
    if pool:
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        res = Conv3D(filters=filters,
                     kernel_size=[1, 1, 1],
                     strides=(2, 2, 2),
                     padding="same")(res)
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    out = Conv3D(filters=filters,
                 kernel_size=[3, 3, 3],
                 strides=[1, 1, 1],
                 padding="same")(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv3D(filters=filters,
                 kernel_size=[3, 3, 3],
                 strides=[1, 1, 1], padding="same")(out)

    out = keras.layers.add([res, out])
    return out


def ResNetModel(input_shape):
    voxels = Input(input_shape)
    net = Conv3D(filters=2,
                 kernel_size=[3, 3, 3],
                 strides=[1, 1, 1],
                 padding="same")(voxels)
    net = ResNetUnit(net, 2)
    net = ResNetUnit(net, 4, pool=True)
    net = ResNetUnit(net, 4)

    net = ResNetUnit(net, 8, pool=True)
    net = ResNetUnit(net, 8)
    """
    net = self.ResNetUnit(net,256,pool=True)
    net = self.ResNetUnit(net,256)
    """
    """
    net = self.ResNetUnit(net,256)
    net = self.ResNetUnit(net,128,pool=True)
    net = self.ResNetUnit(net,128)
    net = self.ResNetUnit(net,128)

    net = self.ResNetUnit(net,64,pool=True)
    net = self.ResNetUnit(net,64)
    net = self.ResNetUnit(net,64)

    net = self.ResNetUnit(net, 32,pool=True)
    net = self.ResNetUnit(net, 32)
    net = self.ResNetUnit(net, 32)
    """

    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    # net = Dropout(0.25)(net)

    net = AveragePooling3D(pool_size=(2, 2, 2))(net)
    net = Flatten()(net)
    net = Dense(32)(net)
    net = Activation("relu")(net)
    net = Dense(1)(net)

    model = Model(inputs=voxels, outputs=net)
    model.compile(optimizer=optimizers.Adam(), loss='mae', metrics=['mse',
                                                                    'mae'])

    return model

def build_shapenet10(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))

    model.add(Conv3D(nfilters,
                    kernel_size=(5, 5, 5),
                    kernel_regularizer=l2(0.001),
                    strides=(2, 2, 2)))
    model.add(LeakyReLU())
    # model.add(Dropout(0.2))

    model.add(Conv3D(nfilters,
                     kernel_size=(3, 3, 3),
                     kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # model.add(Dropout(0.3))

    # model.add(BatchNormalization())
    # model.add(LeakyReLU())
    # model.add(Activation("relu"))
    # model.add(AveragePooling3D(pool_size=(2,2,2)))

    model.add(Flatten())
    model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    # model.add(Dropout(0.4))
    model.add(Dense(1))
    # model.compile(optimizer='rmsprop', loss='mse', metrics=['mse',
    #                                                         'mae',
    #                                                         rmse])
    model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mse',
                                                                    'mae',
                                                                    rmse])
    return model

def build_fcn_model(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.

    """
    Personal model
    """
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

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
    model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mse',
                                                                    'mae'])
    return model


class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

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
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')


    voxels = Input(input_shape)
    net = BatchNormalization()(voxels)
    net = Conv3D(filters=nfilters,
                 kernel_size=[3, 3, 3])(voxels)
    net = MaxPooling3D(pool_size=(2, 2, 2))(net)
    net = LeakyReLU()(net)
    net = Conv3D(filters=nfilters,
                 kernel_size=[3, 3, 3])(voxels)
    net = MaxPooling3D(pool_size=(2, 2, 2))(net)
    net = LeakyReLU()(net)
    net = Flatten()(net)
    # what is net?
    #print(net)
    net = Dense(nunits)(net)
    #net = MyLayer(nunits)(net)
    net = LeakyReLU()(net)
    net = Dense(1)(net)
    model = Model(inputs=voxels, outputs=net)
    model.compile(optimizer=optimizers.Adam(), loss='mae', metrics=['mse',
                                                                    'mae'])


def build_model_(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.

    """
    Personal model
    """
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))

    model.add(Conv3D(nfilters,
                    kernel_size=(2, 2, 2),
                    kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters*2,
                     kernel_size=(2, 2, 2),
                     kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())

    model.add(Conv3D(nfilters*4,
                     kernel_size=(2, 2, 2),
                     kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters*8,
                  kernel_size=(2, 2, 2),
                  kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))


    model.add(Flatten())

    model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    # model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    # model.add(LeakyReLU())

    """
    model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    """

    """
    model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    """
    model.add(Dense(1))
    """
    model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['mse',
                                                                             'mae'])
    """
    model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mse',
                                                                             'mae'])

    return model

def build_model_garbage(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.

    """
    Personal model
    """
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    # input_shape=input_shape,

    model.add(Conv3D(nfilters,
                     kernel_size=(2, 2, 2),
                     strides=(1,1,1)))
    #model.add(LeakyReLU())
    model.add(Activation("relu"))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters,
                     kernel_size=(2, 2, 2),
                     strides=(1,1,1)))
    #model.add(LeakyReLU())
    model.add(Activation("relu"))

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters,
                     kernel_size=(2, 2, 2),
                     strides=(1,1,1)))
    #model.add(LeakyReLU())
    model.add(Activation("relu"))

    model.add(Conv3D(nfilters,
                     kernel_size=(2, 2, 2),
                     strides=(1,1,1)))
    #model.add(LeakyReLU())
    model.add(Activation("relu"))

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters,
                     kernel_size=(2, 2, 2),
                     strides=(1,1,1)))
    #model.add(LeakyReLU())
    model.add(Activation("relu"))



    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # model.add(AveragePooling3D(pool_size=(2, 2, 2)))

    """
    model.add(Conv3D(nfilters*2,
                     kernel_size=(2, 2, 2),
                     strides=(1,1,1)))
    #model.add(LeakyReLU())
    model.add(Activation("relu"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    """


    # model.add(AveragePooling3D(pool_size=(4, 4, 4)))

    """
    model.add(Conv3D(nfilters*2,
                     kernel_size=(4, 4, 4),
                     strides=(1,1,1)))
    model.add(LeakyReLU())
    model.add(AveragePooling3D(pool_size=(2, 2, 2)))


    model.add(Conv3D(nfilters*2,
                     kernel_size=(2, 2, 2),
                     strides=(1,1,1)))
    model.add(LeakyReLU())
    model.add(AveragePooling3D(pool_size=(2, 2, 2)))
    """

    # model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Flatten())

    #model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    model.add(Dense(nunits))
    model.add(Activation("relu"))

    model.add(Dense(nunits))
    model.add(Activation("relu"))

    model.add(Dense(nunits))
    model.add(Activation("relu"))

    #model.add(Dense(int(nunits/2), kernel_regularizer=l2(0.001)))
    # model.add(Dense(nunits))
    # model.add(LeakyReLU())

    # model.add(Dense(nunits))
    # model.add(LeakyReLU())

    model.add(Dense(1))

    """
    model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['mse', 'mae'])
    """
    model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['mse', 'mae'])


    return model

def build_model_okg25s25(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.

    """
    Personal model
    """
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    # input_shape=input_shape,
    model.add(Conv3D(nfilters,
                     kernel_size=(2, 2, 2),
                     strides=(1,1,1)))
    model.add(LeakyReLU())

    model.add(AveragePooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(nfilters*2,
                     kernel_size=(2, 2, 2),
                     strides=(1,1,1)))
    model.add(LeakyReLU())

    model.add(AveragePooling3D(pool_size=(2, 2, 2)))

    """
    model.add(Conv3D(nfilters*4,
                     kernel_size=(2, 2, 2),
                     strides=(1,1,1)))
    model.add(LeakyReLU())

    model.add(AveragePooling3D(pool_size=(2, 2, 2)))
    """

    # model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Flatten())

    #model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    model.add(Dense(nunits))
    model.add(LeakyReLU())

    #model.add(Dense(int(nunits/2), kernel_regularizer=l2(0.001)))
    model.add(Dense(nunits))
    model.add(LeakyReLU())

    model.add(Dense(nunits))
    model.add(LeakyReLU())

    model.add(Dense(1))


    model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['mse', 'mae'])
    """
    model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mse', 'mae'])
    """

    return model

def build_model_not_well_Working(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.

    """
    Personal model
    """
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))

    model.add(Conv3D(nfilters,
                     kernel_size=(4, 4, 4),
                     strides=(1,1,1)))
    model.add(LeakyReLU())

    model.add(Conv3D(nfilters,
                     kernel_size=(3, 3, 3),
                     strides=(1,1,1)))
    model.add(LeakyReLU())

    model.add(Conv3D(nfilters*2,
                     kernel_size=(2, 2, 2),
                     strides=(2, 2, 2)))
    model.add(LeakyReLU())

    model.add(Conv3D(nfilters*2,
                     kernel_size=(2, 2, 2),
                     strides=(2, 2, 2)))
    model.add(LeakyReLU())

    model.add(Flatten())

    model.add(Dense(nunits))
    model.add(LeakyReLU())


    model.add(Dense(nunits))
    model.add(LeakyReLU())

    model.add(Dense(1))

    """
    model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['mse', 'mae'])
    """
    model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['mse', 'mae'])
    # Adagrad ok 0.64

    return model


def build_model(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.

    """
    Personal model
    """
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv3D(nfilters,
                      kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters*2,
                      kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())

    model.add(MaxPooling3D(pool_size=(3, 3, 3)))

    model.add(Flatten())

    model.add(Dense(nunits, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())

    model.add(Dense(int(nunits/2), kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    """
    model.add(Dense(nunits, kernel_regularizer=l3(0.001)))
    model.add(LeakyReLU())
    """
    model.add(Dense(1))

    """
    model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['mse', 'mae'])
    """
    model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['mse', 'mae'])
    # Adagrad ok 0.64

    return model

def build_model_orig(conv3d_chtype, input_shape, ndense_layers, nunits, nfilters):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.

    """
    Personal model
    """
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))

    model.add(Conv3D(nfilters,
                     kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters*2,
                     kernel_size=(4, 4, 4)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(nunits))
    model.add(LeakyReLU())
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mse',
                                                                    'mae',
                                                                    rmse])
    return model


def build_2DData_model(conv3d_chtype,
                        input_shape1,
                        input_shape2,
                        ndense_layers,
                        nunits,
                        nfilters):
    """
    Working with keras 2.2.4
    Personal model
    """
    alpha = 0.3
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')


    "Model branch 1"
    in1 = Input(shape=input_shape1)
    m1 = BatchNormalization()(in1)
    m1 = Conv3D(nfilters,
                kernel_size=(2, 2, 2),
                strides=(1,1,1))(m1)
    m1 = LeakyReLU(alpha)(m1)
    m1 = AveragePooling3D(pool_size=(2, 2, 2))(m1)

    m1 = Conv3D(nfilters*2,
                kernel_size=(2, 2, 2),
                strides=(1,1,1))(m1)
    m1 = LeakyReLU(alpha)(m1)
    m1 = AveragePooling3D(pool_size=(2, 2, 2))(m1)

    m1 = Flatten()(m1)

    "Model branch 2"
    in2 = Input(shape=(input_shape2, ))
    m2 = Dense(nunits)(in2)
    m2 = LeakyReLU(alpha)(m2)
    m2 = Dense(nunits)(m2)
    m2 = LeakyReLU(alpha)(m2)
    m2 = Dense(nunits)(m2)
    m2 = LeakyReLU(alpha)(m2)

    "Concatenation"
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
    fmodel.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['mse', 'mae'])
    return fmodel


def build_2DData_model_old(conv3d_chtype,
                        input_shape1,
                        input_shape2,
                        ndense_layers,
                        nunits,
                        nfilters):
    """
    Working with keras 2.2.4
    Personal model
    """
    alpha = 0.3
    print(conv3d_chtype)
    if "last" in conv3d_chtype:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    "Model branch 1"
    in1 = Input(shape=input_shape1)
    m1 = BatchNormalization()(in1)
    m1 = Conv3D(nfilters,
                kernel_size=(3, 3, 3),
                kernel_regularizer=l2(0.001))(m1)
    m1 = LeakyReLU(alpha)(m1)
    m1 = MaxPooling3D(pool_size=(2, 2, 2))(m1)
    m1 = Conv3D(nfilters*2,
                kernel_size=(3, 3, 3),
                kernel_regularizer=l2(0.001))(m1)
    m1 = LeakyReLU(alpha)(m1)
    m1 = MaxPooling3D(pool_size=(3, 3, 3))(m1)
    m1 = Flatten()(m1)

    "Model branch 2"
    in2 = Input(shape=(input_shape2, ))
    m2 = Dense(2)(in2)
    m2 = LeakyReLU(alpha)(m2)

    "Concatenation"
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
    fmodel.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['mse', 'mae'])

    """
    fmodel.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mse',
                                                                     'mae',
                                                                     rmse])
    """
    return fmodel
