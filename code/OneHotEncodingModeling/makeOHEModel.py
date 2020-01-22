#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
from pathlib import Path
import numpy as np
import random
from datetime import datetime
from keras import optimizers
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from keras.models import Model
from keras.models import Input
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras.layers import AveragePooling2D
from keras.layers import Concatenate
from keras.utils import np_utils
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import ParameterGrid
import sys
from sys import argv
import time
import numpy as np
from keras import backend as K
# Some memory clean-up
K.clear_session()


from OHEDatabase import OHEDatabase

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../Base" % (dir_path))
# from FeatureImportance import FeatureImportance, WriteFeatureImportance
from modelhelpers import GetKerasModel
from modelhelpers import GetLoadModelFcn
from modelhelpers import LoadKerasModels

from modelvalidation import MDCTrainTestSplit
from moedlvalidation import TrainTestSplit
from modelvalidation import RepeatedKFold

from dmnnio import ReadDescriptors
from dmnnio import ReadTarget
from dmnnio import WriteCrossValidationOutput

from numpy_loss_functions import RS
from keras_additional_loss_functions import rmse
from keras_additional_loss_functions import score
from keras_additional_loss_functions import np_score


def example_build_2DData_model(dshape, input_shape2, nfilters, nunits):
    input_shape1 = (dshape[0], dshape[1], 1)
    print(input_shape1)
    "Model branch 1"
    in1 = Input(shape=input_shape1)
    m1 = Conv2D(nfilters,
                kernel_size=(1, 6),
                strides=(1, 2))(in1)
    m1 = Activation('relu')(m1)
    m1 = Conv2D(nfilters,
                kernel_size=(1, 6),
                strides=(1, 2))(m1)
    m1 = Activation('relu')(m1)
    m1 = AveragePooling2D(pool_size=(1, 2))(m1)

    # Add 4 time the same convolution
    for i in range(4):
        m1 = Conv2D(nfilters,
                    kernel_size=(1, 6),
                    strides=(1, 1))(m1)
        m1 = Activation('relu')(m1)
    m1 = AveragePooling2D(pool_size=(1, 3))(m1)
    m1 = Dropout(0.3)(m1)
    m1 = Flatten()(m1)

    """
    for i in range(4):
        m1 = Dense(nunits)(m1)
        m1 = Activation("relu")(m1)
    """
    "Model branch 2"
    in2 = Input(shape=(input_shape2, ))
    m2 = Dense(nunits)(in2)
    m2 = Activation("relu")(m2)
    m2 = Dense(nunits)(m2)
    m2 = Activation("relu")(m2)
    """
    m2 = Dense(nunits)(m2)
    m2 = Activation("relu")(m2)
    m2 = Dense(nunits)(m2)
    m2 = Activation("relu")(m2)
    """
    "Concatenation"
    concat = Concatenate()([m1, m2])

    out = Dense(nunits)(concat)
    out = Activation("relu")(out)

    out = Dense(nunits)(out)
    out = Activation("relu")(out)

    out = Dense(1)(out)

    fmodel = Model([in1, in2], out)
    fmodel.compile(optimizer=optimizers.Adam(lr=0.00001),
                   loss='mse', metrics=['mse', 'mae'])
    return fmodel


def example_build_model(dshape, nfilters, nunits):
    input_shape_ = (dshape[0], dshape[1], 1)
    print(input_shape_)
    model = Sequential()
    # model.add(BatchNormalization(input_shape=input_shape_))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 4),
                     strides=(1, 1),
                     input_shape=input_shape_))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 2),
                     strides=(1, 1),
                     input_shape=input_shape_))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 2),
                     strides=(1, 1),
                     input_shape=input_shape_))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 2),
                     strides=(1, 1),
                     input_shape=input_shape_))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 2),
                     strides=(1, 1),
                     input_shape=input_shape_))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 2),
                     strides=(1, 1),
                     input_shape=input_shape_))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 2),
                     strides=(1, 1),
                     input_shape=input_shape_))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 2),
                     strides=(1, 1),
                     input_shape=input_shape_))
    model.add(Activation('relu'))

    """
    model.add(Conv2D(4*nfilters,
                    kernel_size=(1, 3),
                    strides=(1, 2),
                    input_shape=input_shape_))
    model.add(Activation('relu'))
    """

    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(nunits))
    model.add(Activation('relu'))
    model.add(Dense(nunits))
    model.add(Activation('relu'))
    model.add(Dense(nunits))
    model.add(Activation('relu'))
    """
    model.add(Dense(nunits))
    model.add(Activation('relu'))
    """
    model.add(Dense(1))
    # Compile model
    # model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    model.compile(optimizer=optimizers.Adam(),
                  loss='mse', metrics=['mse', 'mae'])
    # model.compile(loss='mse', optimizer='rmsprop', metrics=['mse', 'mae'])
    # model.compile(loss='mse', optimizer='nadam', metrics=['mse', 'mae'])
    # , rmse])
    return model


def build_gridsearch_model(dshape, n_conv, nfilters, ndense_layers, nunits):
    input_shape_ = (dshape[0], dshape[1], 1)
    print(input_shape_)
    model = Sequential()
    # model.add(BatchNormalization(input_shape=input_shape_))

    """
    model.add(Conv2D(nfilters,
                    kernel_size=(1, 6),
                    strides=(1, 4),
                    input_shape=input_shape_))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                    kernel_size=(1, 6),
                    strides=(1, 2),
                    input_shape=input_shape_))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                    kernel_size=(1, 6),
                    strides=(1, 2),
                    input_shape=input_shape_))
    model.add(Activation('relu'))
    """

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 6),
                     strides=(1, 2),
                     input_shape=input_shape_))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 6),
                     strides=(1, 2)))
    model.add(Activation('relu'))

    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 6),
                     strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 6),
                     strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 6),
                     strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Conv2D(nfilters,
                     kernel_size=(1, 6),
                     strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(AveragePooling2D(pool_size=(1, 3)))

    model.add(Dropout(0.3))

    model.add(Flatten())
    for i in range(ndense_layers):
        model.add(Dense(nunits))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

    model.add(Dense(1))
    # Compile model
    model.compile(loss=score,
                  optimizer=optimizers.Adam(lr=0.0001),
                  metrics=[score, 'mse', 'mae'])
    return model


class NNTrain(object):
    def __init__(self, ohedb_path, target, dx=None, n_descs=None):
        self.db = OHEDatabase()
        self.db.loadOHEdb(ohedb_path)
        self.target = target
        self.tgtshape = None
        try:
            self.tgtshape = len(list(target.values())[0])
        except:
            self.tgtshape = 1
        print(self.tgtshape)
        self.dx = dx
        self.n_descs = n_descs
        self.verbose = 1

    def DataGenerator(self, keys, batch_size_=200):
        """ to be used with fit_generator and steps_per_epoch"""
        if self.dx is not None:
            keylst = list(set(list(self.db.X.keys())).intersection(list(self.target.keys())))
            keylst = list(set(keylst).intersection(list(self.dx.keys())))
            keylst = list(set(keylst).intersection(keys))
            size_tkeys = len(keylst)
            if size_tkeys < batch_size_:
                # Give the entire keylist dataset... no random selection
                while True:
                    random.seed(datetime.now().microsecond)
                    batch_x_bh1 = []
                    batch_x_bh2 = []
                    batch_y = []
                    for i in range(size_tkeys):
                        key_ = keylst[i]
                        tval = self.target[key_]
                        smiohes = self.db.X[key_]
                        smiohes_size = len(smiohes)
                        smiohe = None
                        if smiohes_size > 1:
                            k = random.randint(0, smiohes_size-1)
                            smiohe = self.db.X[key_][k]
                        else:
                            smiohe = self.db.X[key_][0]
                        bfeat = smiohe
                        cfeat = self.dx[key_]
                        batch_y.append(tval)
                        batch_x_bh1.append(bfeat)
                        batch_x_bh2.append(cfeat)
                    batch_x_bh1 = np.array(batch_x_bh1)[:, :, :, np.newaxis]
                    batch_x_bh2 = np.array(batch_x_bh2).astype(float)
                    yield([batch_x_bh1, batch_x_bh2], np.array(batch_y))
            else:
                # Random selection several time...
                while True:
                    random.seed(datetime.now().microsecond)
                    batch_x_bh1 = []
                    batch_x_bh2 = []
                    batch_y = []
                    for i in range(batch_size_):
                        indx = random.randint(0, size_tkeys-1)
                        key_ = keylst[indx]
                        tval = self.target[key_]
                        smiohes = self.db.X[key_]
                        smiohes_size = len(smiohes)
                        smiohe = None
                        if smiohes_size > 1:
                            k = random.randint(0, smiohes_size-1)
                            smiohe = self.db.X[key_][k]
                        else:
                            smiohe = self.db.X[key_][0]
                        bfeat = smiohe
                        cfeat = self.dx[key_]
                        batch_y.append(tval)
                        batch_x_bh1.append(bfeat)
                        batch_x_bh2.append(cfeat)
                    batch_x_bh1 = np.array(batch_x_bh1)[:, :, :, np.newaxis]
                    batch_x_bh2 = np.array(batch_x_bh2).astype(float)
                    yield([batch_x_bh1, batch_x_bh2], np.array(batch_y))
        else:
            keylst = list(set(list(self.db.X.keys())).intersection(list(self.target.keys())))
            keylst = list(set(keylst).intersection(keys))
            size_tkeys = len(keylst)
            if size_tkeys < batch_size_:
                while True:
                    random.seed(datetime.now().microsecond)
                    batch_x = []
                    batch_y = []
                    for i in range(size_tkeys):
                        key_ = keylst[i]
                        tval = self.target[key_]
                        smiohes = self.db.X[key_]
                        smiohes_size = len(smiohes)
                        smiohe = None
                        if smiohes_size > 1:
                            k = random.randint(0, smiohes_size-1)
                            smiohe = self.db.X[key_][k]
                        else:
                            smiohe = self.db.X[key_][0]
                        batch_y.append(tval)
                        batch_x.append(smiohe)
                    batch_x = np.array(batch_x)[:, :, :, np.newaxis]
                    yield(batch_x, np.array(batch_y))
            else:
                while True:
                    random.seed(datetime.now().microsecond)
                    batch_x = []
                    batch_y = []
                    for i in range(batch_size_):
                        indx = random.randint(0, size_tkeys-1)
                        key_ = keylst[indx]
                        tval = self.target[key_]
                        smiohes = self.db.X[key_]
                        smiohes_size = len(smiohes)
                        smiohe = None
                        if smiohes_size > 1:
                            k = random.randint(0, smiohes_size-1)
                            smiohe = self.db.X[key_][k]
                        else:
                            smiohe = self.db.X[key_][0]
                        batch_y.append(tval)
                        batch_x.append(smiohe)
                    batch_x = np.array(batch_x)[:, :, :, np.newaxis]
                    yield(batch_x, np.array(batch_y))

    def GenData(self, keys):
        if self.dx is not None:
            ret_keys = []
            batch_x_bh1 = []
            batch_x_bh2 = []
            batch_y = []
            for key in keys:
                try:
                    tval = self.target[key]
                    cfeat = self.dx[key]
                    batch_y.append(tval)
                    smiohes = self.db.X[key]
                    smiohes_size = len(smiohes)
                    smiohe = None
                    if smiohes_size > 1:
                        k = random.randint(0, smiohes_size-1)
                        smiohe = smiohes[k]
                    else:
                        smiohe = smiohes[0]
                    batch_x_bh1.append(smiohe)
                    batch_x_bh2.append(cfeat)
                    ret_keys.append(key)
                except KeyError:
                    print("Molecule %s not found" % (key))
            batch_x_bh1 = np.array(batch_x_bh1)[:, :, :, np.newaxis]
            batch_x_bh2 = np.array(batch_x_bh2).astype(float)
            return [batch_x_bh1, batch_x_bh2], np.array(batch_y), ret_keys
        else:
            ret_keys = []
            batch_features = []
            batch_target = []
            for key in keys:
                try:
                    tval = self.target[key]
                    smiohes = self.db.X[key]
                    smiohes_size = len(smiohes)
                    smiohe = None
                    if smiohes_size > 1:
                        k = random.randint(0, smiohes_size-1)
                        smiohe = smiohes[k]
                    else:
                        smiohe = smiohes[0]
                    batch_target.append(tval)
                    batch_features.append(smiohe)
                    ret_keys.append(key)
                except KeyError:
                    print("Molecule %s not found" % (key))
                    continue
            # shape = np.array(batch_features).shape
            """
            batch_features = np.array(batch_features).reshape(shape[0],
                                                              shape[1],
                                                              shape[2], 1)
            """
            batch_features = np.array(batch_features)[:, :, :, np.newaxis]
            return batch_features, np.array(batch_target), ret_keys

    def simplerun(self,
                  batch_size_,
                  num_epochs,
                  steps_per_epochs_,
                  nfilters,
                  nunits,
                  mout=None):
        print("N. instances: %d" % (len(self.target)))
        # train_keys, test_keys = MDCTrainTestSplit(self.target, 0)
        train_keys, test_keys = TrainTestSplit(self.target, test_size_=0.20)
        print("Train set size: %d Test set size %d" % (len(train_keys),
                                                       len(test_keys)))
        model = None
        model_ = GetKerasModel()

        if self.dx is not None:
            print("Number of descriptors: %d" % (self.n_descs))
            if model_ is None:
                model = example_build_2DData_model(self.db.input_shape,
                                                   self.n_descs,
                                                   nfilters,
                                                   nunits)
            else:
                model = model_(self.db.input_shape,
                               self.n_descs,
                               nfilters,
                               nunits)
        else:
            if model_ is None:
                model = example_build_model(self.db.input_shape,
                                            nfilters,
                                            nunits)
            else:
                model = model_(self.db.input_shape, nfilters, nunits)

        print(model.summary())

        x_train, y_train, rtrain_keys = self.GenData(train_keys)
        x_test, y_test, rtest_keys = self.GenData(test_keys)
        if self.dx is not None:
            print("Branch 1 size:", np.array(x_train[0]).shape)
            print("Branch 2 size:", np.array(x_train[1]).shape)
        else:
            print(x_train.shape)

        print(y_train.shape)
        b = 0
        if batch_size_ is None:
            b = len(x_test)
        else:
            b = batch_size_

        name = "#b%d_#e%d_#u%d_#f%d_" % (b,
                                         num_epochs,
                                         nunits,
                                         nfilters)
        name += time.strftime("%Y%m%d%H%M%S")
        log_dir_ = ("./logs/%s" % (name))

        callbacks_ = [TensorBoard(log_dir=log_dir_,
                                  histogram_freq=0,
                                  write_graph=False,
                                  write_images=False)]
        """
        callbacks_ = [TensorBoard(log_dir=log_dir_,
                                  histogram_freq=0,
                                  write_graph=False,
                                  write_images=False),
                      EarlyStopping(monitor='val_loss',
                                    min_delta=0,
                                    patience=50,
                                    verbose=0,
                                    mode='auto')]
        """

        #train_steps_per_epoch = int(np.ceil(len(train_keys)/float(batch_size_)))
        train_generator = self.DataGenerator(train_keys, batch_size_)
        model.fit_generator(train_generator,
                            steps_per_epoch=steps_per_epochs_,
                            epochs=num_epochs,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            # validation_data=test_generator,
                            # validation_steps=test_steps_per_epoch,
                            callbacks=callbacks_)
        """
        model.fit(x_train, y_train,
                  epochs=num_epochs,
                  batch_size=b,
                  verbose=self.verbose,
                  validation_data=(x_test, y_test),
                  callbacks=callbacks_)
        """
        yrecalc = model.predict(x_train)
        ypred_test = model.predict(x_test)
        fo = open("%s_pred.csv" % (name), "w")
        if ypred_test.shape[1] > 1:
            for i in range(len(rtest_keys)):
                fo.write("%s," % (rtest_keys[i]))
                for j in range(len(y_test[i])-1):
                    fo.write("%f,%f," % (y_test[i][j], ypred_test[i][j]))
                fo.write("%f,%f\n" % (y_test[i][-1], ypred_test[i][-1]))
            fo.close()
            # Then calculate R2 and Q2 for each output...
            for j in range(ypred_test.shape[1]):
                y_train_ = []
                yrecalc_ = []
                y_test_ = []
                ypred_test_ = []
                for i in range(ypred_test.shape[0]):
                    y_train_.append(y_train[i][j])
                    yrecalc_.append(yrecalc[i][j])
                    y_test_.append(y_test[i][j])
                    ypred_test_.append(ypred_test[i][j])
                print("Output %d R2: %.4f Q2: %.4f" % (j,
                                                       r2_score(y_train_, yrecalc_),
                                                       r2_score(y_test_, ypred_test_)))
        else:
            for i in range(len(rtest_keys)):
                fo.write("%s" % (rtest_keys[i]))
                for j in range(len(y_test[i])):
                    fo.write("%f,%f" % (y_test[i],
                                        ypred_test[i]))
                fo.write("\n")
            fo.close()
            print("R2: %.4f Q2: %.4f" % (r2_score(y_train, yrecalc),
                                         r2_score(y_test, ypred_test)))

    def runcv(self,
              batch_size_,
              num_epochs,
              steps_per_epochs_,
              nfilters,
              nunits,
              cvout,
              n_splits=5,
              n_repeats=10,
              mout=None):
        print("N. instances: %d" % (len(self.target)))

        mout_path = None
        if mout is not None:
            # Utilised to store the out path
            mout_path = Path("%s_%s" % (time.strftime("%Y%m%d%H%M%S"), mout))
            mout_path.mkdir()
            # Save the descriptor order
            """
            f = open("%s/odesc_header.csv" % (str(mout_path.absolute())), "w")
            for name in self.xheader:
                f.write("%s\n" % (name))
            f.close()
            """

        cv_ = 0
        predictions = {}
        recalc = {}
        for key in self.target.keys():
            # N.B.: each molecule can have multiple outputs.
            predictions[key] = []
            recalc[key] = []

        for dataset_keys, test_keys in RepeatedKFold(n_splits,
                                                     n_repeats,
                                                     self.target):
            print("Dataset size: %d Validation  size %d" % (len(dataset_keys),
                                                            len(test_keys)))

            sub_target = {}
            for key in dataset_keys:
                sub_target[key] = self.target[key]

            # ntobj = int(np.ceil(len(sub_target)*0.1))
            # train_keys, test_keys = MDCTrainTestSplit(sub_target, ntobj)
            train_keys, val_keys = TrainTestSplit(sub_target, test_size_=0.20)

            x_train, y_train, rtrain_keys = self.GenData(train_keys)
            x_val, y_val, rval_keys = self.GenData(val_keys)
            x_test, y_test, rtest_keys = self.GenData(test_keys)

            print("Train set size: %d Validation set size %d" % (len(train_keys),
                                                                 len(val_keys)))

            model = None
            model_ = GetKerasModel()
            if self.dx is not None:
                print("Number of descriptors: %d" % (self.n_descs))
                if model_ is None:
                    model = example_build_2DData_model(self.db.input_shape,
                                                       self.n_descs,
                                                       nfilters,
                                                       nunits)
                else:
                    model = model_(self.db.input_shape,
                                   self.n_descs,
                                   nfilters,
                                   nunits)
            else:
                if model_ is None:
                    model = example_build_model(self.db.input_shape,
                                                nfilters,
                                                nunits)
                else:
                    model = model_(self.db.input_shape,
                                   nfilters,
                                   nunits)

            print(model.summary())
            dname = cvout.replace(".csv", "")
            b = 0
            if batch_size_ is None:
                b = len(x_val)
            else:
                b = batch_size_

            name = "cv%d_%s_#b%d_#e%d_#u%d_#f%d_" % (cv_,
                                                     dname,
                                                     b,
                                                     num_epochs,
                                                     nunits,
                                                     nfilters)
            name += time.strftime("%Y%m%d%H%M%S")
            log_dir_ = ("./logs/%s" % (name))

            model_output = None
            if mout_path is not None:
                model_output = "%s/%d.h5" % (str(mout_path.absolute()), cv_)
            if model_output is None:
                callbacks_ = [TensorBoard(log_dir=log_dir_,
                                          histogram_freq=0,
                                          write_graph=False,
                                          write_images=False)]
            else:
                callbacks_ = [TensorBoard(log_dir=log_dir_,
                                          histogram_freq=0,
                                          write_graph=False,
                                          write_images=False),
                              ModelCheckpoint(model_output,
                                              monitor='val_loss',
                                              verbose=0,
                                              save_best_only=True)]

            train_generator = self.DataGenerator(train_keys, batch_size_)
            model.fit_generator(train_generator,
                                steps_per_epoch=steps_per_epochs_,
                                epochs=num_epochs,
                                verbose=1,
                                validation_data=(x_val, y_val),
                                # validation_data=test_generator,
                                # validation_steps=test_steps_per_epoch,
                                callbacks=callbacks_)
            """
            model.fit(x_train, y_train,
                      epochs=num_epochs,
                      batch_size=b,
                      steps_per_epochs=steps_per_epochs_,
                      verbose=1,
                      validation_data=(x_val, y_val),
                      callbacks=callbacks_)
            """
            # WARNING Implement cross validation results for multiple outputs
            bestmodel = load_model(model_output,
                                   custom_objects={"score": score})
            yrecalc = bestmodel.predict(x_train)
            for i in range(len(yrecalc)):
                recalc[train_keys[i]].extend(list(yrecalc[i]))

            ypred_val = bestmodel.predict(x_val)
            print("Test R2: %.4f" % (r2_score(y_val, ypred_val)))

            ypred_test = bestmodel.predict(x_test)
            # exp_pred_plot(y_val_, ypred[:,0])
            print("Validation R2: %.4f" % (r2_score(y_test, ypred_test)))
            for i in range(len(ypred_test)):
                predictions[test_keys[i]].extend(list(ypred_test[i]))

            """
            if fimpfile is not None:
                fimp = FeatureImportance(model, x_val, y_val, self.xheader)
                fires = fimp.Calculate(verbose=1)
                for key in fires.keys():
                    feat_imp[key]['mae'].extend(fires[key]['mae'])
                    feat_imp[key]['mse'].extend(fires[key]['mse'])
            """
            cv_ += 1

        WriteCrossValidationOutput(cvout, self.target, predictions, recalc)
        """
        fo = open(cvout, "w")
        if self.tgtshape > 1:
            for i in range(len(rtest_keys)):
                fo.write("%s," % (rtest_keys[i]))
                for j in range(len(y_test[i])-1):
                    fo.write("%f,%f," % (y_test[i][j], ypred_test[i][j]))
                fo.write("%f,%f\n" % (y_test[i][-1], ypred_test[i][-1]))
            fo.close()
            # Then calculate R2 and Q2 for each output...
            for j in range(ypred_test.shape[1]):
                y_train_ = []
                yrecalc_ = []
                y_test_ = []
                ypred_test_ = []
                for i in range(ypred_test.shape[0]):
                    y_train_.append(y_train[i][j])
                    yrecalc_.append(yrecalc[i][j])
                    y_test_.append(y_test[i][j])
                    ypred_test_.append(ypred_test[i][j])
                print("Output %d R2: %.4f Q2: %.4f" % (j,
                                                       r2_score(y_train_,
                                                                yrecalc_),
                                                       r2_score(y_test_,
                                                                ypred_test_)))
        else:
            for i in range(len(rtest_keys)):
                fo.write("%s,%f,%f\n" % (rtest_keys[i],
                                         y_test[i],
                                         ypred_test[i]))
            fo.close()
            print("R2: %.4f Q2: %.4f" % (r2_score(y_train, yrecalc),
                                         r2_score(y_test, ypred_test)))

        fo = open(cvout, "w")
        if self.tgtshape > 1:
            for key in predictions.keys():
                fo.write("%s," % (key))

                if len(predictions[key]) > 0:
                    freq = len(predictions[key])
                    ypavg = np.mean(predictions[key])
                    ystdev = np.std(predictions[key])
                    res = self.target[key] - ypavg
                    fo.write("%.4f,%.4f,%.4f,%.4f,%d," % (self.target[key],
                                                          ypavg,
                                                          ystdev,
                                                          res,
                                                          freq))
                else:
                    fo.write("%.4f,0.0,0.0,0.0," % (self.target[key]))

                if len(recalc[key]) > 0:
                    freq_r = len(recalc[key])
                    ypavg_r = np.mean(recalc[key])
                    ystdev_r = np.std(recalc[key])
                    res_r = self.target[key] - ypavg_r
                    fo.write("%.4f,%.4f,%.4f,%d\n" % (ypavg_r,
                                                      ystdev_r,
                                                      res_r,
                                                      freq_r))
                else:
                    fo.write("0.0,0.0,0.0\n")
        else:
            for key in predictions.keys():
                fo.write("%s," % (key))
                if len(predictions[key]) > 0:
                    freq = len(predictions[key])
                    ypavg = np.mean(predictions[key])
                    ystdev = np.std(predictions[key])
                    res = self.target[key] - ypavg
                    fo.write("%.4f,%.4f,%.4f,%.4f,%d," % (self.target[key],
                                                          ypavg,
                                                          ystdev,
                                                          res,
                                                          freq))
                else:
                    fo.write("%.4f,0.0,0.0,0.0," % (self.target[key]))

                if len(recalc[key]) > 0:
                    freq_r = len(recalc[key])
                    ypavg_r = np.mean(recalc[key])
                    ystdev_r = np.std(recalc[key])
                    res_r = self.target[key] - ypavg_r
                    fo.write("%.4f,%.4f,%.4f,%d\n" % (ypavg_r,
                                                      ystdev_r,
                                                      res_r,
                                                      freq_r))
                else:
                    fo.write("0.0,0.0,0.0\n")
        fo.close()
        """

    def runloo(self, batch_size_, num_epochs, ndense_layers, nunits, cvout):
        print("N. instances: %d" % (len(self.target)))
        predictions = dict()
        for val_key in self.target.keys():
            sub_target = {}
            for key in self.target.keys():
                if val_key == key:
                    continue
                else:
                    sub_target[key] = self.target[key]
                    # train_keys.append(key)
            print("Validating %s" % (val_key))

            # train_keys, test_keys = MDCTrainTestSplit(sub_target, 0)
            train_keys, test_keys = TrainTestSplit(sub_target, test_size_=0.20)
            x_train, y_train, rtrain_keys = self.GenData(train_keys)
            x_test, y_test, rtest_keys = self.GenData(test_keys)

            model = None
            model_ = GetKerasModel()
            if model_ is None:
                model = example_build_model(self.nfeatures,
                                            nunits,
                                            ndense_layers)
            else:
                model = model_(self.nfeatures,
                               nunits,
                               ndense_layers)

            print(model.summary())
            b = 0
            if batch_size_ is None:
                b = len(x_test)
            else:
                b = batch_size_
            log_dir_ = ("./logs/%s_#b%d_#e%d_#u%d_#dl%d_" % (val_key,
                                                             b,
                                                             num_epochs,
                                                             nunits,
                                                             ndense_layers))
            log_dir_ += time.strftime("%Y%m%d%H%M%S")

            callbacks_ = [TensorBoard(log_dir=log_dir_,
                                      histogram_freq=0,
                                      write_graph=False,
                                      write_images=False)]
            """
            callbacks_ = [TensorBoard(log_dir=log_dir_,
                                      histogram_freq=0,
                                      write_graph=False,
                                      write_images=False),
                          EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=3,
                                        verbose=0,
                                        mode='auto')]
            """

            model.fit(x_train, y_train,
                      epochs=num_epochs,
                      batch_size=b,
                      verbose=1,
                      validation_data=(x_test, y_test),
                      callbacks=callbacks_)

            predictions[val_key] = model.predict(x_test)[0]

        fo = open(cvout, "w")
        for key in predictions.keys():
            fo.write("%s,%.4f,%.4f\n" % (key,
                                         self.target[key],
                                         predictions[key]))
        fo.close()

    def GridSearch(self,
                   batch_size_,
                   steps_per_epochs_,
                   num_epochs,
                   gmout="GridSearchResult"):

        train_keys, test_keys = TrainTestSplit(self.target, test_size_=0.20)
        print("Train set size: %d Test set size %d" % (len(train_keys),
                                                       len(test_keys)))

        # train_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
        # train_generator = self.DataGenerator(train_keys, batch_size_)

        x_train, y_train, rtrain_keys = self.GenData(train_keys)

        # This is unstable
        # test_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
        # test_generator = self.DataGenerator(test_keys, batch_size_)
        # This is more stable
        x_test, y_test, rtest_keys = self.GenData(test_keys)

        # PARAMETER DEFINITIONS
        # simple architecture
        """
        param = {}
        param["nunits"] = [100, 200, 400]
        param["ndense_layers"] = [2, 4, 6]
        param["dropout"] = ["on", "off"]
        #param["activation"] = ["relu", "leakyrelu"]
        param["activation"] = ["relu"]
        """

        # resnet architecture
        param = {}
        param["nunits"] = [200, 400,  600, 800]
        param["ndense_layers"] = [2, 4, 6, 8]

        all_combo = list(ParameterGrid(param))
        print("Evaluating %d combinations of parameters" % (len(all_combo)))

        already_computed_combo = []
        if Path(gmout).is_file():
            fi = open(gmout, "r")
            for line in fi:
                v = str.split(line.strip(), " ")
                """
                # simple architecture
                units = v[0]
                layers = v[1]
                act = v[2]
                drop = v[3]
                s = ("%s-%s-%s-%s" % (units, layers, act, drop))
                """
                # resnet architecture
                units = v[0]
                layers = v[1]
                s = ("%s-%s" % (units, layers))
                already_computed_combo.append(s)
            fi.close()
        model_ = GetKerasModel()
        for c in all_combo:
            """
            # simple architecture
            s = ("%s-%s-%s-%s" % (c["nunits"],
                                  c["ndense_layers"],
                                  c["activation"],
                                  c["dropout"]))
            """
            # resnet architecture
            s = ("%s-%s" % (c["nunits"], c["ndense_layers"]))
            if s in already_computed_combo:
                print("%s already computed... skip..." % (s))
            else:
                """
                model = build_gridsearch_model(self.nfeatures,
                                              c["ndense_layers"],
                                              c["nunits"],
                                              c["activation"],
                                              c["dropout"])
                """
                if model_ is None:
                    model = example_build_model(self.nfeatures,
                                                c["nunits"],
                                                c["ndense_layers"])
                else:
                    model = model_(self.nfeatures,
                                   c["nunits"],
                                   c["ndense_layers"])

                """
                model = build_dnn_resnet_model(self.nfeatures,
                                               c["nunits"],
                                               c["ndense_layers"])
                """

                print(model.summary())
                b = batch_size_
                """
                model_name = ("#b%d_#e%d_#u%d_#dl%d_act-%s_dp-%s" % (b,
                                                                    num_epochs,
                                                                    c["nunits"],
                                                                    c["ndense_layers"],
                                                                    c["activation"],
                                                                    c["dropout"]))
                """

                model_name = ("#b%d_#e%d_#u%d_#dl%d" % (b,
                                                        num_epochs,
                                                        c["nunits"],
                                                        c["ndense_layers"]))
                log_dir_ = ("./logs/%s" % (model_name))

                log_dir_ += time.strftime("%Y%m%d%H%M%S")

                model_output = "%s.h5" % (model_name)
                callbacks_ = [TensorBoard(log_dir=log_dir_,
                                          histogram_freq=0,
                                          write_graph=False,
                                          write_images=False),
                              ModelCheckpoint(model_output,
                                              monitor='val_loss',
                                              verbose=0,
                                              save_best_only=True)]
                """
                callbacks_ = [TensorBoard(log_dir=log_dir_,
                                          histogram_freq=0,
                                          write_graph=False,
                                          write_images=False),
                              EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=50,
                                            verbose=0,
                                            mode='auto')]
                """

                model.fit(x_train, y_train,
                          epochs=num_epochs,
                          batch_size=b,
                          steps_per_epochs=steps_per_epochs_,
                          verbose=self.verbose,
                          validation_data=(x_test, y_test),
                          callbacks=callbacks_)

                bestmodel = load_model(model_output,
                                       custom_objects={"score": score})

                yrecalc_train = bestmodel.predict(x_train)

                """

                model.fit_generator(train_generator,
                                    steps_per_epoch=train_steps_per_epoch,
                                    epochs=num_epochs,
                                    verbose=1,
                                    validation_data=(x_test, y_test),
                                    # validation_data=test_generator,
                                    # validation_steps=test_steps_per_epoch,
                                    callbacks=callbacks_)


                yrecalc_train = []
                y_train = []
                for key in train_keys:
                    a = np.array([self.X_raw[key]])
                    yrecalc_train.extend(model.predict(a))
                    y_train.append(self.target[key])
                """
                ypred_test = bestmodel.predict(x_test)
                r2 = r2_score(y_train, yrecalc_train)
                mse_train = mse(y_train, yrecalc_train)
                mae_train = mae(y_train, yrecalc_train)
                q2 = r2_score(y_test, ypred_test)
                mse_test = mse(y_test, ypred_test)
                mae_test = mae(y_test, ypred_test)
                train_score = np_score(y_train, yrecalc_train)
                test_score = np_score(y_test, ypred_test)
                print("R2: %.4f Train Score: %f Q2: %.4f Test Score: %f" % (r2, train_score, q2, test_score))

                fo = open("%s" % (gmout), "a")
                """
                # simple architecture
                fo.write("%d %d %s %s %f %f %f %f %f %f %f %f\n" % (c["nunits"],
                                                                    c["ndense_layers"],
                                                                    c["activation"],
                                                                    c["dropout"],
                                                                    mse_train,
                                                                    mae_train,
                                                                    r2,
                                                                    train_score,
                                                                    mse_test,
                                                                    mae_test,
                                                                    q2,
                                                                    test_score))
                """
                # resnet architecture
                fo.write("%d %d %f %f %f %f %f %f %f %f\n" % (c["nunits"],
                                                              c["ndense_layers"],
                                                              mse_train,
                                                              mae_train,
                                                              r2,
                                                              train_score,
                                                              mse_test,
                                                              mae_test,
                                                              q2,
                                                              test_score))
                fo.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ohedir', default=None,
                        type=str, help='One-Hot-Encoding directory files')
    parser.add_argument('--xmatrix', default=None,
                        type=str, help='input data matrix')
    parser.add_argument('--ytarget', default=None,
                        type=str, help='input data matrix')
    parser.add_argument('--epochs', default=500, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size')
    parser.add_argument('--steps_per_epochs', default=200, type=int,
                        help='Steps per epochs')
    parser.add_argument('--n_splits', default=5, type=int,
                        help='Number of kfold splits')
    parser.add_argument('--n_repeats', default=20, type=int,
                        help='Number of repetitions')
    parser.add_argument('--cvout', default=None, type=str,
                        help='Cross validation output')
    parser.add_argument('--nfilters', default=16, type=int,
                        help='Number of neurons')
    parser.add_argument('--nunits', default=32, type=int,
                        help='Number of neurons')
    parser.add_argument('--ndense_layers', default=1, type=int,
                        help='Number of dense layers')
    parser.add_argument('--mout', default=None, type=str,
                        help='Model out path')

    args = parser.parse_args(argv[1:])
    if args.ohedir is None or args.ytarget is None:
        print("ERROR! Please specify input ohe directory and a target to train!")
        print("\n Usage: %s --ohedir [input file] --xmatrix [optinal csv descriptors] --ytarget [input file] --epochs [int] --batch_size [int] --nunits [int] --nfilters [int]\n\n" % (argv[0]))
    else:
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        dx = None
        n_descs = None
        desc_headers = None
        if args.xmatrix is not None:
            if ".csv" in args.xmatrix:
                # dx = pd.read_csv(args.xmatrix)
                dx, n_descs, desc_headers = ReadDescriptors(args.xmatrix)
            else:
                # dx = pd.read_table(args.xmatrix, header=0)
                dx, n_descs, desc_headers = ReadDescriptors(args.xmatrix, "\t")
        else:
            dx = None

        target = None
        if ".csv" in args.ytarget:
            target = ReadTarget(args.ytarget)
            # dy = pd.read_csv(args.ytarget)
        else:
            target = ReadTarget(args.ytarget, sep="\t")
            # dy = pd.read_table(args.ytarget, header=0)

        """
        target = {}
        for row in dy.values:
            target[row[0]] = row[1:].astype(float)
        """

        nn = NNTrain(args.ohedir, target, dx, n_descs)
        if args.cvout is None:
            nn.verbose = 1
            nn.simplerun(args.batch_size,
                         args.epochs,
                         args.steps_per_epochs,
                         args.nfilters,
                         args.nunits)
        else:
            nn.runcv(args.batch_size,
                     args.epochs,
                     args.steps_per_epochs,
                     args.nfilters,
                     args.nunits,
                     args.cvout,
                     args.n_splits,
                     args.n_repeats,
                     args.mout)
            """
            nn.runloo(args.batch_size,
                      args.epochs,
                      args.ndense_layers,
                      args.nunits,
                      args.cvout)
            """


if __name__ == '__main__':
    main()
