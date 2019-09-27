#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import argparse
from pathlib import Path
import numpy as np
import random
from datetime import datetime
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.models import Model, Input, Sequential
from keras.models import load_model
from keras.layers import Conv2D, Flatten, Dense, Dropout, Activation, MaxPooling2D, ZeroPadding2D, BatchNormalization, AveragePooling2D, Concatenate
from keras.utils import np_utils
from sklearn.metrics import r2_score
import sys
from sys import argv
import time
import numpy as np
from keras import backend as K
# Some memory clean-up
K.clear_session()

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../Base" % (dir_path))
# from FeatureImportance import FeatureImportance, WriteFeatureImportance
from misc import MDCTrainTestSplit, TrainTestSplit, RepeatedKFold, ReadDescriptors, ReadTarget
from keras_additional_loss_functions import rmse, score

def build_2DData_model(dshape, input_shape2, nfilters, nunits):
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


def build_model(dshape, nfilters, nunits):
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
                  optimizer=optimizers.Adam(lr=0.0001), metrics=[score, 'mse', 'mae'])
    return model

class NNTrain(object):
    def __init__(self, ohe_dir, target, dx=None, n_descs=None):
        self.X, self.input_shape = self.LoadOneHotEncoding(ohe_dir)
        self.target = target
        self.dx = dx
        self.n_descs = n_descs
        self.verbose = 1

    def ReadOneHotEncodingCSV(self, ohe_csv):
        f = open(ohe_csv, "r")
        m = []
        for line in f:
            m.append(str.split(line.strip(), ","))
        f.close()
        return np.array(m)

    def LoadOneHotEncoding(self, ohe_dir):
        p = Path(ohe_dir).glob('**/*')
        X = {}
        for x in p:
            if x.is_file() and ".csv" in str(x):
                key = x.resolve().stem.split(".")[0]
                X[key] = self.ReadOneHotEncodingCSV(x)
            else:
                continue
        return X, list(X.values())[0].shape

    def DataGenerator(self, train_keys, batch_size_=20):
        """ to be used with fit_generator and steps_per_epoch"""
        size_tkeys = len(train_keys)
        batch_size = None

        if train_keys < batch_size_:
            batch_size = size_tkeys
        else:
            batch_size = batch_size_

        if self.dx is not None:
            while True:
                random.seed(datetime.now().microsecond)
                batch_x_bh1 = []
                batch_x_bh2 = []
                batch_y = []
                for i in range(batch_size):
                    indx = random.randint(0, size_tkeys-1)
                    key_ = train_keys[indx]
                    try:
                        tval = self.target[key_]
                        bfeat = self.X[key_]
                        cfeat = self.dx[key_]
                        batch_y.append(tval)
                        batch_x_bh1.append(bfeat)
                        batch_x_bh2.append(cfeat)
                    except KeyError:
                        print("Molecule %s not found" % (key_))
                batch_x_bh1 = np.array(batch_x_bh1)[:, :, :, np.newaxis]
                batch_x_bh2 = np.array(batch_x_bh2).astype(float)
                yield([batch_x_bh1, batch_x_bh2], np.array(batch_y))
        else:
            while True:
                random.seed(datetime.now().microsecond)
                batch_x = []
                batch_y = []
                for i in range(batch_size):
                    indx = random.randint(0, size_tkeys-1)
                    key_ = train_keys[indx]
                    try:
                        tval = self.target[key_]
                        bfeat = self.X[key_]
                        batch_y.append(tval)
                        batch_x.append(bfeat)
                    except KeyError:
                        print("Molecule %s not found" % (key_))
                batch_x = np.array(batch_x)[:, :, :, np.newaxis]
                yield(batch_x, np.array(batch_y))

    def GenData(self, keys):
        if self.dx is not None:
            batch_x_bh1 = []
            batch_x_bh2 = []
            batch_y = []
            for key in keys:
                try:
                    tval = self.target[key]
                    bfeat = self.X[key]
                    cfeat = self.dx[key]
                    batch_y.append(tval)
                    batch_x_bh1.append(bfeat)
                    batch_x_bh2.append(cfeat)
                except KeyError:
                    print("Molecule %s not found" % (key))
            batch_x_bh1 = np.array(batch_x_bh1)[:, :, :, np.newaxis]
            batch_x_bh2 = np.array(batch_x_bh2).astype(float)
            return [batch_x_bh1, batch_x_bh2], np.array(batch_y)
        else:
            batch_features = []
            batch_target = []
            for key in keys:
                try:
                    tval = self.target[key]
                    bfeat = self.X[key]
                    batch_target.append(tval)
                    batch_features.append(bfeat)
                except KeyError:
                    print("Molecule %s not found" % (key))
            # shape = np.array(batch_features).shape
            """
            batch_features = np.array(batch_features).reshape(shape[0],
                                                              shape[1],
                                                              shape[2], 1)
            """
            batch_features = np.array(batch_features)[:, :, :, np.newaxis]
            return batch_features, np.array(batch_target)

    def simplerun(self,
                  batch_size_,
                  num_epochs,
                  nfilters,
                  nunits,
                  mout=None):
        print("N. instances: %d" % (len(self.target)))
        # train_keys, test_keys = MDCTrainTestSplit(self.target, 0)
        train_keys, test_keys = TrainTestSplit(self.target, test_size_=0.20)
        print("Train set size: %d Test set size %d" % (len(train_keys),
                                                       len(test_keys)))
        model = None
        if self.dx is not None:
            print("Number of descriptors: %d" % (self.n_descs))
            model = build_2DData_model(self.input_shape,
                                       self.n_descs,
                                       nfilters,
                                       nunits)
        else:
            model = build_model(self.input_shape, nfilters, nunits)
        print(model.summary())

        x_train, y_train = self.GenData(train_keys)
        x_test, y_test = self.GenData(test_keys)
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
        log_dir_ = ("./logs/#b%d_#e%d_#u%d_#f%d_" % (b,
                                                     num_epochs,
                                                     nunits,
                                                     nfilters))
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
                                    patience=50,
                                    verbose=0,
                                    mode='auto')]
        """
        model.fit(x_train, y_train,
                  epochs=num_epochs,
                  batch_size=b,
                  verbose=self.verbose,
                  validation_data=(x_test, y_test),
                  callbacks=callbacks_)

        yrecalc = model.predict(x_train)
        ypred_test = model.predict(x_test)
        print("R2: %.4f Q2: %.4f" % (r2_score(y_train, yrecalc),
                                     r2_score(y_test, ypred_test)))

    def runcv(self,
              batch_size_,
              num_epochs,
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
            predictions[key] = []
            recalc[key] = []

        for dataset_keys, val_keys in RepeatedKFold(n_splits,
                                                    n_repeats,
                                                    self.target):
            print("Dataset size: %d Validation  size %d" % (len(dataset_keys),
                                                            len(val_keys)))

            sub_target = {}
            for key in dataset_keys:
                sub_target[key] = self.target[key]
            ntobj = int(np.ceil(len(sub_target)*0.1))
            # train_keys, test_keys = MDCTrainTestSplit(sub_target, ntobj)
            train_keys, test_keys = TrainTestSplit(sub_target, test_size_=0.20)

            x_train, y_train = self.GenData(train_keys)
            x_test, y_test = self.GenData(test_keys)
            x_val, y_val = self.GenData(val_keys)

            print("Train set size: %d Test set size %d" % (len(train_keys),
                                                           len(test_keys)))

            model = None
            if self.dx is not None:
                print("Number of descriptors: %d" % (self.n_descs))
                model = build_2DData_model(self.input_shape,
                                           self.n_descs,
                                           nfilters,
                                           nunits)
            else:
                model = build_model(self.input_shape, nfilters, nunits)

            print(model.summary())
            dname = cvout.replace(".csv", "")
            b = 0
            if batch_size_ is None:
                b = len(x_test)
            else:
                b = batch_size_
            log_dir_ = ("./logs/cv%d_%s_#b%d_#e%d_#u%d_#f%d_" % (cv_,
                                                                 dname,
                                                                 b,
                                                                 num_epochs,
                                                                 nunits,
                                                                 nfilters))
            log_dir_ += time.strftime("%Y%m%d%H%M%S")

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

            model.fit(x_train, y_train,
                      epochs=num_epochs,
                      batch_size=b,
                      verbose=1,
                      validation_data=(x_test, y_test),
                      callbacks=callbacks_)

            bestmodel = load_model(model_output, custom_objects={"score": score})
            yrecalc = bestmodel.predict(x_train)
            for i in range(len(yrecalc)):
                recalc[train_keys[i]].extend(list(yrecalc[i]))

            ypred_test = bestmodel.predict(x_test)
            print("Test R2: %.4f" % (r2_score(y_test, ypred_test)))

            ypred = bestmodel.predict(x_val)
            # exp_pred_plot(y_val_, ypred[:,0])
            print("Validation R2: %.4f" % (r2_score(y_val, ypred)))
            for i in range(len(ypred)):
                predictions[val_keys[i]].extend(list(ypred[i]))

            """
            if fimpfile is not None:
                fimp = FeatureImportance(model, x_val, y_val, self.xheader)
                fires = fimp.Calculate(verbose=1)
                for key in fires.keys():
                    feat_imp[key]['mae'].extend(fires[key]['mae'])
                    feat_imp[key]['mse'].extend(fires[key]['mse'])
            """
            cv_ += 1

        fo = open(cvout, "w")
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
            x_train, y_train = self.GenData(train_keys)
            x_test, y_test = self.GenData(test_keys)

            model = build_model(self.nfeatures, nunits, ndense_layers)
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
            fo.write("%s,%.4f,%.4f\n" % (key, self.target[key], predictions[key]))
        fo.close()

    def GridSearch(self,
                   batch_size_,
                   num_epochs,
                   gmout="GridSearchResult"):

        train_keys, test_keys = TrainTestSplit(self.target, test_size_=0.20)
        print("Train set size: %d Test set size %d" % (len(train_keys),
                                                       len(test_keys)))

        # train_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
        # train_generator = self.DataGenerator(train_keys, batch_size_)

        x_train, y_train = self.GenData(train_keys)

        # This is unstable
        # test_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
        # test_generator = self.DataGenerator(test_keys, batch_size_)
        # This is more stable
        x_test, y_test = self.GenData(test_keys)

        ## PARAMETER DEFINITIONS
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
                model = build_dnn_resnet_model(self.nfeatures, c["nunits"],  c["ndense_layers"])

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
                          verbose=self.verbose,
                          validation_data=(x_test, y_test),
                          callbacks=callbacks_)

                bestmodel = load_model(model_output, custom_objects={"score": score})

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
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Batch size')
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
        print("ERROR! Please specify input table to train!")
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
                         args.nfilters,
                         args.nunits)
        else:
            nn.runcv(args.batch_size,
                     args.epochs,
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
