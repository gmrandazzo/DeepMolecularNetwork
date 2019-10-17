#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import argparse
from SMILES2Matrix import SMILES2MX
from pathlib import Path
import numpy as np
import pandas as pd
import random
from datetime import datetime
from keras import optimizers
from keras.callbacks import Callback, EarlyStopping, TensorBoard
from keras.models import Model, Input, Sequential
from keras.layers import LSTM, TimeDistributed, Conv2D, Flatten, Dense, Dropout, Activation, MaxPooling2D, ZeroPadding2D, BatchNormalization, AveragePooling2D, Concatenate
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
from MDC import MDC
from misc import ReadDescriptors, ReadTarget


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def get_model():
    model_build_directory = Path('.')
    sys.path.append("%s" % (str(model_build_directory.absolute())))
    from model import build_model
    return build_model


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
    #m1 = MaxPooling2D(pool_size=(1, 2))(m1)

    # Add 4 time the same convolution
    for i in range(4):
        m1 = Conv2D(nfilters,
                    kernel_size=(1, 6),
                    strides=(1, 1))(m1)
        m1 = Activation('relu')(m1)

    m1 = AveragePooling2D(pool_size=(1, 3))(m1)
    #m1 = MaxPooling2D(pool_size=(1, 3))(m1)
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
    m2 = Dense(nunits)(m2)
    m2 = Activation("relu")(m2)
    m2 = Dense(nunits)(m2)
    m2 = Activation("relu")(m2)

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
    model.add(Dense(nunits))
    model.add(Activation('relu'))
    model.add(Dense(nunits))
    model.add(Activation('relu'))
    model.add(Dense(nunits))
    model.add(Activation('relu'))
    model.add(Dense(nunits))
    model.add(Activation('relu'))
    model.add(Dense(1))
    # Compile model
    # model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    model.compile(optimizer=optimizers.Adam(lr=0.0001),
                  loss='mse', metrics=['mse', 'mae'])
    # model.compile(loss='mse', optimizer='rmsprop', metrics=['mse', 'mae'])
    # model.compile(loss='mse', optimizer='nadam', metrics=['mse', 'mae'])
    # , rmse])
    return model


def MDCTrainTestSplit(dict_target, n_objects=0):
    y = list(dict_target.values())
    keys = list(dict_target.keys())
    dmx = [[1. for j in range(len(y))] for i in range(len(y))]
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            dmx[i][j] = dmx[j][i] = np.sqrt((y[i]-y[j])**2)
    csel = MDC(dmx, n_objects)
    test_keys = []
    for indx in csel.select():
        test_keys.append(keys[indx])
    train_keys = []
    for key in dict_target.keys():
        if key not in test_keys:
            train_keys.append(key)
        else:
            continue
    return train_keys, test_keys


def RepeatedKFold(n_splits, n_repeats, dict_target):
    indexes = [i for i in range(len(dict_target))]
    keys = list(dict_target.keys())
    for i in range(n_repeats):
        random.shuffle(indexes)
        for j in range(n_splits):
            train_keys = []
            test_keys = []
            for k in range(len(dict_target)):
                if k % n_splits != j:
                    train_keys.append(keys[indexes[k]])
                else:
                    test_keys.append(keys[indexes[k]])
            yield (train_keys,
                   test_keys)


class NNTrain(object):
    def __init__(self, smiles_list, target, dx=None, n_descs=None):
        self.smi_padding = 300
        self.X, self.input_shape = self.ReadSMILES(smiles_list)
        self.target = target
        self.dx = dx
        self.n_descs = n_descs
        self.verbose = 1

    def ReadSMILES(self, smiles_list):
        s2m = SMILES2MX(self.smi_padding)
        f = open(smiles_list, "r")
        X = {}
        for line in f:
            v = str.split(line.strip(), "\t")
            print("Parsing: %s" % (v[1]))
            X[v[1]] = np.array(s2m.smi2mx(v[0]))
        f.close()
        return X, s2m.getshape()

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
                # batch_x = np.array(batch_x)[:, :, :, np.newaxis]
                batch_x = np.array(batch_x)[np.newaxis, :, :, :, np.newaxis]
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
                  nunits):
        print("N. instances: %d" % (len(self.target)))
        train_keys, test_keys = MDCTrainTestSplit(self.target, 0)
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
                  verbose=1,
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
            train_keys, test_keys = MDCTrainTestSplit(sub_target, ntobj)

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
                      verbose=1,
                      validation_data=(x_test, y_test),
                      callbacks=callbacks_)

            yrecalc = model.predict(x_train)
            for i in range(len(yrecalc)):
                recalc[train_keys[i]].extend(list(yrecalc[i]))

            ypred_test = model.predict(x_test)
            print("Test R2: %.4f" % (r2_score(y_test, ypred_test)))

            ypred = model.predict(x_val)
            # exp_pred_plot(y_val_, ypred[:,0])
            print("Validation R2: %.4f" % (r2_score(y_val, ypred)))
            for i in range(len(ypred)):
                predictions[val_keys[i]].extend(list(ypred[i]))

            if mout_path is not None:
                model.save("%s/%d.h5" % (str(mout_path.absolute()), cv_))

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

            train_keys, test_keys = MDCTrainTestSplit(sub_target, 0)
            x_train, y_train = self.GenData(train_keys)
            x_test, y_test = self.GenData(test_keys)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', default=None,
                        type=str, help='input smiles data')
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
    if args.smiles is None or args.ytarget is None:
        print("ERROR! Please specify input table to train!")
        print("\n Usage: %s --smiles [input file] --xmatrix [csv descriptors] --ytarget [input file] --epochs [int] --batch_size [int]\n\n" % (argv[0]))
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

        nn = NNTrain(args.smiles, target, dx, n_descs)
        nn.verbose = 1
        if args.cvout is None:
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
