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

from keras import backend as K
# Some memory clean-up
K.clear_session()
from time import sleep
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import random
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import add, Input, Dense, Dropout, BatchNormalization, Conv1D, Activation, LeakyReLU
from keras.models import Sequential, Model
from keras.models import load_model
from keras import optimizers
from keras.utils import np_utils
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import ParameterGrid
import sys
from sys import argv
import time
import numpy as np
from math import ceil
import datetime
# import tqdm

from dnnresnet import dnn_resnet_layer


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../Base" % (dir_path))
from FeatureImportance import FeatureImportance, WriteFeatureImportance
from misc import GetKerasModel
from misc import RepeatedKFold
from misc import MDCTrainTestSplit
from misc import DISCTrainTestSplit
from misc import TrainTestSplit
from misc import ReadDescriptors, ReadTarget
from misc import LoadKerasModels
from keras_additional_loss_functions import rmse, score, np_score
import matplotlib.pyplot as plt

"""
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def score(y_true, y_pred):
    return K.log(K.mean(K.abs(y_true - y_pred), axis=-1))
"""


def example_build_model(nfeatures, nunits, ndense_layers):
    model = Sequential()
    # model.add(Conv1D(16, kernel_size=7, strides=1, activation="relu", input_shape=(nfeatures,)))
    # model.add(Dense(nunits, input_shape=(nfeatures,), activation='relu'))
    model.add(BatchNormalization(input_shape=(nfeatures,)))
    model.add(Dense(nunits, activation='relu'))
    model.add(Dropout(0.15))
    for i in range(ndense_layers):
        model.add(Dense(nunits, activation='relu'))
        model.add(Dropout(0.1))
    model.add(Dense(nunits, activation='relu'))
    model.add(Dense(1))
    # Compile model
    #model.compile(loss='mse',
    model.compile(loss=score,
                  optimizer=optimizers.Adam(lr=0.00005), metrics=['mse', 'mae', score])
    return model


def build_dnn_resnet_model(nfeatures, nunits, ndense_layers):
    """ResNet Version 1 Model builder

    # Arguments
        input_shape (tensor): shape of input tensor
        ndense_layers (int): number of resnet layers
        num_outputs (int): number of output to predict

    # Returns
        model (Model): Keras model instance
    """

    inputs = Input(shape=(nfeatures,))
    x = BatchNormalization()(inputs)
    x = Dense(nunits, activation='relu')(x)
    x = Dropout(0.1)(x)

    # Instantiate the stack of residual units
    for res_block in range(ndense_layers):
          y = dnn_resnet_layer(inputs=x, nunits=nunits, activation='relu')
          y = dnn_resnet_layer(inputs=y, nunits=nunits, activation=None)
          x = add([x, y])
          x = Activation('relu')(x)
          # num_filters *= 2

    outputs = Dense(1)(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=optimizers.Adam(), metrics=['mse', 'mae', score])
    return model


def build_gridsearch_model(nfeatures, ndense_layers, nunits, activation, dropout):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(nfeatures,)))

    for i in range(ndense_layers):
        model.add(Dense(nunits))
        if activation is "relu":
            model.add(Activation('relu'))
        elif activation is "leakyrelu":
            model.add(Activation('relu'))
        else:
            print("No activation %s found" % (activation))

        if dropout is "on":
            model.add(Dropout(0.1))

    model.add(Dense(1))
    # Compile model
    model.compile(loss='mse',
                  optimizer=optimizers.Adam(), metrics=['mse', 'mae'])
    # ,rmse])
    return model


class NNTrain(object):
    def __init__(self, X_raw, target, xheader=None):
        self.X_raw = X_raw
        self.xheader = xheader
        self.target = target
        self.nfeatures = len(list(X_raw.values())[0])
        try:
            self.ntargets = len(list(target.values())[0])
        except TypeError:
            self.ntargets = 1
        self.ntotobj = len(list(self.X_raw.keys()))
        print("#Inst.: %d  #Feat.: %d  #Tar.: %d" % (self.ntotobj,
                                                     self.nfeatures,
                                                     self.ntargets))
        self.verbose = 0
        self.batch_features = None
        self.batch_target = None

    def GenData(self, keys):
        batch_features = np.zeros((len(keys), self.nfeatures))
        batch_target = np.zeros((len(keys), self.ntargets))
        i = 0
        for key in keys:
            try:
                t = float(self.target[key])
                x = self.X_raw[key]
                batch_target[i] = np.copy(t)
                batch_features[i] = np.copy(x)
            except KeyError:
                print("Molecule %s not found" % (key))
            i += 1
        return batch_features, batch_target

    def makeMatrix(self, keys, nobjs, batch_features, batch_target):
        random.seed(datetime.datetime.now().microsecond)
        # random.shuffle(keys)
        for i in range(nobjs):
            # key = keys[i]
            key = keys[random.randint(0, len(keys)-1)]
            try:
                t = self.target[key]
                x = self.X_raw[key]
                batch_target[i] = np.copy(t)
                batch_features[i] = np.copy(x)
            except KeyError:
                print("Molecule %s not found" % (key))
        return 0
        # return np.array(batch_features), np.array(batch_target)

    def DataGenerator(self, keys, batch_size):
        self.batch_features = np.zeros((batch_size, self.nfeatures))
        self.batch_target = np.zeros((batch_size, self.ntargets))
        while True:
            self.makeMatrix(keys,
                            batch_size,
                            self.batch_features,
                            self.batch_target)
            yield self.batch_features, self.batch_target

    def makeMatrixiFromiTo(self,
                           keys,
                           i_from,
                           i_to,
                           batch_features,
                           batch_target):
        if i_to > len(keys):
            i_to = len(keys)
        for i in range(i_from, i_to):
            key = keys[i]
            try:
                t = self.target[key]
                x = self.X_raw[key]
                batch_target[i-i_from] = np.copy(t)
                batch_features[i-i_from] = np.copy(x)
            except KeyError:
                print("Molecule %s not found" % (key))
        # return batch_features, batch_target
        return 1

    def SampleDataset(self, keys, batch_size):
        batch_features = np.zeros((batch_size, self.nfeatures))
        batch_target = np.zeros((batch_size, self.ntargets))
        keys_length = len(keys)
        for i in range(0, keys_length, batch_size):
            self.makeMatrixiFromiTo(keys,
                                    i,
                                    i+batch_size,
                                    batch_features, batch_target)
            yield batch_features, batch_target

    # Public Methods
    def GridSearch(self,
                   batch_size_,
                   num_epochs,
                   gmout="GridSearchResult"):
        """
        Run GridSearch to find best parameters.
        """
        # train_keys, test_keys = MDCTrainTestSplit(self.target, 0)
        # train_keys, test_keys = DISCTrainTestSplit(self.target)
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

        # PARAMETER DEFINITIONS
        # simple architecture

        param = {}
        param["nunits"] = [200, 400, 800]
        param["ndense_layers"] = [2, 4, 6]
        # param["dropout"] = ["on", "off"]
        param["dropout"] = ["on"]
        # param["activation"] = ["relu", "leakyrelu"]
        param["activation"] = ["relu"]

        """
        # resnet architecture
        param = {}
        param["nunits"] = [200, 400,  600, 800]
        param["ndense_layers"] = [2, 4, 6]
        """

        all_combo = list(ParameterGrid(param))
        print("Evaluating %d combinations of parameters" % (len(all_combo)))

        already_computed_combo = []
        if Path(gmout).is_file():
            fi = open(gmout, "r")
            for line in fi:
                v = str.split(line.strip(), " ")
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
                """

                already_computed_combo.append(s)
            fi.close()

        for c in all_combo:
            # simple architecture
            s = ("%s-%s-%s-%s" % (c["nunits"],
                                  c["ndense_layers"],
                                  c["activation"],
                                  c["dropout"]))
            """
            # resnet architecture
            s = ("%s-%s" % (c["nunits"], c["ndense_layers"]))
            """

            if s in already_computed_combo:
                print("%s already computed... skip..." % (s))
            else:
                model = build_gridsearch_model(self.nfeatures,
                                               c["ndense_layers"],
                                               c["nunits"],
                                               c["activation"],
                                               c["dropout"])
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

    def simplerun(self,
                  batch_size_,
                  num_epochs,
                  ndense_layers,
                  nunits,
                  plot=0,
                  model_output=None):
        """
        Run a simple model...
        """
        # train_keys, test_keys = MDCTrainTestSplit(self.target, 0)
        # train_keys, test_keys = DISCTrainTestSplit(self.target)
        train_keys, test_keys = TrainTestSplit(self.target, test_size_=0.20)
        print("Train set size: %d Test set size %d" % (len(train_keys),
                                                       len(test_keys)))

        model = None
        if model_output is not None and Path(model_output).is_file():
            model = load_model(model_output)
        else:
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

        # train_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
        # train_generator = self.DataGenerator(train_keys, batch_size_)

        x_train, y_train = self.GenData(train_keys)

        # This is unstable
        # test_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
        # test_generator = self.DataGenerator(test_keys, batch_size_)
        # This is more stable
        x_test, y_test = self.GenData(test_keys)

        b = batch_size_
        log_dir_ = ("./logs/#b%d_#e%d_#u%d_#dl%d_" % (b,
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

        yrecalc_train = model.predict(x_train)

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
        ypred_test = model.predict(x_test)
        print("R2: %.4f Q2: %.4f MSE: %.4f" % (r2_score(y_train, yrecalc_train),
                                               r2_score(y_test, ypred_test),
                                               mse(y_test, ypred_test)))
        """
        print("R2: %.4f Q2: %.4f" % (r2_score(y_train, yrecalc_train),
                                     r2_score(y_test, ypred_test)))
        """
        if plot == 1:
            plt.scatter(y_test, ypred_test)
            plt.show()
        fo = open("%s_pred.csv" % time.strftime("%Y%m%d%H%M%S"), "w")
        for i in range(len(y_test)):
            fo.write("%s,%f,%f\n" % (test_keys[i],
                                     y_test[i],
                                     ypred_test[i]))
        fo.close()

        if model_output is not None:
            model.save(model_output)

    def runcv(self,
              batch_size_,
              num_epochs,
              ndense_layers,
              nunits,
              cvout,
              n_splits=5,
              n_repeats=10,
              mout=None,
              fimpfile=None):
        print("N. instances: %d" % (len(self.target)))

        mout_path = None
        if mout is not None:
            # Utilised to store the out path
            mout_path = Path("%s_%s" % (time.strftime("%Y%m%d%H%M%S"), mout))
            mout_path.mkdir()
            # Save the descriptor order
            f = open("%s/odesc_header.csv" % (str(mout_path.absolute())), "w")
            for dname in self.xheader:
                f.write("%s\n" % (dname))
            f.close()

        feat_imp = {}
        if fimpfile is not None:
            for feat_name in self.xheader:
                feat_imp[feat_name] = {'mae': [], 'mse': []}

        cv_ = 0
        predictions = {}
        recalc = {}
        for key in self.target.keys():
            predictions[key] = []
            recalc[key] = []

        for dataset_keys, test_keys in RepeatedKFold(n_splits,
                                                     n_repeats,
                                                     self.target):
            print("Dataset size: %d Test  size %d" % (len(dataset_keys),
                                                      len(test_keys)))

            sub_target = {}
            for key in dataset_keys:
                sub_target[key] = self.target[key]
            # ntobj = int(np.ceil(len(sub_target)*0.1))
            # train_keys, test_keys = MDCTrainTestSplit(sub_target, ntobj)
            # train_keys, test_keys = DISCTrainTestSplit(sub_target)
            train_keys, val_keys = TrainTestSplit(sub_target, test_size_=0.20)
            train_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
            train_generator = self.DataGenerator(train_keys, batch_size_)
            # x_train, y_train = self.GenData(train_keys)

            # test_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
            # test_generator = self.DataGenerator(test_keys, batch_size_)
            x_test, y_test = self.GenData(test_keys)
            x_val, y_val = self.GenData(val_keys)
            print("Train set size: %d Val set size %d" % (len(train_keys),
                                                          len(val_keys)))

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
            dname = cvout.replace(".csv", "")
            b = batch_size_
            log_dir_ = ("./logs/cv%d_%s_#b%d_#e%d_#u%d_#dl%d_" % (cv_,
                                                                  dname,
                                                                  b,
                                                                  num_epochs,
                                                                  nunits,
                                                                  ndense_layers))
            log_dir_ += time.strftime("%Y%m%d%H%M%S")

            """
            callbacks_ = [TensorBoard(log_dir=log_dir_,
                                      histogram_freq=0,
                                      write_graph=False,
                                      write_images=False)]
            """
            model_output = "%s/%d.h5" % (str(mout_path.absolute()), cv_)
            callbacks_ = [TensorBoard(log_dir=log_dir_,
                                      histogram_freq=0,
                                      write_graph=False,
                                      write_images=False),
                          ModelCheckpoint(model_output,
                                          monitor='val_score',
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

            model.fit_generator(train_generator,
                                steps_per_epoch=train_steps_per_epoch,
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
                      verbose=self.verbose,
                      validation_data=(x_test, y_test),
                      callbacks=callbacks_)
            """
            model_ = load_model(model_output, custom_objects={"score": score})

            y_recalc = []
            y_true_recalc = []
            for key in train_keys:
                row = np.array([self.X_raw[key]])
                p = model_.predict(row)
                y_recalc.extend(p)
                y_true_recalc.append(self.target[key])
                recalc[key].extend(list(p))

            ypred_test = model_.predict(x_test)
            ypred_val = model_.predict(x_val)
            r2 = r2_score(y_true_recalc, y_recalc)
            q2 = r2_score(y_test, ypred_test)
            tr2 = r2_score(y_val, ypred_val)
            print("Train R2: %.4f Test Q2: %.4f Val: R2: %.4f" % (r2, q2, tr2))

            # Store validation prediction
            for i in range(len(ypred_test)):
                predictions[test_keys[i]].extend(list(ypred_test[i]))

            # Store the cross validation model
            #if mout_path is not None:
            #    model.save("%s/%d.h5" % (str(mout_path.absolute()), cv_))

            if fimpfile is not None:
                fimp = FeatureImportance(model, x_test, y_test, self.xheader)
                fires = fimp.Calculate(verbose=1)
                for key in fires.keys():
                    feat_imp[key]['mae'].extend(fires[key]['mae'])
                    feat_imp[key]['mse'].extend(fires[key]['mse'])
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

        if fimpfile is not None:
            WriteFeatureImportance(feat_imp, fimpfile)

    def runloo(self, batch_size_, num_epochs, ndense_layers, nunits, cvout):
        """
        Only for small datasets!!!
        """
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
            # train_keys, test_keys = DISCTrainTestSplit(sub_target)
            train_keys, test_keys = TrainTestSplit(sub_target, )
            x_train, y_train = self.GenData(train_keys)
            x_test, y_test = self.GenData(test_keys)

            model = example_build_model(self.nfeatures, nunits, ndense_layers)
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
                      verbose=self.verbose,
                      validation_data=(x_test, y_test),
                      callbacks=callbacks_)

            predictions[val_key] = model.predict(x_test)[0]

        fo = open(cvout, "w")
        for key in predictions.keys():
            fo.write("%s,%.4f,%.4f\n" % (key,
                                         self.target[key],
                                         predictions[key]))
        fo.close()

    def continue_training(self,
                          batch_size_,
                          num_epochs,
                          models_path,
                          n_splits,
                          n_repeats,
                          mout,
                          cvout,
                          fimpfile):
        print("N. instances: %d" % (len(self.target)))
        models, odesc = LoadKerasModels(models_path, {"score": score})

        mout_path = None
        if mout is not None:
            # Utilised to store the out path
            mout_path = Path("%s_%s" % (time.strftime("%Y%m%d%H%M%S"), mout))
            mout_path.mkdir()
            # Save the descriptor order
            f = open("%s/odesc_header.csv" % (str(mout_path.absolute())), "w")
            for dname in self.xheader:
                f.write("%s\n" % (dname))
            f.close()

        feat_imp = {}
        if fimpfile is not None:
            for feat_name in self.xheader:
                feat_imp[feat_name] = {'mae': [], 'mse': []}

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
            # ntobj = int(np.ceil(len(sub_target)*0.1))
            # train_keys, test_keys = MDCTrainTestSplit(sub_target, ntobj)
            # train_keys, test_keys = DISCTrainTestSplit(sub_target)
            train_keys, test_keys = TrainTestSplit(sub_target, test_size_=0.20)
            train_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
            train_generator = self.DataGenerator(train_keys, batch_size_)
            # x_train, y_train = self.GenData(train_keys)

            # test_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
            # test_generator = self.DataGenerator(test_keys, batch_size_)
            x_test, y_test = self.GenData(test_keys)
            x_val, y_val = self.GenData(val_keys)
            print("Train set size: %d Test set size %d" % (len(train_keys),
                                                           len(test_keys)))

            model = example_build_model(self.nfeatures, nunits, ndense_layers)
            print(model.summary())
            dname = cvout.replace(".csv", "")
            b = batch_size_
            log_dir_ = ("./logs/cv%d_%s_#b%d_#e%d_#u%d_#dl%d_" % (cv_,
                                                                  dname,
                                                                  b,
                                                                  num_epochs,
                                                                  nunits,
                                                                  ndense_layers))
            log_dir_ += time.strftime("%Y%m%d%H%M%S")

            """
            callbacks_ = [TensorBoard(log_dir=log_dir_,
                                      histogram_freq=0,
                                      write_graph=False,
                                      write_images=False)]
            """
            model_output = "%s/%d.h5" % (str(mout_path.absolute()), cv_)
            callbacks_ = [TensorBoard(log_dir=log_dir_,
                                      histogram_freq=0,
                                      write_graph=False,
                                      write_images=False),
                          ModelCheckpoint(model_output,
                                          monitor='val_score',
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

            model.fit_generator(train_generator,
                                steps_per_epoch=train_steps_per_epoch,
                                epochs=num_epochs,
                                verbose=self.verbose,
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
            model_ = load_model(model_output, custom_objects={"score": score})

            y_recalc = []
            y_true_recalc = []
            for key in train_keys:
                row = np.array([self.X_raw[key]])
                p = model_.predict(row)
                y_recalc.extend(p)
                y_true_recalc.append(self.target[key])
                recalc[key].extend(list(p))

            ypred_test = model_.predict(x_test)
            ypred_val = model_.predict(x_val)
            r2 = r2_score(y_true_recalc, y_recalc)
            q2 = r2_score(y_test, ypred_test)
            tr2 = r2_score(y_val, ypred_val)
            print("Train R2: %.4f Test Q2: %.4f Val: R2: %.4f" % (r2, q2, tr2))

            # Store validation prediction
            for i in range(len(ypred_val)):
                predictions[val_keys[i]].extend(list(ypred_val[i]))

            # Store the cross validation model
            #if mout_path is not None:
            #    model.save("%s/%d.h5" % (str(mout_path.absolute()), cv_))

            if fimpfile is not None:
                fimp = FeatureImportance(model, x_val, y_val, self.xheader)
                fires = fimp.Calculate(verbose=1)
                for key in fires.keys():
                    feat_imp[key]['mae'].extend(fires[key]['mae'])
                    feat_imp[key]['mse'].extend(fires[key]['mse'])
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

        if fimpfile is not None:
            WriteFeatureImportance(feat_imp, fimpfile)
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xmatrix', default=None,
                        type=str, help='input data matrix')
    parser.add_argument('--ytarget', default=None,
                        type=str, help='input data matrix')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Batch size')
    parser.add_argument('--gsout', default=None, type=str,
                        help='Grid Search output')
    parser.add_argument('--n_splits', default=5, type=int,
                        help='Number of kfold splits')
    parser.add_argument('--n_repeats', default=20, type=int,
                        help='Number of repetitions')
    parser.add_argument('--cvout', default=None, type=str,
                        help='Cross validation output')
    parser.add_argument('--nunits', default=32, type=int,
                        help='Number of neurons')
    parser.add_argument('--ndense_layers', default=1, type=int,
                        help='Number of dense layers')
    parser.add_argument('--mout', default=None, type=str,
                        help='Model out path')
    parser.add_argument('--featimp', default=None, type=str,
                        help='Feature Importance file')
    parser.add_argument('--plot', default=0, type=int,
                        help='Plot output')

    args = parser.parse_args(argv[1:])
    if args.xmatrix is None or args.ytarget is None:
        print("ERROR! Please specify input table to train!")
        print("\n Usage: %s --xmatrix [input file] --ytarget [input file] --epochs [int] --batch_size [int]" % (argv[0]))
    else:
        # fix random seed for reproducibility
        seed = 981723
        np.random.seed(seed)
        random.seed(seed)
        """
        dx = None
        if ".csv" in args.xmatrix:
            dx = pd.read_csv(args.xmatrix)
        else:
            dx = pd.read_table(args.xmatrix, header=0)

        dy = None
        if ".csv" in args.ytarget:
            dy = pd.read_csv(args.ytarget)
        else:
            dy = pd.read_table(args.ytarget, header=0)


        X_raw = {}
        for row in dx.values:
            X_raw[row[0]] = row[1:].astype(float)

        target = {}
        for row in dy.values:
            target[row[0]] = row[1:].astype(float)
        """
        X_raw, nfeatures_, xheader = ReadDescriptors(args.xmatrix)
        target = ReadTarget(args.ytarget)
        """
        Necessary to speedup the data!

        for key in X_raw.keys():
            X_raw[key] = np.array(X_raw[key]).astype(float)
            target[key] = np.array(target[key]).astype(float)
        """
        nn = NNTrain(X_raw, target, xheader)
        nn.verbose = 0
        if args.cvout is None and args.gsout is None:
            nn.simplerun(args.batch_size,
                         args.epochs,
                         args.ndense_layers,
                         args.nunits,
                         plot=args.plot)
        elif args.cvout is not None and args.gsout is None:
            nn.runcv(args.batch_size,
                     args.epochs,
                     args.ndense_layers,
                     args.nunits,
                     args.cvout,
                     args.n_splits,
                     args.n_repeats,
                     args.mout,
                     args.featimp)
        else:
            nn.GridSearch(args.batch_size,
                          args.epochs,
                          args.gsout)


if __name__ == '__main__':
    main()
