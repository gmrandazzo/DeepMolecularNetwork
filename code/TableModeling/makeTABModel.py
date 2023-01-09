#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details
import os
import sys
from sys import argv
import argparse
from pathlib import Path
import numpy as np
import random
import time

import tensorflow as tf
if int(tf.__version__[0]) > 1:
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.callbacks import TensorBoard

    from tensorflow.keras.models import Model
    from tensorflow.keras.models import Sequential

    from tensorflow.keras.layers import add
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import LeakyReLU
    from tensorflow.keras import optimizers
else:
    from keras import backend as K
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import TensorBoard

    from keras.models import Model
    from keras.models import Sequential

    from keras.layers import add
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    from keras.layers import LeakyReLU
    from keras import optimizers

from sklearn.model_selection import ParameterGrid

import time
from math import ceil
import datetime
from dnnresnet import dnn_resnet_layer


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../Base" % (dir_path))
from FeatureImportance import FeatureImportance, WriteFeatureImportance

from modelhelpers import GetTrainTestFnc
from modelhelpers import GetValidationFnc
from modelhelpers import GetKerasModel
from modelhelpers import GetLoadModelFnc
from modelhelpers import LoadKerasModels

from modelvalidation import RepeatedKFold
from modelvalidation import MDCTrainTestSplit
from modelvalidation import DISCTrainTestSplit
from modelvalidation import TrainTestSplit

from numpy_loss_functions import RSQ, MSE, MAE

from dmnnio import WriteCrossValidationOutput
from dmnnio import ReadDescriptors
from dmnnio import ReadTarget


def example_build_model(nfeatures, nunits, ndense_layers, ntargets):
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
    model.add(Dense(ntargets))
    # Compile model
    # model.compile(loss='mse',
    model.compile(loss='mse',
                  optimizer=optimizers.Adam(lr=0.00005),
                  metrics=['mse', 'mae'])
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
    model.compile(loss='mse',
                  optimizer=optimizers.Adam(),
                  metrics=['mse', 'mae'])
    return model


def build_gridsearch_model(nfeatures,
                           ndense_layers,
                           nunits,
                           activation,
                           dropout):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(nfeatures,)))

    for i in range(ndense_layers):
        model.add(Dense(nunits))
        if activation == "relu":
            model.add(Activation('relu'))
        elif activation == "leakyrelu":
            model.add(LeakyReLU())
        else:
            print("No activation %s found" % (activation))

        if dropout == "on":
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
                t = self.target[key]
                x = self.X_raw[key]
                batch_target[i] = np.copy(t)
                batch_features[i] = np.copy(x)
            except KeyError:
                print("Molecule %s not found" % (key))
            i += 1
        return batch_features, batch_target

    def makeMatrix(self,
                   keys,
                   nobjs,
                   batch_features,
                   batch_target,
                   batch_mode=0):
        random.seed(datetime.datetime.now().microsecond)
        # random.shuffle(keys)
        if batch_mode == 0:
            for i in range(nobjs):
                key = keys[random.randint(0, len(keys)-1)]
                try:
                    t = self.target[key]
                    x = self.X_raw[key]
                    batch_target[i] = np.copy(t)
                    batch_features[i] = np.copy(x)
                except KeyError:
                    print("Molecule %s not found" % (key))
        else:
            btask = [0 for i in range(self.ntargets)]
            for i in range(nobjs):
                key = None
                key_size = len(keys)-1
                j = None
                if sum(btask) == self.ntargets:
                    for i in range(len(btask)):
                        btask[i] = 0
                    j = random.randint(0, self.ntargets-1)
                else:
                    j = btask.index(min(btask))
        
                while True:
                    key = keys[random.randint(0, key_size)]
                    # Check for missing values. In case of, continue!
                    if int(np.fabs(self.target[key][j]-9999.)) == 0:
                        continue
                    else:
                        break
                try:
                    t = self.target[key]
                    x = self.X_raw[key]
                    batch_target[i] = np.copy(t)
                    batch_features[i] = np.copy(x)
                except KeyError:
                    print("Molecule %s not found" % (key))
        return 0
        # return np.array(batch_features), np.array(batch_target)

    def DataGenerator(self, keys, batch_size, batch_mode=0):
        """
        batch_mode=0 -> Random batch
        batch_mode=1 -> Balanced batch
        """
        self.batch_features = np.zeros((batch_size, self.nfeatures))
        self.batch_target = np.zeros((batch_size, self.ntargets))
        while True:
            self.makeMatrix(keys,
                            batch_size,
                            self.batch_features,
                            self.batch_target,
                            batch_mode)
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
    
    def makePrediction(self, model, keys):
        predictions = {}
        x = []
        for key in keys:
            x.append(self.X_raw[key])
        x = np.array(x)
        xp = model.predict(x)
        for i in range(len(keys)):
            key = keys[i]
            predictions[key] = xp[i]

        return predictions
        
    # Public Methods
    def GridSearch(self,
                   batch_size_,
                   num_epochs,
                   random_state,
                   gmout="GridSearchResult"):
        """
        Run GridSearch to find best parameters.
        """
        # train_keys, test_keys = MDCTrainTestSplit(self.target, 0)
        # train_keys, test_keys = DISCTrainTestSplit(self.target)
        train_keys, test_keys = TrainTestSplit(list(self.target.keys()),
                                               test_size=0.20,
                                               random_state=random_state)
        print("Train set size: %d Test set size %d" % (len(train_keys),
                                                       len(test_keys)))

        # train_steps_per_epoch = ceil(len(train_keys)/loat(batch_size_))
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

                bestmodel = GetLoadModelFnc()(model_output)

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
                r2 = RSQ(y_train, yrecalc_train)
                mse_train = MSE(y_train, yrecalc_train)
                mae_train = MAE(y_train, yrecalc_train)
                q2 = RSQ(y_test, ypred_test)
                mse_test = MSE(y_test, ypred_test)
                mae_test = MAE(y_test, ypred_test)
                print("R2: %.4f  Q2: %.4f" % (r2, q2))

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
                fo.write("%d %d %f %f %f %f %f %f\n" % (c["nunits"],
                                                        c["ndense_layers"],
                                                        mse_train,
                                                        mae_train,
                                                        r2,
                                                        mse_test,
                                                        mae_test,
                                                        q2))
                fo.close()

    def simplerun(self,
                  batch_size_,
                  batch_mode_,
                  num_epochs,
                  ndense_layers,
                  nunits,
                  random_state,
                  model_output=None):
        """
        Run a simple model...
        """
        # train_keys, test_keys = MDCTrainTestSplit(self.target, 0)
        # train_keys, test_keys = DISCTrainTestSplit(self.target)

        ttfn = GetTrainTestFnc()
        if ttfn is None:
            ttfn = TrainTestSplit
        else:
            print("Using custom train/test split function")

        train_keys, test_keys = ttfn(list(self.target.keys()),
                                     test_size=0.20,
                                     random_state=random_state)

        print("Train set size: %d Test set size %d" % (len(train_keys),
                                                       len(test_keys)))

        model = None
        if model_output is not None and Path(model_output).is_file():
            model = GetLoadModelFnc()(model_output)
        else:
            model_ = GetKerasModel()
            if model_ is None:
                model = example_build_model(self.nfeatures,
                                            nunits,
                                            ndense_layers,
                                            self.ntargets)
            else:
                model = model_(self.nfeatures,
                               nunits,
                               ndense_layers)
        print(model.summary())

        train_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
        train_generator = self.DataGenerator(train_keys,
                                             batch_size_,
                                             batch_mode_)

        #x_train, y_train = self.GenData(train_keys)

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

        y_recalc_train = self.makePrediction(model, train_keys)
        y_pred_test = self.makePrediction(model, test_keys)
       
        ytrain_recalc = []
        ytrain_true = []
        for key in train_keys:
            ytrain_recalc.append(y_recalc_train[key])
            ytrain_true.append(self.target[key])
    
        ytest_pred = []
        ytest_true = []
        for key in test_keys:
            ytest_pred.append(y_pred_test[key])
            ytest_true.append(self.target[key])
    
        print("R2: %.4f Q2: %.4f MSE: %.4f" % (RSQ(ytrain_true, ytrain_recalc),
                                               RSQ(ytest_pred, ytest_true),
                                               MSE(ytest_pred, ytest_true)))

        fo = open("%s_pred.csv" % time.strftime("%Y%m%d%H%M%S"), "w")
        for i in range(len(ytest_true)):
            fo.write("%s" % (test_keys[i]))
            for j in range(len(ytest_true[i])):
                fo.write(",%f,%f" % (ytest_true[i][j], ytest_pred[i][j]))
            fo.write("\n")
        fo.close()

        if model_output is not None:
            model.save(model_output)

    def runcv(self,
              batch_size_,
              batch_mode_,
              num_epochs,
              ndense_layers,
              nunits,
              cvout,
              n_splits=5,
              n_repeats=10,
              random_state=None,
              mout=None,
              fimpfile=None):
        print("N. instances: %d" % (len(self.target)))

        mout_path = None
        if mout is not None:
            # Utilised to store the out path
            # mout_path = Path("%s_%s" % (time.strftime("%Y%m%d%H%M%S"), mout))
            mout_path = Path(mout)
        else:
            # Utilised to store the out path
            mout_path = Path("%s_model" % (time.strftime("%Y%m%d%H%M%S")))

        last_model = None
        if mout_path.exists() is True:
            # Find the last model and restart the calculation from it.
            p = Path(mout_path).glob('**/*.h5')
            # getonlzfile numbers
            mids = [int(x.stem) for x in p if x.is_file()]
            if len(mids) > 0:
                # Restart from here...
                last_model = max(mids)
            else:
                last_model = None
        else:
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

        valfn = GetValidationFnc()
        if valfn is None:
            valfn = RepeatedKFold(list(self.target.keys()),
                                  n_splits,
                                  n_repeats,
                                  random_state=random_state,
                                  test_size=0.2)
        else:
            print("Using custom validation split function")
            valfn = valfn(list(self.target.keys()))

        for train_keys, val_keys, test_keys in valfn:
            # Some memory clean-up
            K.clear_session()
            train_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
            train_generator = self.DataGenerator(train_keys,
                                                 batch_size_,
                                                 batch_mode_)
            # x_train, y_train = self.GenData(train_keys)
            # test_steps_per_epoch = ceil(len(train_keys)/float(batch_size_))
            # test_generator = self.DataGenerator(test_keys, batch_size_)
            x_test, y_test = self.GenData(test_keys)
            x_val, y_val = self.GenData(val_keys)
            print("Train set size: %d Val set size %d Test set size: %d" % (len(train_keys),
                                                                            len(val_keys),
                                                                            len(test_keys)))
            model_output = "%s/%d.h5" % (str(mout_path.absolute()), cv_)

            if last_model is None:
                model = None
                model_ = GetKerasModel()
                if model_ is None:
                    model = example_build_model(self.nfeatures,
                                                nunits,
                                                ndense_layers,
                                                self.ntargets)
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
                callbacks_ = [TensorBoard(log_dir=log_dir_,
                                          histogram_freq=0,
                                          write_graph=False,
                                          write_images=False),
                              ModelCheckpoint(model_output,
                                              monitor='val_loss',
                                              verbose=0,
                                              save_best_only=True)]

                model.fit_generator(train_generator,
                                    steps_per_epoch=train_steps_per_epoch,
                                    epochs=num_epochs,
                                    verbose=self.verbose,
                                    validation_data=(x_val, y_val),
                                    # validation_data=test_generator,
                                    # validation_steps=test_steps_per_epoch,
                                    callbacks=callbacks_)
            else:
                if last_model-1 == cv_:
                    last_model = None

            model_ = GetLoadModelFnc()(model_output)


            y_recalc_train = self.makePrediction(model_, train_keys)
            y_pred_val = self.makePrediction(model_, val_keys)
            y_pred_test = self.makePrediction(model_, test_keys)
           
            y_recalc = []
            y_true_recalc = []
            for key in train_keys:
                y_recalc.append(y_recalc_train[key])
                y_true_recalc.append(self.target[key])
                recalc[key].append(y_recalc_train[key])

            ypred_val = []
            ytrue_val = []
            for key in val_keys:
                ypred_val.append(y_pred_val[key])
                ytrue_val.append(self.target[key])
            
            ypred_test = []
            ytrue_test = []
            for key in test_keys:
                ypred_test.append(y_pred_test[key])
                ytrue_test.append(self.target[key])
                # Store validation prediction
                predictions[key].append(y_pred_test[key])

            r2 = RSQ(y_true_recalc, y_recalc)
            q2 = RSQ(ytrue_test, ypred_test)
            tr2 = RSQ(ytrue_val, ypred_val)
            print("Train R2: %.4f Test Q2: %.4f Val: R2: %.4f\n" % (r2,
                                                                    q2,
                                                                    tr2))

            # Store the cross validation model
            # if mout_path is not None:
            #    model.save("%s/%d.h5" % (str(mout_path.absolute()), cv_))

            if fimpfile is not None:
                fimp = FeatureImportance(model, x_test, y_test, self.xheader)
                fires = fimp.Calculate(verbose=1)
                for key in fires.keys():
                    feat_imp[key]['mae'].extend(fires[key]['mae'])
                    feat_imp[key]['mse'].extend(fires[key]['mse'])
            cv_ += 1

        WriteCrossValidationOutput(cvout, self.target, predictions, recalc)

        if fimpfile is not None:
            WriteFeatureImportance(feat_imp, fimpfile)


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
    parser.add_argument('--batch_mode', default=0, type=int,
                        help='Batch mode: 0=random; 1=balanced')
    parser.add_argument('--random_state', default=123458976, type=int,
                        help='Random state')
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
    parser.add_argument('--verbose', default=0, type=int,
                        help='Set verbosity 0, 1, 2')
    parser.add_argument('--mtype', default=0, type=int,
                        help='Regression: 0, classification: 1')

    args = parser.parse_args(argv[1:])
    if args.xmatrix is None or args.ytarget is None or args.batch_size is None:
        print("ERROR! Please specify input table to train!")
        print("\n Usage: %s --xmatrix [input file] --ytarget [input file] --epochs [int] --batch_size [int]" % (argv[0]))
    else:
        # fix random seed for reproducibility
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        X_raw, nfeatures_, xheader = ReadDescriptors(args.xmatrix)
        target = ReadTarget(args.ytarget)
        nn = NNTrain(X_raw, target, xheader)
        nn.verbose = args.verbose
        if args.cvout is None and args.gsout is None:
            nn.simplerun(args.batch_size,
                         args.batch_mode,
                         args.epochs,
                         args.ndense_layers,
                         args.nunits,
                         args.random_state,
                         args.mout)
        elif args.cvout is not None and args.gsout is None:
            nn.runcv(args.batch_size,
                     args.batch_mode,
                     args.epochs,
                     args.ndense_layers,
                     args.nunits,
                     args.cvout,
                     args.n_splits,
                     args.n_repeats,
                     args.random_state,
                     args.mout,
                     args.featimp)
        else:
            nn.GridSearch(args.batch_size,
                          args.epochs,
                          args.gsout)


if __name__ == '__main__':
    main()
