#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import tensorflow as tf
if int(tf.__version__[0]) > 1:
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import TensorBoard
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.utils import plot_model
else:
    from keras import backend as K
    from keras.callbacks import TensorBoard
    from keras.callbacks import ModelCheckpoint
    from keras.utils import plot_model

import argparse
from model_builder import *
import numpy as np
from pathlib import Path
import random
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from Voxel import LoadVoxelDatabase
from datetime import datetime
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../Base" % (dir_path))
# from FeatureImportance import FeatureImportance, WriteFeatureImportance
from modelhelpers import GetKerasModel
from modelhelpers import GetLoadModelFnc
from modelhelpers import GetTrainTestFnc

from dmnnio import ReadDescriptors
from dmnnio import ReadTarget
from dmnnio import WriteCrossValidationOutput
from modelvalidation import TrainTestSplit
from modelvalidation import RepeatedKFold
from modelvalidation import CVGroupRead
from modelvalidation import StaticGroupCV
from modelvalidation import RepeatedStratifiedCV


class AIModel(object):
    def __init__(self, csv_target, db, csv_descriptors=None):
        self.voxels = LoadVoxelDatabase(db)
        print("Max Conformations %d " % (len(list(self.voxels.values())[0])))
        if csv_descriptors is not None:
            self.other_descriptors, self.nfeatures, self.header = ReadDescriptors(csv_descriptors)
        else:
            self.other_descriptors = None
            self.nfeatures = 0
            self.header = None
        self.target = ReadTarget(csv_target)
        self.input_shape, self.conv3d_chtype = self.getInputVoxelShape()
        # voxel conformation selection statistics
        # to check how many times the voxel conformation
        # is selected
        self.statvoxconf = {}
        for key in self.voxels.keys():
            max_conf = len(self.voxels[key])
            # max_rot = len(self.voxels[key][0])
            self.statvoxconf[key] = [[0 for j in range(len(self.voxels[key][i]))] for i in range(max_conf)]
        return

    def getInputVoxelShape(self):
        input_shape = []
        for voxel in self.voxels.values():
            voxel = np.array(voxel)
            # print(voxel.shape)
            # (12, 50, 25, 25, 25)
            # 12 = conformations
            # 50 = rotations
            # 25, 25, 25 = single channel voxels
            # Doble channel example!
            # (1, 20, 2, 50, 50, 50)
            # 1 = conformation
            # 20 = rotations
            # 2, 50, 50, 50 = two channel voxels
            #print(voxel.shape)
            if len(input_shape) is 0:
                input_shape = voxel[0][0].shape
                #input_shape = voxel[0].shape
            else:
                for i in range(len(input_shape)):
                    if voxel[0][0].shape[i] != input_shape[i]:
                    #if voxel[0].shape[i] != input_shape[i]:
                        print("Input shape Error!")
                        assert voxel[0][0].shape[i] == input_shape[i]
                        #assert voxel[0].shape[i] == input_shape[i]
                    else:
                        continue
        input_shape = list(input_shape)
        if len(input_shape) == 3:
            # Channel last
            input_shape.append(1)
            conv3d_chtype = 'channels_last'
        else:
            conv3d_chtype = 'channels_first'
        print("Input shape: {}".format(input_shape))
        print("Channel type: %s" % (conv3d_chtype))
        return tuple(input_shape), conv3d_chtype

    def GetAvailableKeys(self):
        allkeys = list(self.voxels.keys())
        allkeys.extend(list(self.target.keys()))
        if self.other_descriptors is not None:
            allkeys.extend(list(self.other_descriptors.keys()))
        allkeys = list(set(allkeys))
        keys = []
        if self.other_descriptors is not None:
            for key in allkeys:
                if key in self.voxels.keys() and key in self.other_descriptors.keys() and key in self.target.keys():
                    keys.append(key)
                else:
                    continue
        else:
            for key in allkeys:
                if key in self.voxels.keys() and key in self.target.keys():
                    keys.append(key)
                else:
                    continue
                
    def makeDataset(self, keys, nrotations):
        """
        Create a dataset giving:
        keys: key names
        nrotations: number of rotations for data augmentation
        return X and y
        """
        batch_features = None
        if "last" in self.conv3d_chtype:
            batch_features = np.array([]).reshape(0,
                                                  self.input_shape[0],
                                                  self.input_shape[1],
                                                  self.input_shape[2])
        else:
            batch_features = np.array([]).reshape(0,
                                                  self.input_shape[0],
                                                  self.input_shape[1],
                                                  self.input_shape[2],
                                                  self.input_shape[3])
        batch_desc_features = None
        if self.other_descriptors is not None:
            batch_desc_features = np.array([]).reshape(0, self.nfeatures)
        else:
            batch_desc_features = None

        batch_target = []
        selected_confs = []
        for i in range(nrotations):
            for key in keys:
                max_conf = len(self.voxels[key])
                max_rot = len(self.voxels[key][0])
                # could be float or array
                target_val = self.target[key]
                rconf = random.randint(0, max_conf-1)
                rrot = random.randint(0, max_rot-1)
                while True:
                    # print("BAO")
                    if "%s-%d-%d" % (key, rconf, rrot) in selected_confs:
                        rconf = random.randint(0, max_conf-1)
                        rrot = random.randint(0, max_rot-1)
                    else:
                        break
                # print("END")
                selected_confs.append("%s-%d-%d" % (key, rconf, rrot))
                self.statvoxconf[key][rconf][rrot] += 1
                voxel = self.voxels[key][rconf][rrot]
                batch_features = np.vstack([batch_features, [voxel]])
                batch_target.append(target_val)
                if self.other_descriptors is not None:
                    batch_desc_features = np.vstack([batch_desc_features, self.other_descriptors[key]])

        if "last" in self.conv3d_chtype:
            batch_features = batch_features.reshape(batch_features.shape[0],
                                                    batch_features.shape[1],
                                                    batch_features.shape[2],
                                                    batch_features.shape[3], 1)

        if batch_desc_features is not None:
            """
            Double branch neural network: CNN+Descriptors
            """
            return [batch_features, batch_desc_features], np.array(batch_target)
        else:
            """
            Single CNN
            """
            return batch_features, np.array(batch_target)

    def VoxelTrainGenerator(self, keys, n_rot_repetitions=1):
        """
        Keras voxel data augmentation
        """
        i = 0
        while True:
            random.seed(datetime.now().microsecond)
            #startTime = time.time()
            X, y = self.makeDataset(keys, n_rot_repetitions)
            #elapsedTime = time.time() - startTime
            # print('[VoxelTrainGenerator] {} finished in {} ms'.format(i, int(elapsedTime * 1000)))
            #print("Dataset no. %d\n\n" % (i))
            i += 1
            yield X, y

    def VoxelTestSetGenerator(self, keys, random_voxel_rotations):
        X, y = self.makeDataset(keys, random_voxel_rotations)
        return X, y

    def FeaturePermutation(self, batch_features, fid=-1):
        batch_perm = batch_features.copy()
        if fid < self.nfeatures:
            # permutate csv descriptor
            np.random.shuffle(batch_perm[1][:, fid])
        else:
            # permuitate voxel info
            np.random.shuffle(batch_perm[0][:])
        return batch_perm


def simplerun(db,
              csv_target,
              csv_descriptors,
              num_epochs,
              n_rot_train,
              train_steps_per_epoch_,
              n_rotation_test,
              test_steps_per_epoch_,
              ndense_layers,
              nunits,
              nfilters,
              random_state,
              outmodel=None,
              fcvgroup=None,
              tid=None):
    # Load the dataset
    ai = AIModel(csv_target, db, csv_descriptors)
    available_keys = ai.GetAvailableKeys()
    train_keys = None
    test_keys = None

    if fcvgroup is not None:
        cvgroups = CVGroupRead(fcvgroup)
        tkey = None
        if tid is not None:
            tkey = int(tid)
            print(tkey)
        else:
            tkey = random.choice(list(cvgroups.keys()))
        print(cvgroups[tkey])
        test_keys = cvgroups[tkey]
        train_keys = []
        for key in cvgroups.keys():
            if key == tkey:
                continue
            else:
                train_keys.textened(cvgroups[key])
    else:
        ttfn = GetTrainTestFnc()
        if ttfn is None:
            ttfn = TrainTestSplit
        else:
            print("Using custom train/test split function")
        train_keys, test_keys = ttfn(list(self.target.keys()),
                                test_size=0.20,
                                random_state=random_state)
        

    print("Trainin set size: %d Validation set size %d" % (len(train_keys),
                                                            len(test_keys)))

    train_generator = ai.VoxelTrainGenerator(train_keys, n_rot_train)

    print(train_keys)
    print(test_keys)

    model = None
    model_ = GetKerasModel()
    if ai.other_descriptors is None:
        if model_ is None:
            model = build_model(ai.conv3d_chtype,
                                ai.input_shape,
                                ndense_layers,
                                nunits,
                                nfilters)
        else:
            model = model_(ai.conv3d_chtype,
                           ai.input_shape,
                           ndense_layers,
                           nunits,
                           nfilters)
        # model = build_fcn_model(ai.conv3d_chtype, ai.input_shape, ndense_layers, nunits, nfilters)
        # model = model_scirep(ai.conv3d_chtype, ai.input_shape, ndense_layers, nunits, nfilters)
        # model = ResNetModel(ai.input_shape)
        print(model.summary())
    else:
        if model_ is None:
            model = build_2DData_model(ai.conv3d_chtype,
                                       ai.input_shape,
                                       ai.nfeatures,
                                       ndense_layers,
                                       nunits,
                                       nfilters)
        else:
            model = model_(ai.conv3d_chtype,
                           ai.input_shape,
                           ai.nfeatures,
                           ndense_layers,
                           nunits,
                           nfilters)

        """
        for l in model.layers[0].layers:
            print(l.summary())
        """
        print("Total Summary")
        print(model.summary())
    plot_model(model, to_file="model.png", show_shapes=True)

    dname = os.path.basename(csv_target).replace(".csv", "")
    dname += os.path.basename(db)
    log_dir_ = ("./logs/%s_%d_#rot%d_#f%d_#dl%d_#u%d_" % (dname,
                                                          num_epochs,
                                                          train_steps_per_epoch_,
                                                          nfilters,
                                                          ndense_layers,
                                                          nunits))
    log_dir_ += time.strftime("%Y%m%d%H%M%S")
    callbacks_ = [TensorBoard(log_dir=log_dir_,
                              histogram_freq=0,
                              write_graph=False,
                              write_images=False)]
    """
    ,
                  EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=3,
                                verbose=0,
                                mode='auto')
    """

    test_generator = ai.VoxelTrainGenerator(test_keys, n_rot_train)

    model.fit_generator(train_generator,
                        epochs=num_epochs,
                        steps_per_epoch=train_steps_per_epoch_,
                        verbose=1,
                        # max_queue_size=2,
                        # workers=0,
                        # validation_data=(x_test_, y_test_),
                        validation_data=test_generator,
                        validation_steps=test_steps_per_epoch_,
                        # nb_val_samples=x_test.shape[0],
                        callbacks=callbacks_,
                        use_multiprocessing=True)

    x_test_, y_test_ = ai.VoxelTestSetGenerator(test_keys, n_rotation_test)
    y_pred_ = model.predict(x_test_)
    print("Test R2: %.4f" % (r2_score(y_test_, y_pred_)))

    fo = open("statconf.csv", "w")
    for key in ai.statvoxconf.keys():
        fo.write("%s," % (key))
        for i in range(len(ai.statvoxconf[key])):
            for j in range(len(ai.statvoxconf[key][i])):
                fo.write("%d," % (ai.statvoxconf[key][i][j]))
        fo.write("\n")
    fo.close()
    # score = model.evaluate(x_test_, y_test_, verbose=0)
    # print(score)

    if outmodel is not None:
        model.save(outmodel)


def cv(db,
       csv_target,
       csv_descriptors,
       n_splits_,
       n_repeats_,
       num_epochs,
       n_rot_train,
       train_steps_per_epoch_,
       n_rot_test,
       test_steps_per_epoch_,
       ndense_layers,
       nunits,
       nfilters,
       random_state,
       cvout=None,
       fcvgroup=None,
       featimp_out=None,
       y_recalc=False,
       mout=None):
    # Load the dataset
    ai = AIModel(csv_target, db, csv_descriptors)
    available_keys = ai.GetAvailableKeys()
    print("N. instances: %d" % (len(ai.target)))
    predictions = dict()
    valpredictions = dict()
    for key in ai.target.keys():
        predictions[key] = []
        valpredictions[key] = []

    feat_imp = None
    feat_imp_iterations = 20

    if featimp_out is not None:
        # feature importance list for csv descriptors
        if ai.other_descriptors is not None:
            feat_imp = [[] for p in range(ai.nfeatures)]
            # charge voxel descriptor
            feat_imp.append([])
        else:
            print("Feature Importance calculation: DISABLED")

    # Create directory to store all the models
    mout_path = None
    if mout is not None:
        # Utilised to store the out path
        mout_path = Path("%s_%s" % (time.strftime("%Y%m%d%H%M%S"), mout))
        mout_path.mkdir(exist_ok=True, parents=True)
        if ai.other_descriptors is not None:
            # Save the descriptor order
            f = open("%s/odesc_header.csv" % (str(mout_path.absolute())), "w")
            for item in ai.header:
                f.write("%s\n" % (item))
            f.close()
    # Choose between static manual cross validation group or
    # Repeated KFold Cross Validation
    cvmethod = None
    cvgroups = None
    if fcvgroup is not None:
        cvgroups = CVGroupRead(fcvgroup)
        cvmethod = StaticGroupCV(cvgroups)
        # cvmethod = RepeatedStratifiedCV(cvgroups, n_repeats_, 2)
    else:
        cvmethod = RepeatedKFold(available_keys,
                                 n_splits_,
                                 n_repeats_,
                                 random_state,
                                 test_size=0.2)
    cv_ = 0
    for train_keys, val_keys, test_keys in cvmethod:
        print("Train set size: %d Val set size %d Test set size: %d" % (len(train_keys),
                                                                        len(val_keys),
                                                                        len(test_keys)))
        # Some memory clean-up
        K.clear_session()
        # print(global_test_intexes)
        model = None
        model_ = GetKerasModel()
        if ai.other_descriptors is None:
            if model_ is None:
                model = build_model(ai.conv3d_chtype,
                                    ai.input_shape,
                                    ndense_layers,
                                    nunits,
                                    nfilters)
            else:
                model = model_(ai.conv3d_chtype,
                               ai.input_shape,
                               ndense_layers,
                               nunits,
                               nfilters)
            # model = model_scirep(ai.conv3d_chtype, ai.input_shape, ndense_layers, nunits, nfilters)
            # model = ResNetModel(ai.input_shape)
            print(model.summary())
        else:
            if model_ is None:
                model = build_2DData_model(ai.conv3d_chtype,
                                           ai.input_shape,
                                           ai.nfeatures,
                                           ndense_layers,
                                           nunits,
                                           nfilters)
            else:
                model = model_(ai.conv3d_chtype,
                               ai.input_shape,
                               ndense_layers,
                               nunits,
                               nfilters)

            """
            for l in model.layers[0].layers:
                print(l.summary())
            """
            print("Total Summary")
            print(model.summary())

        dname = os.path.basename(csv_target).replace(".csv", "")
        log_dir_ = ("./logs/cv%d_%s_%d_#rot%d_#f%d_#dl%d_#u%d_" % (cv_,
                                                                   dname,
                                                                   num_epochs,
                                                                   train_steps_per_epoch_,
                                                                   nfilters,
                                                                   ndense_layers,
                                                                   nunits))
        log_dir_ += time.strftime("%Y%m%d%H%M%S")
        
        model_outfile = "%s/%d.h5" % (str(mout_path.absolute()), cv_)
        callbacks_ = [TensorBoard(log_dir=log_dir_,
                                  histogram_freq=0,
                                  write_graph=False,
                                  write_images=False),
                      ModelCheckpoint(model_outfile,
                                      monitor='val_loss',
                                      verbose=0,
                                      save_best_only=True)]

        train_generator = ai.VoxelTrainGenerator(train_keys, n_rot_train)
        x_train_, y_train_ = ai.VoxelTestSetGenerator(train_keys, n_rot_train)
        x_test_, y_test_ = ai.VoxelTestSetGenerator(test_keys, n_rot_test)
        x_val_, y_val_ = ai.VoxelTestSetGenerator(val_keys, n_rot_test)
        val_generator = ai.VoxelTrainGenerator(val_keys, n_rot_test)
        model.fit_generator(train_generator,
                            epochs=num_epochs,
                            steps_per_epoch=train_steps_per_epoch_,
                            verbose=1,
                            # validation_data=(x_test_, y_test_),
                            validation_data=val_generator,
                            validation_steps=test_steps_per_epoch_,
                            callbacks=callbacks_,
                            use_multiprocessing=True)
        """
        if y_recalc is True:
            # Recalculate y it takes a lot of time
            x_dataset_, y_dataset_ = ai.VoxelTestSetGenerator(train_keys, n_rotation_test)
            yrecalc = model.predict(x_dataset_)
            # Store the recalculated y
            k = 0
            c = 0
            for i in range(len(yrecalc)):
                recalc[train_keys[k]].extend(list(yrecalc[i]))
                if c == n_rotation_test-1:
                    k += 1
                    c = 0
                else:
                    c += 1
        """

        """
        test_scores = model.evaluate(x_test_, y_test_)
        print("Test Scores: {}".format(test_scores))
        """
        model = GetLoadModelFnc()(model_outfile)
        y_recalc = model.predict(x_train_)
        ypred_test = model.predict(x_test_)
        ypred_val = model.predict(x_val_)
        # exp_pred_plot(y_test_, ypred_test[:,0])
        r2 = RSQ(y_train_, y_recalc)
        q2 = RSQ(y_test_, ypred_test)
        vr2 = RSQ(y_val_, ypred_val)
        print("Train R2: %.4f Test Q2: %.4f Val: R2: %.4f\n" % (r2, q2,     vr2))

        # Store the test prediction result
        k = 0
        c = 0
        for i in range(len(ypred_val)):
            valpredictions[test_keys[k]].append(list(ypred_val[i]))
            if c == n_rot_test-1:
                k += 1
                c = 0
            else:
                c += 1

        # Store the cross validation result
        k = 0
        c = 0
        for i in range(len(ypred_test)):
            predictions[test_keys[k]].append(list(ypred_test[i]))
            if c == n_rot_test-1:
                k += 1
                c = 0
            else:
                c += 1

        """
        Compute the feature importance according to the Breiman-Fisher-Rudin-Dominici-Algorithm
        Train a model f with a feature map X and a target vector y. Measure th error L(y, y_pred) = e_original

        Input: trained model f, feature matrix X, target vector y, error measure L(y, y_pred)
        1) Estimate the original model error
        2) For each feature:
          - Generate a feature matrix with the p feature permutated N times to breaks the
            association between Xj and y
          - estimate the error using the permutated X feature matrix
          - calculate the feature importance FI = e_perm/e_original or FI = e_perm - e_original
        3) Sort variables by descending Fi

        The error estimation utilised is the mean squared error calculated with this formula
        mse = (np.square(A - B)).mean(axis=0)
        """
        if feat_imp is not None:
            # e_orig = MSE(list(y_test_), list(ypred))
            e_orig = MAE(list(y_test_), list(ypred_test))
            # calculate the feature importance for the descriptors
            for fid_ in range(ai.nfeatures):
                for it in range(feat_imp_iterations):
                    x_val_perm = ai.FeaturePermutation(x_val_, fid=fid_)
                    ypred_perm = model.predict(x_val_perm)
                    # e_perm = MSE(list(y_test_), list(ypred_perm))
                    e_perm = MAE(list(y_test_), list(ypred_perm))
                    feat_imp[fid_].append(e_perm/e_orig)

            # Calculate the feature importance for the voxel information
            for it in range(feat_imp_iterations):
                x_val_perm = ai.FeaturePermutation(x_val_, fid=9999)
                ypred_perm = model.predict(x_val_perm)
                e_perm = MAE(list(y_test_), list(ypred_perm))
                feat_imp[-1].append(e_perm/e_orig)

        if mout_path is not None:
            model.save("%s/%d.h5" % (str(mout_path.absolute()), cv_))
        # Update the cross validation id
        cv_ += 1

    if cvout is not None:
        WriteCrossValidationOutput(cvout, self.target, predictions, None)
    
    if feat_imp is not None:
        fo = open("%s" % (featimp_out), "w")
        for i in range(ai.nfeatures):
            """
            fo.write("%s," % (ai.header[i]))
            for j in range(len(feat_imp[i])-1):
                fo.write("%.4f," % (feat_imp[i][j]))
            fo.write("%.4f\n" % (feat_imp[i][-1]))
            """
            a = np.array(feat_imp[i])
            min_a = a.min()
            q1 = np.percentile(a, 25)
            med_a = np.percentile(a, 50)
            q3 = np.percentile(a, 75)
            max_a = a.max()
            fo.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f\n" % (ai.header[i],
                                                        min_a,
                                                        q1,
                                                        med_a,
                                                        q3,
                                                        max_a))
        a = np.array(feat_imp[-1])
        min_a = a.min()
        q1 = np.percentile(a, 25)
        med_a = np.percentile(a, 50)
        q3 = np.percentile(a, 75)
        max_a = a.max()
        fo.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f\n" % ("qm_voxel_charge",
                                                    min_a,
                                                    q1,
                                                    med_a,
                                                    q3,
                                                    max_a))
        """
        fo.write("%s,\n" % ("qm_voxel_charge"))
        for j in range(len(feat_imp[-1])-1):
            fo.write("%.4f," % (feat_imp[-1][j]))
        fo.write("%.4f\n" % (feat_imp[-1][-1]))
        """
        fo.close()
    
    ycvp = {}
    for key in predictions.keys():
        if len(predictions[key]) > 0:
            ycvp[key] = np.mean(predictions[key])
        else:
            continue
    return ycvp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target_csv',
                   default=None,
                   type=str,
                   help='Voxel database')
    p.add_argument('--desc_csv',
                   default=None,
                   type=str,
                   help='Molecule descriptors')
    p.add_argument('--db',
                   default=None,
                   type=str,
                   help='DX path')
    p.add_argument('--regression',
                   default=True,
                   type=bool,
                   help='regression model type')
    p.add_argument('--classification',
                   default=False,
                   type=bool,
                   help='classification model type')
    p.add_argument('--mout',
                   default=None,
                   type=str,
                   help='model output')
    p.add_argument('--epochs',
                   default=100,
                   type=int,
                   help='Number of epochs')
    p.add_argument('--random_state',
                   default=123458976,
                   type=int,
                   help='Number of epochs')
    p.add_argument('--n_rot_train',
                   default=10,
                   type=int,
                   help='Number of voxel rotation multiplication for the train. (batch_size = n_rot_train*n_molecules)')
    p.add_argument('--train_steps_per_epoch',
                   default=500,
                   type=int,
                   help='How many times the voxel data generator with its augmentation is called for the training set)')
    p.add_argument('--n_rot_validation',
                   default=5,
                   type=int,
                   help='Number of voxel rotation for the validation test')
    p.add_argument('--val_steps_per_epoch',
                   default=100,
                   type=int,
                   help='How many times the voxel data generator with its augmentation is called for the validation set)')
    p.add_argument('--nlayers',
                   default=10,
                   type=int,
                   help='Number of dense NN layers')
    p.add_argument('--nunits',
                   default=64,
                   type=int,
                   help='Number of units in DNN')
    p.add_argument('--nfilters',
                   default=16,
                   type=int,
                   help='Number of filters for Conv3d')
    p.add_argument('--cvout',
                   default=None,
                   type=str,
                   help='Run Cross Validation')
    p.add_argument('--n_splits',
                   default=5,
                   type=int,
                   help='Number of Cross Validation splits')
    p.add_argument('--n_repeats',
                   default=20,
                   type=int,
                   help='Number of Cross Validation splits')
    p.add_argument('--cvgroupfile',
                   default=None,
                   type=str,
                   help='Static cross-validation group file')
    p.add_argument('--tid',
                   default=None,
                   type=int,
                   help='Test Group to use in simple run for model evaluation')
    p.add_argument('--featimp_out',
                   default=None,
                   type=str,
                   help='Feature Importance output')
    args = p.parse_args(sys.argv[1:])

    if args.target_csv is None and args.db is None:
        print("\nUsage:")
        print("%s --csv [CSV database]" % (sys.argv[0]))
        print("--db [Voxel Database path]")
        print("--mout [keras model out]")
        print("--epochs [default 100]")
        print("--n_rot_train [default 1]")
        print("--train_steps_per_epoch [default 500]")
        print("--n_rot_validation [default 1]")
        print("--val_steps_per_epoch [default 100]")
        print("--nlayers [default 10]")
        print("--nunits [default 64]")
        print("--mout [default None]")
        print("--cvout [default None]")
        print("--n_splits [default None]")
        print("--n_repeats [default None]")
        print("\nUsage simple run test: make3DCNNmodel.py --db Gasteiger_Database_g50_s25/ --target_csv TorusDiolSFC.csv  --epochs 1000 --n_rot_train 4 --train_steps_per_epoch 25 --n_rot_validation 1 --val_steps_per_epoch 5 --nunits 32  --nfilters 8")
        print("\nUsage model run test example: python ../Code/make3DCNNmodel.py --db OptimisedConformation/--target_csv LogKwActivity.csv  --n_rot_validation 2 --n_rot_train 4 --train_steps_per_epoch 25 --val_steps_per_epoch 5  --epochs 30 --nlayers 13 --mout logKwModel_rt50_rv100_e30_dl13 --cvgroupfile cvgroup.csv --tid 6")
        print("\nUsage model validation example: python ../Code/make3DCNNmodel.py --db VDatabase_g20_s20_r500 --target_csv LogKwActivity.csv --n_rot_validation 100 --n_rot_train 50 --epochs 30 --nlayers 13 --cvout LogKwValidation.csv --n_repeats 3\n")
    else:
        if args.cvout is not None:
            cv(args.db,
               args.target_csv,
               args.desc_csv,
               args.n_splits,
               args.n_repeats,
               args.epochs,
               args.n_rot_train,
               args.train_steps_per_epoch,
               args.n_rot_validation,
               args.val_steps_per_epoch,
               args.nlayers,
               args.nunits,
               args.nfilters,
               args.random_state,
               args.cvout,
               args.cvgroupfile,
               args.featimp_out,
               False,
               args.mout)
            """
            loo(args.db,
               args.target_csv,
               args.desc_csv,
               args.epochs,
               args.n_rot_train,
               args.steps_per_epoch,
               args.n_rot_validation,
               args.nlayers,
               args.nunits,
               args.nfilters,
               args.cvout)
            """
        else:
            simplerun(args.db,
                      args.target_csv,
                      args.desc_csv,
                      args.epochs,
                      args.n_rot_train,
                      args.train_steps_per_epoch,
                      args.n_rot_validation,
                      args.val_steps_per_epoch,
                      args.nlayers,
                      args.nunits,
                      args.nfilters,
                      args.random_state,
                      args.mout,
                      args.cvgroupfile,
                      args.tid)
    return 0


if __name__ in "__main__":
    main()
