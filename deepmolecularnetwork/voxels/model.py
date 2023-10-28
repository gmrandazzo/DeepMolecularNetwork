"""
opendx file read/write
(c) 2018-2023 gmrandazzo@gmail.com
This file is part of DeepMolecularNetwork.
You can use,modify, and distribute it under
the terms of the GNU General Public Licenze, version 3.
See the file LICENSE for details
"""

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

import argparse
from model_builder import *
import numpy as np
from pathlib import Path
import random
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from voxels import LoadVoxelDatabase
from datetime import datetime
import os
import logging

from deepmolecularnetwork.utility.modelhelpers import (
    GetKerasModel,
    GetLoadModelFnc,
    GetTrainTestFnc
)

from deepmolecularnetwork.utility.io import (
    ReadDescriptors,
    ReadTarget,
    WriteCrossValidationOutput
)
from deepmolecularnetwork.utility.modelvalidation import (
    TrainTestSplit,
    RepeatedKFold,
    CVGroupRead,
    StaticGroupCV,
    RepeatedStratifiedCV
)

class Voxels3DModel:
    def __init__(self,
                 csv_target: str,
                 db: str,
                 csv_descriptors: str = None):
        """
        init function
        """
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
                        print("Input shape Error!")
                        assert voxel[0][0].shape[i] == input_shape[i]
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
    vm = Voxel3DModel(csv_target, db, csv_descriptors)
    available_keys = vm.GetAvailableKeys()
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

    train_generator = vm.VoxelTrainGenerator(train_keys, n_rot_train)

    print(train_keys)
    print(test_keys)

    model = None
    model_ = GetKerasModel()
    if vm.other_descriptors is None:
        if model_ is None:
            model = build_model(vm.conv3d_chtype,
                                vm.input_shape,
                                ndense_layers,
                                nunits,
                                nfilters)
        else:
            model = model_(vm.conv3d_chtype,
                           vm.input_shape,
                           ndense_layers,
                           nunits,
                           nfilters)
        print(model.summary())
    else:
        if model_ is None:
            model = build_2DData_model(vm.conv3d_chtype,
                                       vm.input_shape,
                                       vm.nfeatures,
                                       ndense_layers,
                                       nunits,
                                       nfilters)
        else:
            model = model_(vm.conv3d_chtype,
                           vm.input_shape,
                           vm.nfeatures,
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

    test_generator = vm.VoxelTrainGenerator(test_keys, n_rot_train)

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
                        use_multiprocessing=False)

    x_test_, y_test_ = vm.VoxelTestSetGenerator(test_keys, n_rotation_test)
    y_pred_ = model.predict(x_test_)
    print("Test R2: %.4f" % (r2_score(y_test_, y_pred_)))

    fo = open("statconf.csv", "w")
    for key in vm.statvoxconf.keys():
        fo.write("%s," % (key))
        for i in range(len(vm.statvoxconf[key])):
            for j in range(len(vm.statvoxconf[key][i])):
                fo.write("%d," % (vm.statvoxconf[key][i][j]))
        fo.write("\n")
    fo.close()

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
       y_recalc=None,
       mout=None):
    
    # Load the dataset
    vm = Voxel3DModel(csv_target, db, csv_descriptors)
    available_keys = vm.GetAvailableKeys()
    print("N. instances: %d" % (len(vm.target)))
    # Create storage for results
    predictions = dict()
    valpredictions = dict()
    for key in vm.target.keys():
        predictions[key] = []
        valpredictions[key] = []

    feat_imp = None
    feat_imp_iterations = 20

    if featimp_out is not None:
        # feature importance list for csv descriptors
        if vm.other_descriptors is not None:
            feat_imp = [[] for p in range(vm.nfeatures)]
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
        if vm.other_descriptors is not None:
            # Save the descriptor order
            f = open("%s/odesc_header.csv" % (str(mout_path.absolute())), "w")
            for item in vm.header:
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
        model = None
        model_ = GetKerasModel()
        if vm.other_descriptors is None:
            if model_ is None:
                model = build_model(vm.conv3d_chtype,
                                    vm.input_shape,
                                    ndense_layers,
                                    nunits,
                                    nfilters)
            else:
                model = model_(vm.conv3d_chtype,
                               vm.input_shape,
                               ndense_layers,
                               nunits,
                               nfilters)
            # model = model_scirep(vm.conv3d_chtype, vm.input_shape, ndense_layers, nunits, nfilters)
            # model = ResNetModel(vm.input_shape)
            print(model.summary())
        else:
            if model_ is None:
                model = build_2DData_model(vm.conv3d_chtype,
                                           vm.input_shape,
                                           vm.nfeatures,
                                           ndense_layers,
                                           nunits,
                                           nfilters)
            else:
                model = model_(vm.conv3d_chtype,
                               vm.input_shape,
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
        
        #check if model exist....
        if Path(model_outfile).exists():
            print("The model %s exist.\nLoading the model and continue for prediction")
        else:
            callbacks_ = [TensorBoard(log_dir=log_dir_,
                                    histogram_freq=0,
                                    write_graph=False,
                                    write_images=False),
                        ModelCheckpoint(model_outfile,
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True)]

            train_generator = vm.VoxelTrainGenerator(train_keys, n_rot_train)
            x_train_, y_train_ = vm.VoxelTestSetGenerator(train_keys, n_rot_train)
            x_test_, y_test_ = vm.VoxelTestSetGenerator(test_keys, n_rot_test)
            x_val_, y_val_ = vm.VoxelTestSetGenerator(val_keys, n_rot_test)
            val_generator = vm.VoxelTrainGenerator(val_keys, n_rot_test)
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
            for fid_ in range(vm.nfeatures):
                for it in range(feat_imp_iterations):
                    x_val_perm = vm.FeaturePermutation(x_val_, fid=fid_)
                    ypred_perm = model.predict(x_val_perm)
                    # e_perm = MSE(list(y_test_), list(ypred_perm))
                    e_perm = MAE(list(y_test_), list(ypred_perm))
                    feat_imp[fid_].append(e_perm/e_orig)

            # Calculate the feature importance for the voxel information
            for it in range(feat_imp_iterations):
                x_val_perm = vm.FeaturePermutation(x_val_, fid=9999)
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
        for i in range(vm.nfeatures):
            """
            fo.write("%s," % (vm.header[i]))
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
            fo.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f\n" % (vm.header[i],
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

