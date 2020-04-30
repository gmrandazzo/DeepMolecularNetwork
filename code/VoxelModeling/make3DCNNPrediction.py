#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import argparse
# from keras.callbacks import Callback, EarlyStopping, TensorBoard
# from miscfun import exp_pred_plot

import numpy as np
from pathlib import Path
import random
import sys
# from sklearn.model_selection import RepeatedKFold
import time
# from tensorflow import set_random_seed
from Voxel import LoadVoxelDatabase
from datetime import datetime
from keras import backend as K
import os

# Some memory clean-up
K.clear_session()
# Add path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../Base" % (dir_path))
from dmnnio import ReadDescriptors
from modelhelpers import LoadKerasModels


class ModelPredictor(object):
    def __init__(self, mpath, db, csv_descriptors=None):
        self.mpath = mpath
        self.voxels = LoadVoxelDatabase(db)
        print("Max Conformations %d " % (len(list(self.voxels.values())[0])))
        if csv_descriptors is not None:
            self.other_descriptors, self.nfeatures, self.header = ReadDescriptors(csv_descriptors)
        else:
            self.other_descriptors = None
            self.nfeatures = 0
            self.header = None
        self.input_shape, self.conv3d_chtype = self.getInputVoxelShape()

        self.keys = []
        all_keys = list(self.voxels.keys())
        if self.other_descriptors is not None:
            all_keys.extend(list(self.other_descriptors.keys()))
            # remove duplicate from list
            all_keys = list(set(all_keys))
            for key in all_keys:
                if key in self.voxels.keys() and key in self.other_descriptors.keys():
                    self.keys.append(key)
                else:
                    continue
        else:
            self.keys = all_keys
        return

    def getInputVoxelShape(self):
        input_shape = []
        for voxel in self.voxels.values():
            if len(input_shape) == 0:
                input_shape = voxel[0][0].shape
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

    def VoxelDataGenerator(self, random_voxel_rotations):
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

        random.seed(datetime.now().microsecond)
        selected_confs = []
        for key in self.keys:
            # try:
            max_conf = len(self.voxels[key])
            max_rot = len(self.voxels[key][0])
            if random_voxel_rotations > max_rot*max_conf:
                """
                Fix the maximum rotation for the validation
                and return the entire database!
                """
                for c in range(max_conf):
                    for i in range(max_rot):
                        voxel = self.voxels[key][c][i]
                        batch_features = np.vstack([batch_features, [voxel]])
                        if self.other_descriptors is not None:
                            batch_desc_features = np.vstack([batch_desc_features, self.other_descriptors[key]])
            else:
                for i in range(random_voxel_rotations):
                    rconf = random.randint(0, max_conf-1)
                    rrot = random.randint(0, max_rot-1)
                    while True:
                        if "%s-%d-%d" % (key, rconf, rrot) in selected_confs:
                            rconf = random.randint(0, max_conf-1)
                            rrot = random.randint(0, max_rot-1)
                        else:
                            break
                    selected_confs.append("%s-%d-%d" % (key, rconf, rrot))
                    # rconf = random.randint(0, max_conf-1)
                    # rrot = random.randint(0, max_rot-1)
                    voxel = self.voxels[key][rconf][rrot]
                    batch_features = np.vstack([batch_features, [voxel]])
                    if self.other_descriptors is not None:
                        batch_desc_features = np.vstack([batch_desc_features, self.other_descriptors[key]])
            # except:
            #    print("Error while generating the test for %s" % (key))

        # Reshape dataset for Conv3D
        if "last" in self.conv3d_chtype:
            batch_features = batch_features.reshape(batch_features.shape[0],
                                                    batch_features.shape[1],
                                                    batch_features.shape[2],
                                                    batch_features.shape[3], 1)
        if batch_desc_features is not None:
            """
            batch_desc_features = batch_desc_features.reshape(batch_desc_features.shape[0],
                                                              batch_desc_features.shape[1],
                                                              1)
            """
            return [batch_features, batch_desc_features]
        else:
            return batch_features


    def predict(self,
                n_rotations,
                pout):

        predictions = {}
        for key in self.keys:
            predictions[key] = []

        print("N. rotations: %d" % (n_rotations))
        x_topred = self.VoxelDataGenerator(n_rotations)
        for model, _ in LoadKerasModels(self.mpath):
            y_pred = list(model.predict(x_topred))

            print(x_topred.shape)
            #fv = FeatureVisualization(model, x_topred)
            #fv.GradCAMAlgorithm()

            # Store the prediction results based on the input generation
            k = 0
            c = 0
            for i in range(len(y_pred)):
                predictions[self.keys[k]].append(y_pred[i])
                if c == n_rotations-1:
                    k += 1
                    c = 0
                else:
                    c += 1

        fo = open(pout, "w")
        for key in predictions.keys():
            print("Store %s" % (key))
            if len(predictions[key]) > 0:
                ypavg = np.mean(predictions[key])
                ystdev = np.std(predictions[key])
                y_min = np.min(predictions[key])
                y_max = np.max(predictions[key])
                fo.write("%s,%.4f,%.4f,%.4f,%.4f\n" % (key,
                                                    ypavg,
                                                    ystdev,
                                                    y_min,
                                                    y_max))
            else:
                continue
        fo.close()
        fo = open("all_y_%s" % (pout), "w")
        for key in predictions.keys():
            fo.write("%s," % (key))
            for i in range(len(predictions[key])-1):
                fo.write("%.4f," % (predictions[key][i]))
            fo.write("%.4f\n" % (predictions[key][-1]))
        fo.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--desc_csv',
                   default=None,
                   type=str,
                   help='Molecule descriptors')
    p.add_argument('--db',
                   default=None,
                   type=str,
                   help='DX path')
    p.add_argument('--minp',
                   default=None,
                   type=str,
                   help='model input')
    p.add_argument('--n_rotations',
                   default=100,
                   type=int,
                   help='Number of voxel rotation for the train')
    p.add_argument('--pout',
                   default=None,
                   type=str,
                   help='Prediction output')
    args = p.parse_args(sys.argv[1:])

    if args.pout is None or args.db is None or args.minp is None:
        print("\nUsage:")
        print("--db [Voxel Database path]")
        print("--desc_csv [Molecular descriptors file]")
        print("--minp [keras model out]")
        print("--n_rotations [default 100]")
        print("--pout [default None]")
        print("\nUsage model prediction example: python %s --db OptimisedConformation/ --desc_csv rdkit.desc.csv --nr_rotations 100 --minp model --out output.csv\n" % (sys.argv[0]))
    else:
        
        mp = ModelPredictor(args.minp, args.db, args.desc_csv)
        mp.predict(args.n_rotations, args.pout)
    return 0


if __name__ in "__main__":
    main()
