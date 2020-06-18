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
import sys
# from sklearn.model_selection import RepeatedKFold
# from tensorflow import set_random_seed
import os
import tensorflow 
if int(tensorflow.__version__[0]) > 1:
    from tensorflow.keras import backend as K
else:
    from keras import backend as K
# Some memory clean-up
K.clear_session()

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../Base" % (dir_path))
from dmnnio import ReadDescriptors
from modelhelpers import LoadKerasModels
from modelhelpers import ReadDescriptorOrder
from keras_additional_loss_functions import rmse, score

class ModelPredictor(object):
    def __init__(self, mpath, csv_descriptors):
        self.mpath = mpath
        self.odesc = ReadDescriptorOrder(mpath)
        self.desc, self.nfeatures, self.header = ReadDescriptors(csv_descriptors)
        self.keys = list(self.desc.keys())
        # check that all the descriptors
        # are present in the previous loaded desc
        nf = []
        self.dmap = {}
        for i in range(len(self.odesc)):
            if self.odesc[i] in self.header:
                self.dmap[self.odesc[i]] = i
            else:
                nf.append(self.odesc[i])
        if len(nf) > 0:
            print("Error! Missing some descriptors")
            print("Prediction not possible!")
            print(nf)
            exit()
        return

    def GenData(self):
        # batch_features = np.array([]).reshape(0, self.nfeatures)
        batch_features = []
        for key in self.desc.keys():
            x = [0 for i in range(len(self.odesc))]
            for i in range(len(self.desc[key])):
                if self.header[i] in self.dmap.keys():
                    p = self.dmap[self.header[i]]
                    x[p] = self.desc[key][i]
                else:
                    continue
            # align column header
            batch_features.append(x)
            # batch_features = np.vstack([batch_features, x])
        return np.array(batch_features)
    
    def predict(self, pout):
        predictions = {}
        for key in self.keys:
            predictions[key] = []

        x_topred = self.GenData()
        for model in LoadKerasModels(self.mpath):
            y_pred = list(model.predict(x_topred))
            # Store the prediction results based on the input generation
            for i in range(len(y_pred)):
                predictions[self.keys[i]].append(y_pred[i])

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
    p.add_argument('--minp',
                   default=None,
                   type=str,
                   help='model input')
    p.add_argument('--pout',
                   default=None,
                   type=str,
                   help='Prediction output')
    args = p.parse_args(sys.argv[1:])

    if args.pout is None or args.desc_csv is None or args.minp is None:
        print("\nUsage:")
        print("--desc_csv [Molecular descriptors file]")
        print("--minp [keras model out]")
        print("--pout [default None]")
        print("\nUsage model prediction example: python %s --desc_csv rdkit.desc.csv --minp model --out output.csv\n" % (sys.argv[0]))
    else:
        mp = ModelPredictor(args.minp,
                            args.desc_csv)
        mp.predict(args.pout)
    return 0


if __name__ in "__main__":
    main()
