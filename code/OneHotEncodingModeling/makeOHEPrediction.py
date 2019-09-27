#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

from SMILES2Matrix import SMILES2MX
import argparse
# from keras.callbacks import Callback, EarlyStopping, TensorBoard
# from miscfun import exp_pred_plot
import numpy as np
import sys
# from sklearn.model_selection import RepeatedKFold
# from tensorflow import set_random_seed
import os
from keras import backend as K
# Some memory clean-up
K.clear_session()


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../Base" % (dir_path))
from misc import ReadDescriptors, LoadKerasModels


class ModelPredictor(object):
    def __init__(self, mpath, smiles, csv_descriptors=None):
        self.models, self.odesc = LoadKerasModels(mpath)
        self.X, self.input_shape = self.ReadSMILES(smiles)
        if csv_descriptors is not None:
            self.desc, self.nfeatures, self.header = ReadDescriptors(csv_descriptors)
        else:
            self.desc = None
            self.nfeatures = 0
            self.header = None

        self.keys = None
        if self.nfeatures > 0:
            self.keys = []
            all_keys_ = None
            if len(self.X.keys()) > len(self.desc.keys()):
                all_keys_ = self.X.keys()
            else:
                all_keys_ = self.desc.keys()
            for key in all_keys_:
                if key in self.X.keys() and key in self.desc.keys():
                    self.keys.append(key)
                else:
                    continue
        else:
            self.keys = self.X.keys()
        self.dmap = None
        if self.nfeatures > 0:
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
                print(nf)
                exit()
        return

    def ReadSMILES(self, smiles_list):
        s2m = SMILES2MX(512)
        f = open(smiles_list, "r")
        X = {}
        for line in f:
            v = str.split(line.strip(), "\t")
            print("Parsing: %s" % (v[1]))
            X[v[1]] = np.array(s2m.smi2mx(v[0]))
        f.close()
        return X, s2m.getshape()

    def GenData(self):
        batch_features = np.array([]).reshape(0, self.nfeatures)
        for key in self.desc.keys():
            x = [0 for i in range(self.nfeatures)]
            for i in range(len(self.X_raw[key])):
                p = self.dmap[self.header[i]]
                x[p] = self.X_raw[key][i]
            # align column header
            batch_features = np.vstack([batch_features, x])
        return batch_features


def predict(desc_csv,
            minp,
            pout):
    mp = ModelPredictor(minp, desc_csv)
    predictions = {}
    for key in mp.keys:
        predictions[key] = []

    x_topred = mp.GenData()
    for model in mp.models:
        y_pred = list(model.predict(x_topred))
        # Store the prediction results based on the input generation
        for i in range(len(y_pred)):
            predictions[mp.keys[i]].append(y_pred[i])

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
    p.add_argument('--smiles',
                   default=None,
                   type=str,
                   help='Smiles molecules')
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

    if args.pout is None or args.db is None or args.minp is None:
        print("\nUsage:")
        print("--smiles [Smiles molecules]")
        print("--desc_csv [Molecular descriptors file]")
        print("--minp [keras model out]")
        print("--pout [default None]")
        print("\nUsage model prediction example: python %s --smiles dataset.smi --desc_csv rdkit.desc.csv --minp model --out output.csv\n" % (sys.argv[0]))
    else:
        predict(args.desc_csv,
                args.minp,
                args.pout)
    return 0


if __name__ in "__main__":
    main()
