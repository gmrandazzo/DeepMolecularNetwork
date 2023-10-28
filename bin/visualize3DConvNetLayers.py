#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import argparse
import sys
from keras import models
from keras.models import load_model
from openDX import *
import numpy as np
from pathlib import Path


def write_grid_layer(activations, xmin, ymin, zmin, edim, outname):
    ly = 0
    for act in activations:
        print(act.shape)
        for ch in range(act.shape[-1]):
            t = None
            if len(act.shape) == 4:
                t = act[:, :, :, ch]
            else:
                t = act[0, :, :, :, ch]

            try:
                gout = "ly%d_ch%d_%s.dx" % (ly, ch, outname)
                print("Writing: %s" % (gout))
                writeOpenDX(t, o, d, gout)
            except:
                print("Error")
        ly += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='model input')
    parser.add_argument('--dx', default=None, type=str, help='DX input')
    parser.add_argument('--edim', default=None,
                        type=int, help='Voxel edge size')
    args = parser.parse_args(sys.argv[1:])
    if args.model is None and args.dx is None:
        print("\nUsage: %s --model [model input] --dx [file dx to predict]\n" % (sys.argv[0]))
        print("Example usage: ./model/visualize_layer.py --model 5ht1a_human_model --dx DX_lowres/CHEMBL209060.pqr.dx\n")
    else:
        m = load_model(args.model)
        m.summary()
        x_pred, o, d = readOpenDX(args.dx)
        # prepare the input
        x_pred = x_pred.reshape(1,
                                x_pred.shape[0],
                                x_pred.shape[1],
                                x_pred.shape[2],
                                1)
        # print(x_pred.shape)
        # initialising the model from an input tensor
        # and a list of output tensors
        maxlayer = int(input("Insert the max layer to get:\n"))
        layer_outputs = [layer.output for layer in m.layers[:int(maxlayer)]]
        activation_model = models.Model(input=m.input, output=layer_outputs)

        # run the model in prediction mode
        activations = activation_model.predict(x_pred)

        # save the layers outputs
        xmin, ymin, zmin = o
        basename = Path(args.dx).resolve().stem
        write_grid_layer(activations, xmin, ymin, zmin, args.edim, basename)

        print("Prediction probabilities: ", (m.predict(x_pred)))
        print("Done!")


if __name__ in "__main__":
    main()
