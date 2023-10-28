#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details
import argparse
import numpy as np
from pathlib import Path
import random
import sys
import time
from Voxel import VoxelObject
from openDX import writeOpenDX
from keras import backend as K
from keras.models import load_model
from math import ceil

import os

# Some memory clean-up
K.clear_session()


class FeatureVisualization(object):
    def __init__(self, f_model, dxfile):
        self.model = load_model(f_model)
        self.vobj = VoxelObject([dxfile])
        self.voxels = self.vobj.ComputeVoxelRotation(0, False)

    def get_name_last_conv_layer_info(self,):
        last_conv_net_name = None
        nfilters = None
        for layer in self.model.layers:
            if "conv3d" in layer.name:
                last_conv_net_name = str(layer.name)
                nfilters = layer.filters
            else:
                continue
        return last_conv_net_name, nfilters

    def GradCAMAlgorithm(self, outdx=None):
        """
        input_shape = self.voxels.shape
        batch_features = np.array([]).reshape(0,
                                              input_shape[0],
                                              input_shape[1],
                                              input_shape[2])
        """
        # print(self.voxels.shape)
        # convnet input must be of size 5
        # (molecules, grid_x, grid_y, grid_z, additional_shape_for_nn)
        # print(self.voxels[0][0][1][2])
        self.voxels = np.expand_dims(self.voxels, axis=-1)
        # print(self.voxels[0][0][1][2])
        # print(self.voxels.shape)
        print("Predicted output %.4f" % (self.model.predict(self.voxels)[0]))
        # for regression is :, 0
        model_output = self.model.output[:, 0]
        last_conv_net_name, nfilters = self.get_name_last_conv_layer_info()
        print("Last conv layer name: %s" % (last_conv_net_name))
        last_conv_layer = self.model.get_layer(last_conv_net_name)
        grads = K.gradients(model_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2, 3))
        iterate = K.function([self.model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([self.voxels])
        for i in range(nfilters): # nfilters is the number of filters...
            conv_layer_output_value[:, :, :, i] *= pooled_grads_value[i]
        heatcubemap = np.mean(conv_layer_output_value, axis=-1)
        # normalize between 0 and 1
        min_heatcubemap = np.min(heatcubemap)
        max_heatcubemap = np.max(heatcubemap)
        heatcubemap -= min_heatcubemap
        heatcubemap /= (max_heatcubemap-min_heatcubemap)
        # heatcubemap = np.maximum(heatcubemap, 0)
        # heatcubemap /= np.max(heatcubemap)

        # Rescale the heatcubemap to the original size
        voxel_sizes = self.voxels.shape[1:-1]
        lx = (self.vobj.deltas[0][0]*voxel_sizes[0])
        ly = (self.vobj.deltas[0][1]*voxel_sizes[1])
        lz = (self.vobj.deltas[0][2]*voxel_sizes[2])
        # print(lx, ly, lz)
        hcm_sizes = heatcubemap.shape
        new_deltas = [ceil(lx/float(hcm_sizes[0])),
                      ceil(ly/float(hcm_sizes[1])),
                      ceil(lz/float(hcm_sizes[2]))]
        # TODO: Interpolation to get fine cube definition
        # np.save("heatcubemap.npy", heatcubemap)
        writeOpenDX(heatcubemap,
                    self.vobj.origins[0],
                    new_deltas,
                    outdx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dx',
                   default=None,
                   type=str,
                   help='DX file')
    p.add_argument('--model',
                   default=None,
                   type=str,
                   help='h5 model input')
    p.add_argument('--outdx',
                   default=None,
                   type=str,
                   help='heatcube output')
    args = p.parse_args(sys.argv[1:])

    if args.dx is None or args.model is None or args.outdx is None:
        print("\nUsage:")
        print("--dx [openDX file]")
        print("--model [keras h5 model]")
        print("--outdx [voxels heatmap]")
        print("\nUsage voxel heatmapt visualization: python %s --dx molecule.dx --moodel blabla/0.h5 --outdx out_voxel_heatmap.dx\n" % (sys.argv[0]))
    else:
        fv = FeatureVisualization(args.model, args.dx)
        fv.GradCAMAlgorithm(args.outdx)
    return 0

if __name__ == '__main__':
    main()
