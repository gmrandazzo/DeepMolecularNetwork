#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import numpy as np
import sys
import random
import math
from openDX import readOpenDX, writeOpenDX
from Voxel import VoxelObject
from time import sleep


def main():
    voxel, origin, delta = readOpenDX(sys.argv[1])
    vobj = VoxelObject(sys.argv[1])
    xrot = vobj.ComputeVoxelRotation(1)
    writeOpenDX(xrot[0], origin, delta, sys.argv[2])
    ch_voxels = vobj.ComputeChannelGeneration(xrot)
    for c in range(ch_voxels.shape[1]):
        outname = "ch%d_%s.dx" % (c, sys.argv[2])
        writeOpenDX(ch_voxels[0][c], origin, delta, outname)


if __name__ == '__main__':
    main()
