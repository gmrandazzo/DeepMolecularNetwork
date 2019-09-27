#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import numpy as np
from multiprocessing import cpu_count, Pool
import os
from openDX import readOpenDX, writeOpenDX
from pathlib import Path
import random
import sys


def SaveVoxelDatabase(path_, name, voxels):
    print(voxels.shape)
    vpath_ = "%s/%s.vdb" % (path_, name)
    i = 0
    if Path(vpath_).is_dir() is False:
        os.makedirs(vpath_, exist_ok=True)
    else:
        i = len(list(Path(vpath_).glob('**/*.npy')))
    db_path = "%s/db%d" % (vpath_, i)
    np.save(db_path, voxels)


def LoadVoxelDatabase(path_):
    d = {}
    for p in Path(path_).iterdir():
        if p.is_dir() and ".vdb" in str(p):
            name = p.resolve().stem.split(".")[0]
            print("Loading %s" % (name))
            for db in Path(str(p)).iterdir():
                if db.is_file() and ".npy" in str(db):
                    arr = np.load(str(db), mmap_mode=None)
                    #arr = np.load(str(db), mmap_mode='r')
                    if name in d.keys():
                        # Append multiple conformations
                        # stored with the same name
                        d[name].append(arr)
                    else:
                        d[name] = [arr]
                else:
                    continue
    return d


class VoxelObject(object):
    """
    Object to define the 3D Object dataset
    """
    def __init__(self, dxnames):
        """
        Initialisation voxel dataset
        dxnames is a list for multichannel
        """
        self.name = str(Path(dxnames[0]).resolve().stem.split(".")[0])
        if "Conf" in self.name:
            #self.name = str.split(self.name, "_")[-1]
            a = str.split(self.name, "_")[1:]
            self.name = ""
            for i in range(len(a)-1):
                self.name += a[i]+"_"
            self.name += a[-1]

        self.voxels = []
        self.origins = []
        self.deltas = []
        for dxname in dxnames:
            v, o, d = self.dxRead(dxname)
            self.voxels.append(v)
            self.origins.append(o)
            self.deltas.append(d)

    def dxRead(self, dxfile):
        """ Read the DX file and convert to numpy array """
        print("Loading %s ..." % (dxfile))
        v, o, d, = readOpenDX(dxfile)
        return v, o, d

    def toFlatten(self):
        row = []
        for ch in range(len(self.voxels)):
            for i in range(len(self.voxels[ch])):
                for j in range(len(self.voxels[ch][i])):
                    for k in range(len(self.voxels[ch][i][j])):
                        row.append(self.voxels[ch][i][j][k])
        return np.array(row)



    def ComputeVoxelRotation(self, n_rotations, multiple_ch=False):
        if n_rotations == 0:
            # Return just the voxels in numy array form
            # In this case, several rototranslation are provided as input file
            # in this manner mol.1.epot.dx mol.2.epot.dx mol.N.epot.dx
            x = []
            for v in self.voxels:
                x.append(v)
            if multiple_ch is True:
                # return one conformation with an array of mutliple channel
                return np.array([x])
            else:
                return np.array(x)
        else:
            print("Rotating %s %d times" % (self.name, n_rotations))
            p = Pool(processes=cpu_count())
            x_rotated = p.map(self.VoxelRotation, [self.voxels for j in range(n_rotations)])
            p.close()
            return np.array(x_rotated)



    """
    def ComputeChannelGeneration(self, voxels):
        p = Pool(processes=cpu_count())
        ch_voxels = p.map(self.ChannelsGenerator, [voxel for voxel in voxels])
        p.close()
        return np.array(ch_voxels)


    def ChannelsGenerator(self, voxel, isovalues=[-0.3, -0.2, -0.15, -0.1, -0.05, 0.05, 0.10, 0.15, 0.20, 0.30]):
       #  N.B.: with this function the data_format have the channel as first in Conv3D!!
        ch = [np.zeros(voxel.shape) for i in range(len(isovalues)+1)]
        for c in range(len(ch)):
            for i in range(voxel.shape[0]):
                for j in range(voxel.shape[1]):
                    for k in range(voxel.shape[2]):
                        if c == 0:
                            if voxel[i][j][k] < isovalues[c]:
                                ch[c][i][j][k] = voxel[i][j][k]
                            else:
                                continue
                        elif c == len(ch)-1:
                            if voxel[i][j][k] > isovalues[c-1]:
                                ch[c][i][j][k] = voxel[i][j][k]
                            else:
                                continue
                        else:
                            if (voxel[i][j][k] < isovalues[c] and
                               voxel[i][j][k] > isovalues[c-1]):
                                ch[c][i][j][k] = voxel[i][j][k]
                            else:
                                continue
        ch = np.array(ch)
        #print(ch.shape)
        return ch
    """

    """ Low level methods """

    def mk3dtr(self, x, y, z):
        """
        pure translatin
        """
        ret = np.eye(4)
        ret[0:3, 3] = x, y, z
        return ret

    def mk3drotx(self, theta):
        """
        pure rotation around x axis
        """
        ret = np.eye(4)
        ret[1, [1, 2]] = [np.cos(theta), -np.sin(theta)]
        ret[2, [1, 2]] = [np.sin(theta), np.cos(theta)]
        return ret

    def mk3droty(self, theta):
        """
        pure rotation around y axis
        """
        ret = np.eye(4)
        ret[0, [0, 2]] = [np.cos(theta), -np.sin(theta)]
        ret[2, [0, 2]] = [np.sin(theta), np.cos(theta)]
        return ret

    def mk3drotz(self, theta):
        """
        pure rotation around z axis
        """
        ret = np.eye(4)
        ret[0, [0, 1]] = [np.cos(theta), -np.sin(theta)]
        ret[1, [0, 1]] = [np.sin(theta), np.cos(theta)]
        return ret

    def shake(self, min_, max_):
        xs = random.randint(min_, max_)
        ys = random.randint(min_, max_)
        zs = random.randint(min_, max_)
        return xs, ys, zs

    def VoxelRotation(self, voxels):
        gs = []
        ngs = []
        for voxel in voxels:
            gs.append(np.array(voxel))
            ngs.append(np.zeros(voxel.shape))
        # g = np.array(voxel)
        # new_grid = np.zeros(g.shape)
        h_x = gs[0].shape[0] / 2.
        h_y = gs[0].shape[1] / 2.
        h_z = gs[0].shape[2] / 2.
        yaw = random.uniform(-np.pi, np.pi)
        pitch = random.uniform(-np.pi, np.pi)
        roll = random.uniform(-np.pi, np.pi)
        # print ("yaw: %.4f pitch: %.4f roll: %.4f" % (yaw, pitch, roll))
        # 1) Translate the center of the voxel to the origin of the voxel
        f_t = self.mk3dtr(-h_x, -h_y, -h_z)
        # 2) Rotate the voxel to some random euler transformation
        f_r = self.mk3drotz(yaw) @ self.mk3droty(pitch) @ self.mk3drotx(roll)
        # 3) Retranslate the actual origin of the voxel to the original center
        f_b = self.mk3dtr(h_x, h_y, h_z)

        # 3) Shake a bit near the actual origin of the voxel to the original center
        """
        xs, ys, zs = self.shake(-2, 2)
        f_b = self.mk3dtr(h_x+xs, h_y+ys, h_z+zs)
        """

        for i in range(len(gs[0])):
            for j in range(len(gs[0][i])):
                for k in range(len(gs[0][i][j])):
                    p = [i, j, k, 1]
                    np_ = f_t @ p
                    np_r = f_r @ np_
                    np_r_t = f_b @ np_r
                    n_i, n_j, n_k, n_t = np.floor(np_r_t).astype('int')
                    if (n_i >= 0 and n_j >= 0 and n_k >= 0) and (n_i < len(gs[0]) and n_j < len(gs[0][i]) and n_k < len(gs[0][i][j])):
                        # new_grid[n_i][n_j][n_k] = g[i][j][k]
                        for ch in range(len(gs)):
                            ngs[ch][n_i][n_j][n_k] = gs[ch][i][j][k]
                    else:
                        # print("Warning voxel point out of space ", (n_i, n_j, n_k, len(gs[0]), len(gs[0][i]), len(gs[0][i][j])))
                        continue
        return ngs


if __name__ == '__main__':
    if len(sys.argv) == 3:
        vobj = VoxelObject([sys.argv[1]])
        new_voxels = vobj.VoxelRotation(vobj.voxels)
        writeOpenDX(new_voxels[0], vobj.origins[0], vobj.deltas[0], sys.argv[2])
    else:
        print("\n Usage: %s [input dx] [output dx]\n" % (sys.argv[0]))
