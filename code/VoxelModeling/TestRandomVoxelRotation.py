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
from openDX import *
from time import sleep

def mk3dtr(x, y, z):
  """
  pure translatin
  """
  ret = np.eye(4)
  ret[0:3,3] = x,y,z
  return ret


def mk3drotx(theta):
    """
    pure rotation around x axis
    """
    ret = np.eye(4)
    ret[1, [1, 2]] = [np.cos(theta), -np.sin(theta)]
    ret[2, [1, 2]] = [np.sin(theta), np.cos(theta)]
    return ret


def mk3droty(theta):
    """
    pure rotation around y axis
    """
    ret = np.eye(4)
    ret[0, [0, 2]] = [np.cos(theta), -np.sin(theta)]
    ret[2, [0, 2]] = [np.sin(theta), np.cos(theta)]
    return ret


def mk3drotz(theta):
    """
    pure rotation around z axis
    """
    ret = np.eye(4)
    ret[0, [0, 1]] = [np.cos(theta), -np.sin(theta)]
    ret[1, [0, 1]] = [np.sin(theta), np.cos(theta)]
    return ret


def VoxelRotation(voxel):
  g = voxel
  new_grid = np.zeros(g.shape)
  h = len(g)/2.
  f_t = mk3dtr(-h, -h, -h)
  yaw = random.uniform(-np.pi, np.pi)
  pitch = random.uniform(-np.pi, np.pi)
  roll = random.uniform(-np.pi, np.pi)
  print ("yaw: %.4f pitch: %.4f roll: %.4f" % (yaw, pitch, roll))
  f_r = mk3drotz(yaw) @ mk3droty(pitch) @ mk3drotx(roll)
  f_b = mk3dtr(h, h, h)
  for i in range(len(g)):
    for j in range(len(g[i])):
      for k in range(len(g[i][j])):
        p = [i, j , k, 1]
        np_ = f_t @ p
        np_r = f_r @ np_
        np_r_t = f_b @ np_r
        n_i, n_j, n_k, n_t = np.floor(np_r_t).astype('int')
        if (n_i >=0 and n_j >=0 and n_k >=0) and (n_i < len(g) and n_j < len(g[i]) and n_k < len(g[i][j])):
          new_grid[n_i][n_j][n_k] = g[i][j][k]
        #else:
        #  print(n_i, n_j, n_k)
  return new_grid

def main():
  if len(sys.argv) != 3:
    voxel, origin, delta = readOpenDX(sys.argv[1]):
    new_voxel = VoxelRotation(voxel)
    writeOpenDX(new_voxel, origin, delta, sys.argv[2])
  else:
    print("\n Usage: %s [input dx] [output dx]")


if __name__ == '__main__':
  main()
