#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import numpy as np

"""
def readOpenDX(ifile):
  voxel = None
  origin = None
  delta = []
  f = open(ifile, "r")
  c = 1
  values = []
  grid_sizes = []
  for line in f:
    if "gridpositions counts" in line:
      v = str.split(line.strip(), " ")
      grid_size = [int(v[5]), int(v[6]), int(v[7])]
      voxel = np.zeros(grid_size)
    elif "origin" in line:
      v = str.split(line.strip(), " ")
      origin = [float(v[1]), float(v[2]), float(v[3])]
    elif "delta" in line:
      v = str.split(line.strip(), " ")
      delta.append(float(v[c]))
      c += 1
    else:
      if 'attribute "dep" string positions' in line.strip():
          line = str.split(line.strip(), "attribute")[0]
      else:
          line = line.strip()
      v = str.split(line, " ")
      if len(v) == 3 or len(v) == 2 or len(v) == 1:
        for item in v:
          values.append(float(item))
      else:
        continue
  c = 0
  for i in range(grid_size[0]):
    for j in range(grid_size[1]):
      for k in range(grid_size[2]):
        voxel[i][j][k] = values[c]
        c += 1

  return voxel, origin, delta
"""

def nsplit(line, sep):
    v = str.split(line.strip(), sep)
    return list(filter(None, v))

def readOpenDX(ifile):
  voxel = None
  origin = None
  delta = []
  f = open(ifile, "r")
  c = 1
  values = []
  grid_sizes = []
  for line in f:
    if "gridpositions counts" in line:
      v = nsplit(line.strip(), " ")
      grid_size = [int(v[5]), int(v[6]), int(v[7])]
      voxel = np.zeros(grid_size)
    elif "origin" in line:
      v = nsplit(line.strip(), " ")
      origin = [float(v[1]), float(v[2]), float(v[3])]
    elif "delta" in line:
      v = nsplit(line.strip(), " ")
      delta.append(float(v[c]))
      c += 1
    else:
      if 'attribute' in line.strip():
          # line = str(nsplit(line.strip(), "attribute")[:-1][0])
          line = nsplit(line.strip(), "attribute")[:-1]
          if len(line) > 0:
              line = str(line[0])
          else:
              continue
      else:
          line = line.strip()

      v = nsplit(line, " ")
      if len(v) == 3 or len(v) == 2 or len(v) == 1:
        for item in v:
          values.append(float(item))
      else:
        continue
  c = 0
  for i in range(grid_size[0]):
    for j in range(grid_size[1]):
      for k in range(grid_size[2]):
        voxel[i][j][k] = values[c]
        c += 1

  return voxel, origin, delta


def writeOpenDX(voxel, origin, delta, outfile):
  fo = open(outfile, "w")
  fo.write("object 1 class gridpositions counts %d %d %d\n" % (voxel.shape[0], voxel.shape[1], voxel.shape[2]))
  fo.write("origin %f %f %f\n" % (origin[0], origin[1], origin[2]))
  fo.write("delta %f 0.0 0.0\n" % (delta[0]))
  fo.write("delta 0.0 %f 0.0\n" % (delta[1]))
  fo.write("delta 0.0 0.0 %f\n" % (delta[2]))
  fo.write("object 2 class gridconnections counts %d %d %d\n" % (voxel.shape[0], voxel.shape[1], voxel.shape[2]))
  fo.write("object 3 class array type float rank 0 items %d data follows\n" % (voxel.shape[0]*voxel.shape[1]*voxel.shape[2]))
  n = 0
  for i in range(len(voxel)):
    for j in range(len(voxel[i])):
      for k in range(len(voxel[i][j])):
        if n == 2:
          fo.write("%.6e\n" % (voxel[i][j][k]))
          n = 0
        else:
          fo.write("%.6e " % (voxel[i][j][k]))
          n += 1
  if n != 0:
    fo.write("\n")
  # fo.write("attribute \"dep\" string positions\n")
  fo.write("object  \"regular positions regular connections\" class field\n")
  fo.write("component \"positions\" value 1\n")
  fo.write("component \"connections\" value 2\n")
  fo.write("component \"data\" value 3\n")
  fo.close()
