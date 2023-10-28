#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import numpy as np
import sys
import shutil
import subprocess
from openDX import *


def run_cmd(cmd):
  return subprocess.call(cmd, shell=True)


def VoxelGenerator(molecule, formal_charge, probes_ff, probe_ids, dx_out):
  ffgen_bin = "/home/marco/MolDesc/build/src/ForceFieldGen"
  mifgen_bin = "/home/marco/MolDesc/build/src/MIFieldsGen"

  # Create a forcefield file
  ff = molecule.replace(".mol2", ".ff")
  shutil.copyfile(probes_ff, ff)
  cmd = "%s %s %s" % (ffgen_bin, molecule, ff)
  run_cmd(cmd)

  vfiles = []
  for id_ in probe_ids:
    vfiles.append("%s.%s.dx" % (molecule.replace(".mol2", ""), id_))
    cmd = "%s %s %s %s %s 20 20 %s " % (mifgen_bin, molecule, formal_charge, ff, id_, vfiles[-1])
    run_cmd(cmd)

  # Read Voxels and make a global sum
  voxels = []
  origin = []
  delta = []
  for vfile in vfiles:
    print(vfile)
    v, o, d = readOpenDX(vfile)
    voxels.append(v)
    origin.append(o)
    delta.append(d)

  shape = voxels[0].shape
  gsum = np.zeros(shape)
  for i in range(shape[0]):
    for j in range(shape[1]):
      for k in range(shape[2]):
        for l in range(len(voxels)):
          gsum[i][j][k] += voxels[l][i][j][k]
  writeOpenDX(gsum, origin[0], delta[0], dx_out)
  return 0


def main():
  if len(sys.argv) >= 5:
    VoxelGenerator(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:-1], sys.argv[-1])
  else:
    print("\nUsage: %s [molecule (mol2)] [formal charge] [probe forcefield (txt)] [probe ids (int)] [out dx (txt)]" % (sys.argv[0]))

if __name__ == '__main__':
  main()
