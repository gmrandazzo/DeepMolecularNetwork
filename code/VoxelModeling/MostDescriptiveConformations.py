#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import multiprocessing
import numpy as np
import os
import sys
import subprocess
import shutil

"""
Global variables
"""
#bins_path = "/Users/marco/projects/QStudioMetrics/build/src/"
bins_path = "/home/marco/QStudioMetrics/build/src/"
pca_bin = "%s/qsm-pca" % (bins_path)
mdc_bin = "%s/qsm-mdc" % (bins_path)


def MDC(finput, ncmp, foutput):
    ncpu = multiprocessing.cpu_count()
    cmd = "%s %s %s %d %d" % (mdc_bin, finput, foutput, ncmp, ncpu)
    return subprocess.call(cmd, shell=True)


def PCA(finput, foutput, ncomponents):
    ncpu = multiprocessing.cpu_count()
    cmd = "%s -model -i %s -o %s -c %d -a 1 -nth %d" % (pca_bin, finput, foutput, ncomponents, ncpu)
    return subprocess.call(cmd, shell=True)


def RMSDRead(rmsdin):
  fi = open(rmsdin, "r")
  names = []
  mx = []
  for line in fi:
    if "matrix" in line:
      continue
    else:
      v = str.split(line.strip(), "\t")
      names.append(v[0])
      mx.append([])
      for i in range(1, len(v)):
        mx[-1].append(float(v[i]))
  fi.close()
  return names, np.array(mx)


def SelectConformations(rmsdin, nameout):
  names, mx = RMSDRead(rmsdin)
  pcamx = "pca_mx_%s" % (rmsdin)
  np.savetxt(pcamx, mx, delimiter="\t", fmt="%.18f")
  pcamodel = "model_%s" % (rmsdin)
  PCA(pcamx, pcamodel, 2)
  mdcin = "%s/T-Scores.txt" % (pcamodel)
  mdcout = "mdc_selection_%s" % (rmsdin)
  MDC(mdcin, 4, mdcout)
  fi = open(mdcout, "r")
  ids = []
  for line in fi:
    ids.append(int(line.strip()))
  fi.close()
  fo = open(nameout, "w")
  for id_ in ids:
    fo.write("%s\n" % (names[id_]))
  fo.close()
  os.remove(pcamx)
  os.remove(mdcout)
  shutil.rmtree(pcamodel)

def main():
  if len(sys.argv) == 3:
    SelectConformations(sys.argv[1], sys.argv[2])
  else:
    print("\nUsage %s [RMSD Matrix] [Compound name out]" % (sys.argv[0]))


if __name__ == '__main__':
  main()
