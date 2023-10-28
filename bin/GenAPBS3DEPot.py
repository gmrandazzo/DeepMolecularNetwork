#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import sys
import subprocess
import multiprocessing

mol2pqrbin = "/home/marco/MolecularFramework_build/src/tests/testMol2ToPQR"

def Mol2PQR(mol2):
  pqr = mol2.replace(".mol2", ".pqr")
  cmd = "%s %s %s" % (mol2pqrbin, mol2, pqr)
  subprocess.call(cmd, shell=True)
  return pqr

def genAPBS(pqr, grid_size=25, cglen_=50.0):
  apbs_in = pqr.replace(".pqr", ".in")
  f = open(apbs_in, "w")
  f.write("read\n")
  f.write("    mol pqr %s\n"% (pqr))
  f.write("end\n")
  f.write("elec\n")
  f.write("    mg-manual\n")
  f.write("    dime %d %d %d\n" % (grid_size, grid_size, grid_size))
  f.write("    cglen %f %f %f\n" % (cglen_, cglen_, cglen_))
  f.write("    grid 1.0 1.0 1.0\n")
  f.write("    gcent mol 1\n")
  f.write("    mol 1\n")
  f.write("    lpbe\n")
  f.write("    bcfl mdh\n")
  f.write("    pdie 1.0000\n")
  f.write("    sdie 78.5400\n")
  f.write("    srfm smol\n")
  f.write("    chgm spl2\n")
  f.write("    sdens 10.00\n")
  f.write("    srad 1.40\n")
  f.write("    swin 0.30\n")
  f.write("    temp 298.15\n")
  f.write("    calcenergy total\n")
  f.write("    calcforce no\n")
  f.write("    write pot dx %s\n" % (pqr.replace(".pqr", "")))
  f.write("end\n")
  f.write("print elecEnergy 1 end\n")
  f.write("quit\n")
  f.close()
  return apbs_in

def run_cmd(cmd):
  return subprocess.call(cmd, shell=True)


def main():
  if len(sys.argv) >= 2:
    listcmd = []
    for i in range(1, len(sys.argv)):
      pqr = Mol2PQR(sys.argv[i])
      apbs_in = genAPBS(pqr)
      listcmd.append("apbs %s" % (apbs_in))
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(run_cmd, listcmd)
  else:
    print("\nUsage: %s file1.mol2 file2.mol2 ... fileN.mol2\n" % (sys.argv[0]))
  return


if __name__ in "__main__":
    main()
