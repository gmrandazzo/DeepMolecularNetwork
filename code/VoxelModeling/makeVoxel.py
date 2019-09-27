#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import sys
import os
from multiprocessing import Pool, cpu_count

molvdb_bin="/home/marco/MolDesc/build/src/makeMolVoxelDB"


def f(cmd):
    os.system(cmd)


def MultiThreadRun(mollst, nrot, gsize, ssize, outdir):
  p = Pool(cpu_count())
  std_stuff="%d %d %d %s" % (nrot, gsize, ssize, outdir)
  cmds = []
  for mol in mollst:
      cmds.append('%s "%s" %s' % (molvdb_bin, mol, std_stuff))
  p.map(f, cmds)


def main():
    if len(sys.argv) >= 6:
        MultiThreadRun(sys.argv[1:-4],
                       int(sys.argv[-4]),
                       int(sys.argv[-3]),
                       int(sys.argv[-2]),
                       sys.argv[-1])
    else:
        print("\nUsage: %s [mol2's] [nrot] [grid size] [step size] [out dir]\n" % (sys.argv[0]))
    return

if __name__ == '__main__':
    main()
