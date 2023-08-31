#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import sys
from pathlib import Path
from Voxel import LoadVoxelDatabase


def main():
    if len(sys.argv) != 4:
        print("Usage: %s [csv actity] [dx path] [csv out]" % (sys.argv[0]))
    else:
        fi = open(sys.argv[1], "r")
        d = {}
        for line in fi:
          v = str.split(line.strip(), ",")
          if "Molecule" in line:
              continue
          else:
              d[v[0]] = float(v[1])
        fi.close()

        # Load the rototranslated database
        dataset = []
        vdb = LoadVoxelDatabase(sys.argv[2])
        for key in vdb.keys():
            # try:
            target_val = float(d[key])
            c = 0
            for i in range(len(vdb[key])):
                for j in range(len(vdb[key][i])):
                  dataset.append([])
                  dataset[-1].append("%s_%d" % (key, c))
                  dataset[-1].append(str("0"))
                  #dataset[-1].append(str(target_val))
                  dataset[-1].extend(vdb[key][i][j].flatten())
                  c += 1
            # except:
            #     print("Error for %s" % (key))
            #     continue

        fo = open(sys.argv[3], "w")
        fo.write("MOLNAME,ACTIVITY,")
        for j in range(len(dataset[0][2:])-1):
            fo.write("vpnt%d," %(j+2))
        fo.write("vpnt%d\n" % (len(dataset[0])-2))


        for i in range(len(dataset)):
            fo.write("%s,%s," % (dataset[i][0], dataset[i][1]))
            for j in range(2, len(dataset[i])-1):
                fo.write("%s," % (dataset[i][j]))
            fo.write("%s\n" % (dataset[i][-1]))
        fo.close()


if __name__ in "__main__":
    main()
