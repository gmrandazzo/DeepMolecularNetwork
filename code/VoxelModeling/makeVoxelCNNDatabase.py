#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import sys
from pathlib import Path
from Voxel import VoxelObject, SaveVoxelDatabase
from openDX import readOpenDX, writeOpenDX
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dxdbs',
                   default=None,
                   type=str,
                   help='DX Voxel databases')
    p.add_argument('--multichannel',
                   default=False,
                   type=bool,
                   help='Set MultiChannel DB construction')
    p.add_argument('--dbout',
                   default=None,
                   type=str,
                   help='Database output')
    p.add_argument('--nrotations',
                   default=50,
                   type=int,
                   help='Number rotations')
    args = p.parse_args(sys.argv[1:])



    if args.dxdbs is None and args.dbout is None:
        print("\nUsage: %s --dxdbs [dx path] --nrotations [num rotations] --dbout [out path]" % (sys.argv[0]))
        print("Usage: %s --dxdbs \"[dx path1] [dx path2] [...] [dx pathN] \" --nrotations [num rotations] --dbout [out path] # for multichannel DX\n" % (sys.argv[0]))
        print("\n N.B.: If you provide num rotation = 0, this means that your dx path contains several\n")
        print("\n       rototranslation in the form mol.1.epot.dx mol.2.epot.dx mol.N.epot.dx")
        print("\n       or your databases contains multiple channels for conformations")
    else:
        dbpaths = str.split(args.dxdbs, " ")
        #print("Multichannel: ", args.multichannel)
        # Load the multichannel dx files
        dxfile = {}
        for dir_ in dbpaths:
            for p in Path(dir_).iterdir():
                if p.is_file() and ".dx" in str(p):
                    name = str(p.stem).split(".")[0]
                    if name in dxfile.keys():
                        dxfile[name].append(str(p))
                    else:
                        dxfile[name] = [str(p)]
                else:
                    continue

        for key in dxfile.keys():
            print("Processing %s" % (key))
            # print(dxfile[key])
            vobj = VoxelObject(dxfile[key])
            # print(dxfile[key])
            xrot = vobj.ComputeVoxelRotation(args.nrotations, args.multichannel)
            SaveVoxelDatabase(args.dbout, vobj.name, xrot)

if __name__ in "__main__":
    main()
