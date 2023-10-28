#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import argparse
import numpy as np
import sys
import os
from DNNDB import AddDNNDatabase, LoadDNNDatabase


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xmatrix',
                   default=None,
                   type=str,
                   help='Molecular descriptors')
    p.add_argument('--dbout',
                   default=None,
                   type=str,
                   help='Database output path')
    args = p.parse_args(sys.argv[1:])

    if args.xmatrix is None or args.dbout is None:
        print("\nUsage:")
        print("--xmatrix [Molecular descriptors file]")
        print("--dbout [Database output path]")
        print("\nUsage: python %s --xmatrix rdkit.desc.csv --dbout rdkitdesc\n" % (sys.argv[0]))
    else:
        fi = open(args.xmatrix, "r")
        mx = []
        header = []
        names = []
        for line in fi:
            if "Objects" in line or "Molec" in line:
                header.extend(str.split(line.strip(), ",")[0])
            else:
                v = str.split(line.strip(), ",")
                mx.append(v[1:])
                names.append(v[0])
        fi.close()
        AddDNNDatabase(args.dbout, names, mx)
    return 0


if __name__ == '__main__':
    main()
