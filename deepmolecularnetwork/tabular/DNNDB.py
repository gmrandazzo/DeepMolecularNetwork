#!/usr/bin/env python3
# (c) 2018-2023 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import numpy as np
import os
from pathlib import Path


def WriteNameList(namelst, fout):
    with open(fout, "w") as fo:
        for name in namelst:
            fo.write("%s\n" % (name))


def LoadNameList(fnames):
    names = []
    with open(fnames) as fi:
        for line in fi:
            names.append(line.strip())
    return names


def AddDNNDatabase(path_, names, xmatrix_):
    xmatrix = np.array(xmatrix_)
    dbpath = "%s.dnndb" % (path_)
    i = 0
    if not Path(dbpath).is_dir():
        os.makedirs(dbpath, exist_ok=True)
    else:
        i = len(list(Path(dbpath).glob("**/*.npy")))
    fdb = "%s/db%d" % (dbpath, i)
    shape = xmatrix.shape
    print(shape)
    with open("%s/db.info" % (dbpath), "a") as finfo:
        finfo.write("db%d;%d;%d\n" % (i, shape[0], shape[1]))
    np.save(fdb, xmatrix)
    WriteNameList(names, "%s/db%d.names" % (dbpath, i))


def LoadDNNDatabase(dbpath):
    d = {}
    with open("%s/db.info" % (dbpath)) as fi:
        for line in fi:
            v = str.split(line.strip(), ";")
            names = LoadNameList(f"{dbpath}/{v[0]}.names")
            mx = np.load(f"{dbpath}/{v[0]}.npy", mmap_mode=None)
            d[v[0]] = [names, mx, v[1], v[2]]
    return d
