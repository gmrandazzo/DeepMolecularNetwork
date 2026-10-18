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
    fo = open(fout, "w")
    for name in namelst:
        fo.write("%s\n" % (name))
    fo.close()


def LoadNameList(fnames):
    names = []
    fi = open(fnames)
    for line in fi:
        names.append(line.strip())
    fi.close()
    return names


def AddDNNDatabase(path_, names, xmatrix_):
    xmatrix = np.array(xmatrix_)
    dbpath = "%s.dnndb" % (path_)
    i = 0
    if Path(dbpath).is_dir() is False:
        os.makedirs(dbpath, exist_ok=True)
    else:
        i = len(list(Path(dbpath).glob("**/*.npy")))
    fdb = "%s/db%d" % (dbpath, i)
    shape = xmatrix.shape
    print(shape)
    finfo = open("%s/db.info" % (dbpath), "a")
    finfo.write("db%d;%d;%d\n" % (i, shape[0], shape[1]))
    finfo.close()
    np.save(fdb, xmatrix)
    WriteNameList(names, "%s/db%d.names" % (dbpath, i))


def LoadDNNDatabase(dbpath):
    d = {}
    fi = open("%s/db.info" % (dbpath))
    for line in fi:
        v = str.split(line.strip(), ";")
        names = LoadNameList(f"{dbpath}/{v[0]}.names")
        mx = np.load(f"{dbpath}/{v[0]}.npy", mmap_mode=None)
        d[v[0]] = [names, mx, v[1], v[2]]
    fi.close()
    return d
