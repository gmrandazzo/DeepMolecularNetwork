#!/usr/bin/env python3
# (c) 2018-2023 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import os
from pathlib import Path
import numpy as np


def write_name_list(namelst, fout):
    """Write list of molecule names to file."""
    with open(fout, "w") as fo:
        for name in namelst:
            fo.write("%s\n" % (name))


def load_name_list(fnames):
    """Load list of molecule names from file."""
    names = []
    with open(fnames) as fi:
        for line in fi:
            names.append(line.strip())
    return names


def add_dnn_database(path_, names, xmatrix_):
    """Append a descriptor matrix and names to DNN database directory."""
    xmatrix = np.array(xmatrix_)
    dbpath = str(path_) if str(path_).endswith(".dnndb") else f"{path_}.dnndb"
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
    write_name_list(names, "%s/db%d.names" % (dbpath, i))


def load_dnn_database(dbpath):
    """Load stored DNN database batches into a dictionary."""
    d = {}
    with open("%s/db.info" % (dbpath)) as fi:
        for line in fi:
            v = str.split(line.strip(), ";")
            names = load_name_list(f"{dbpath}/{v[0]}.names")
            mx = np.load(f"{dbpath}/{v[0]}.npy", mmap_mode=None)
            d[v[0]] = [names, mx, v[1], v[2]]
    return d


# Backward compatible aliases
WriteNameList = write_name_list
LoadNameList = load_name_list
AddDNNDatabase = add_dnn_database
LoadDNNDatabase = load_dnn_database
