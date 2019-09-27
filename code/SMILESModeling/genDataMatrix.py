#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import sys
from SMILES2Matrix import SMILES2MX


def main():
    if len(sys.argv) == 3:
        s = SMILES2MX(1024)
        fi = open(sys.argv[1], "r")
        fo = open(sys.argv[2], "w")
        for line in fi:
            v = str.split(line.strip(), "\t")
            m = s.smi2mx(v[0].strip())
            fo.write("%s_%s," % (sys.argv[1].replace(".smi", ""), v[1]))
            for i in range(len(m)-1):
                for j in range(len(m[i])):
                    fo.write("%d," % (m[i][j]))
            for j in range(len(m[-1])-1):
                fo.write("%d," % (m[-1][j]))
            fo.write("%d\n" % (m[-1][-1]))
        fi.close()
        fo.close()
    return


if __name__ in "__main__":
    main()
