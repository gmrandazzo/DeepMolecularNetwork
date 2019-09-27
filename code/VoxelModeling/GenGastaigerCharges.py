#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import sys
from rdkit import Chem
from rdkit.Chem import AllChem


def nsplit(s, delim=None):
    return [x for x in s.split(delim) if x]

def main():
    if len(sys.argv) != 3:
        print("\nUsage %s input.mol2 output.mol2\n")
    else:
        m = Chem.MolFromMol2File(sys.argv[1], removeHs=False)
        AllChem.ComputeGasteigerCharges(m)
        chgs = [x.GetDoubleProp('_GasteigerCharge') for x in m.GetAtoms()]
        # symbls = [x.GetSymbol() for x in m.GetAtoms()]
        fi = open(sys.argv[1], "r")
        fo = open(sys.argv[2], "w")
        i = 0
        next_atom = False
        for line in fi:
            if "ATOM" in line:
                next_atom = True
                fo.write(line)
            elif "BOND" in line:
                next_atom = False
                fo.write(line)
            else:
                if next_atom is True:
                    v = nsplit(line.strip(), " ")
                    fo.write("%7d %-4s       % 3.4f   % 3.4f   % 3.4f %-4s %4d  %4s       % 2.4f\n" %
                             (int(v[0]),
                              v[1],
                              float(v[2]),
                              float(v[3]),
                              float(v[4]),
                              v[5],
                              int(v[6]),
                              v[7],
                              chgs[i]))
                    i += 1
                else:
                    fo.write(line)

        fo.close()
        fi.close()


if __name__ == '__main__':
    main()
