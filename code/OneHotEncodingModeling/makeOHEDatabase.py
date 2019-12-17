#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details
import sys
from OHEDatabase import OHEDatabase


def main():
    if len(sys.argv) != 3:
        print("\nUsage: %s [CSV database in] [OHE database out]\n" % (sys.argv[0]))
    else:
        ohedb = OHEDatabase()
        ohedb.loadOHEdbFromCSVDir(sys.argv[1])
        ohedb.saveOHEdb(sys.argv[2])


if __name__ in "__main__":
    main()
