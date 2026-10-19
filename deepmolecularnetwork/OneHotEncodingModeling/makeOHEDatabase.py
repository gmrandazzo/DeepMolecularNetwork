#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""CLI tool to create an OHEDatabase from a directory of CSV files."""

import sys
from deepmolecularnetwork.ohe.database import OHEDatabase


def main() -> None:
    """CLI runner to convert CSV directory into OHEDatabase."""
    if len(sys.argv) != 3:
        print(f"\nUsage: {sys.argv[0]} [CSV database in] [OHE database out]\n")
    else:
        ohedb = OHEDatabase()
        ohedb.loadOHEdbFromCSVDir(sys.argv[1])
        ohedb.saveOHEdb(sys.argv[2])


if __name__ == "__main__":
    main()
