#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""CLI tool to create an OHEDatabase from a directory of CSV files."""

import sys
from deepmolecularnetwork.ohe.database import OHEDatabase


def main(argv: list[str] | None = None) -> int:
    """CLI runner to convert CSV directory into OHEDatabase."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("\nUsage: makeOHEDatabase [CSV directory in] [OHE database out]\n")
        return 1
    ohedb = OHEDatabase()
    ohedb.load_ohe_db_from_csv_dir(args[0])
    ohedb.save_ohe_db(args[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
