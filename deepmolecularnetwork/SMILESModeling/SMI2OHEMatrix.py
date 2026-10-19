#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""CLI script converting SMILES file to OHE CSV matrix."""

import sys
from deepmolecularnetwork.ohe.smiles2matrix import SMILES2MX


def main() -> None:
    """Read SMILES file and generate OHE CSV."""
    if len(sys.argv) == 4:
        s = SMILES2MX(int(sys.argv[2]))
        with (
            open(sys.argv[1], encoding="utf-8") as fi,
            open(sys.argv[3], "w", encoding="utf-8") as fo,
        ):
            for line in fi:
                stripped = line.strip()
                if not stripped:
                    continue
                v = stripped.split("\t") if "\t" in stripped else stripped.split()
                m = s.smi2mx(v[0].strip())
                for row in m:
                    fo.write(",".join(str(val) for val in row) + "\n")
                fo.write("END\n")
    else:
        print(f"\nUsage {sys.argv[0]} [smiles] [max padding (int)] [csv output]\n")


if __name__ == "__main__":
    main()
