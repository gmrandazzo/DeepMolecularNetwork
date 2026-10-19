#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""CLI tool and helper to generate a tabular DNN database from descriptor CSV."""

import argparse
import sys

from deepmolecularnetwork.tabular.DNNDB import AddDNNDatabase


def create_dnn_database(xmatrix_path: str, dbout_path: str) -> None:
    """Read a CSV descriptor file and persist it as a tabular DNN database.

    Args:
        xmatrix_path: Path to the molecular descriptors CSV file.
        dbout_path: Directory path for the generated DNN database.
    """
    with open(xmatrix_path, encoding="utf-8") as fi:
        mx: list[list[str]] = []
        names: list[str] = []
        for line in fi:
            stripped = line.strip()
            if not stripped:
                continue
            if "Objects" in stripped or "Molec" in stripped:
                continue
            parts = stripped.split(",")
            names.append(parts[0])
            mx.append(parts[1:])

    AddDNNDatabase(dbout_path, names, mx)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for makeDNNDB."""
    p = argparse.ArgumentParser(description="Create a DNN database from a CSV descriptor matrix")
    p.add_argument("--xmatrix", default=None, type=str, help="Molecular descriptors CSV file")
    p.add_argument("--dbout", default=None, type=str, help="Database output path")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.xmatrix is None or args.dbout is None:
        p.print_help()
        return 1

    create_dnn_database(args.xmatrix, args.dbout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
