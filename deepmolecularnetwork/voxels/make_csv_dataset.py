#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""Utility to export flattened 3D voxel matrices to CSV tabular format."""

import argparse
import sys

import numpy as np

from deepmolecularnetwork.voxels.voxels import LoadVoxelDatabase


def make_csv_dataset(csv_activity: str, vdb_path: str, csv_out: str) -> None:
    """Convert a 3D Voxel database and activity CSV into a tabular dataset.

    Args:
        csv_activity: Path to CSV file containing molecule IDs and activity values.
        vdb_path: Path to the 3D Voxel database directory.
        csv_out: Path to write the output flattened CSV table.
    """
    activities: dict[str, float] = {}
    with open(csv_activity) as fi:
        for line in fi:
            line_str = line.strip()
            if not line_str or line_str.startswith("#") or "Molecule" in line_str:
                continue
            parts = line_str.split(",")
            if len(parts) >= 2:
                try:
                    activities[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    continue

    vdb = LoadVoxelDatabase(vdb_path)
    dataset: list[list[str]] = []

    for key, conf_list in vdb.items():
        act_val = activities.get(key, 0.0)
        c = 0
        for conf in conf_list:
            for rot_vox in conf:
                row: list[str] = [f"{key}_{c}", f"{act_val:.6f}"]
                flattened = np.asarray(rot_vox).flatten()
                row.extend(f"{val:.6f}" for val in flattened)
                dataset.append(row)
                c += 1

    if not dataset:
        raise ValueError("No matching records found between activity file and voxel database.")

    num_features = len(dataset[0]) - 2
    header = ["MOLNAME", "ACTIVITY"] + [f"vpnt{i+1}" for i in range(num_features)]

    with open(csv_out, "w") as fo:
        fo.write(",".join(header) + "\n")
        for row in dataset:
            fo.write(",".join(row) + "\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for converting voxel databases to CSV tables."""
    parser = argparse.ArgumentParser(
        description="Convert a 3D Voxel database into a tabular CSV dataset."
    )
    parser.add_argument(
        "--activity_csv", default=None, type=str, help="Path to CSV containing molecule activity"
    )
    parser.add_argument("--db", default=None, type=str, help="Path to 3D Voxel database directory")
    parser.add_argument("--csv_out", default=None, type=str, help="Path for output tabular CSV")

    args = parser.parse_args(argv)

    if args.activity_csv is None or args.db is None or args.csv_out is None:
        parser.print_help()
        print("\nERROR! --activity_csv, --db, and --csv_out are required.")
        return 1

    make_csv_dataset(args.activity_csv, args.db, args.csv_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
