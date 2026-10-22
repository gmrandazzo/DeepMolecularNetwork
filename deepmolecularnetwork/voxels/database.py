#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""3D Voxel database generation and manipulation utilities."""

import argparse
from pathlib import Path
import sys

from deepmolecularnetwork.voxels.voxels import SaveVoxelDatabase, VoxelObject


def build_voxel_database(
    dxdbs: str | list[str],
    dbout: str,
    nrotations: int = 50,
    multichannel: bool = False,
) -> None:
    """Build a voxel database from OpenDX directory paths.

    Args:
        dxdbs: Space-separated string or list of directory paths containing .dx files.
        dbout: Target directory path where the voxel database will be saved.
        nrotations: Number of random rotations to compute for each molecule.
        multichannel: Whether to build multichannel voxel tensors.
    """
    if isinstance(dxdbs, str):
        dbpaths = [p for p in dxdbs.split() if p]
    else:
        dbpaths = [str(p) for p in dxdbs]

    dxfile: dict[str, list[str]] = {}
    for dir_ in dbpaths:
        path_obj = Path(dir_)
        if not path_obj.exists():
            continue
        for p in path_obj.iterdir():
            if p.is_file() and p.name.endswith(".dx"):
                name = str(p.stem).split(".")[0]
                dxfile.setdefault(name, []).append(str(p))

    for key, files in dxfile.items():
        vobj = VoxelObject(files)
        xrot = vobj.ComputeVoxelRotation(nrotations, multichannel)
        SaveVoxelDatabase(dbout, vobj.name, xrot)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for building a 3D Voxel CNN database."""
    p = argparse.ArgumentParser(
        description="Construct a 3D Voxel CNN database from OpenDX volumetric potential files."
    )
    p.add_argument(
        "--dxdbs",
        default=None,
        type=str,
        help="Space-separated directory paths containing OpenDX (.dx) files",
    )
    p.add_argument(
        "--multichannel",
        action="store_true",
        default=False,
        help="Build multi-channel voxel representation from multiple .dx files per molecule",
    )
    p.add_argument(
        "--dbout",
        default=None,
        type=str,
        help="Output directory path for the generated voxel database",
    )
    p.add_argument(
        "--nrotations",
        default=50,
        type=int,
        help="Number of random 3D rotations to compute per conformation (default: 50)",
    )

    args = p.parse_args(argv)

    if args.dxdbs is None or args.dbout is None:
        p.print_help()
        print("\nERROR! Both --dxdbs and --dbout must be provided.")
        return 1

    build_voxel_database(
        dxdbs=args.dxdbs,
        dbout=args.dbout,
        nrotations=args.nrotations,
        multichannel=args.multichannel,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
