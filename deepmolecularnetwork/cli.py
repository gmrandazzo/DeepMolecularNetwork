#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""Unified Command-Line Interface (CLI) dispatcher for DeepMolecularNetwork."""

import importlib
import sys
from collections.abc import Callable

from deepmolecularnetwork import __version__

# Command registry: (module_path, function_name, description)
COMMANDS_REGISTRY: dict[str, tuple[str, str, str]] = {
    # Tabular Modeling
    "makednndb": (
        "deepmolecularnetwork.tabular.makeDNNDB",
        "main",
        "Create tabular DNN database from CSV descriptors",
    ),
    "tab-makedb": (
        "deepmolecularnetwork.tabular.makeDNNDB",
        "main",
        "Alias for makednndb",
    ),
    "maketabmodel": (
        "deepmolecularnetwork.tabular.makeTABModel",
        "main",
        "Train tabular DNN model (regression/classification)",
    ),
    "tab-train": (
        "deepmolecularnetwork.tabular.makeTABModel",
        "main",
        "Alias for maketabmodel",
    ),
    "maketabprediction": (
        "deepmolecularnetwork.tabular.makeTABPrediction",
        "main",
        "Make predictions with trained tabular models",
    ),
    "tab-predict": (
        "deepmolecularnetwork.tabular.makeTABPrediction",
        "main",
        "Alias for maketabprediction",
    ),
    # One-Hot Encoding & SMILES Modeling
    "makeohedb": (
        "deepmolecularnetwork.OneHotEncodingModeling.makeOHEDatabase",
        "main",
        "Build One-Hot Encoding database from SMILES",
    ),
    "ohe-makedb": (
        "deepmolecularnetwork.OneHotEncodingModeling.makeOHEDatabase",
        "main",
        "Alias for makeohedb",
    ),
    "makeohemodel": (
        "deepmolecularnetwork.ohe.model",
        "main",
        "Train neural network models on OHE representations",
    ),
    "ohe-train": (
        "deepmolecularnetwork.ohe.model",
        "main",
        "Alias for makeohemodel",
    ),
    "makeoheprediction": (
        "deepmolecularnetwork.ohe.predict",
        "main",
        "Predict targets with trained OHE models",
    ),
    "ohe-predict": (
        "deepmolecularnetwork.ohe.predict",
        "main",
        "Alias for makeoheprediction",
    ),
    "smi2ohe": (
        "deepmolecularnetwork.SMILESModeling.SMI2OHEMatrix",
        "main",
        "Convert SMILES to One-Hot Encoding matrices",
    ),
    # 3D Voxel Modeling
    "makevoxeldb": (
        "deepmolecularnetwork.voxels.database",
        "main",
        "Build 3D voxel database from OpenDX files",
    ),
    "voxel-makedb": (
        "deepmolecularnetwork.voxels.database",
        "main",
        "Alias for makevoxeldb",
    ),
    "make3dcnnmodel": (
        "deepmolecularnetwork.voxels.model",
        "main",
        "Train 3D CNN model on voxel representations",
    ),
    "voxel-train": (
        "deepmolecularnetwork.voxels.model",
        "main",
        "Alias for make3dcnnmodel",
    ),
    "make3dcnnprediction": (
        "deepmolecularnetwork.voxels.predict",
        "main",
        "Predict targets using 3D CNN voxel models",
    ),
    "voxel-predict": (
        "deepmolecularnetwork.voxels.predict",
        "main",
        "Alias for make3dcnnprediction",
    ),
    "makecsvdataset": (
        "deepmolecularnetwork.voxels.make_csv_dataset",
        "main",
        "Export 3D voxel database to flattened tabular CSV",
    ),
    "voxel-makecsv": (
        "deepmolecularnetwork.voxels.make_csv_dataset",
        "main",
        "Alias for makecsvdataset",
    ),
    "smi2voxel": (
        "deepmolecularnetwork.voxels.rdkit_voxelizer",
        "main",
        "Generate multichannel OpenDX voxels from SMILES with RDKit",
    ),
    "voxel-fromsmi": (
        "deepmolecularnetwork.voxels.rdkit_voxelizer",
        "main",
        "Alias for smi2voxel",
    ),
    # Visualizations
    "img2dvis": (
        "deepmolecularnetwork.images2d.visualize",
        "main",
        "Visualize 2D CNN layer feature activations",
    ),
    "vis-2dcnn": (
        "deepmolecularnetwork.images2d.visualize",
        "main",
        "Alias for img2dvis",
    ),
    "vis2dcnn": (
        "deepmolecularnetwork.images2d.visualize",
        "main",
        "Alias for img2dvis",
    ),
    "vis-3dcnn": (
        "deepmolecularnetwork.voxels.visualize",
        "main",
        "Visualize 3D CNN layer activations in OpenDX format",
    ),
    "vis3dcnn": (
        "deepmolecularnetwork.voxels.visualize",
        "main",
        "Alias for vis-3dcnn",
    ),
}


def _load_command(cmd_name: str) -> Callable[[list[str] | None], int]:
    """Dynamically import and return the target command handler."""
    module_path, func_name, _ = COMMANDS_REGISTRY[cmd_name]
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        print(f"ERROR: Could not load subcommand '{cmd_name}' due to missing dependencies:")
        print(f"  {exc}")
        print(
            "\nPlease ensure all required extras are installed "
            "(e.g., pip install deepmolecularnetwork[keras3])."
        )
        sys.exit(1)

    handler = getattr(mod, func_name)
    return handler


def main(argv: list[str] | None = None) -> int:
    """Unified entry point for the 'dmn' / 'deepmolecularnetwork' command."""
    if argv is None:
        argv = sys.argv[1:]

    description = f"""
DeepMolecularNetwork (DMN) v{__version__}
A deep learning framework for QSAR/QSPR modeling with 3D voxels,
tabular descriptors, and molecular representations.
"""

    epilog = """
Primary Subcommands:
  Tabular Modeling:
    makednndb / tab-makedb             Build tabular DNN database
    maketabmodel / tab-train           Train tabular DNN model
    maketabprediction / tab-predict   Predict with tabular model

  One-Hot Encoding & SMILES:
    smi2ohe                            Convert SMILES to OHE matrix
    makeohedb / ohe-makedb             Build OHE database
    makeohemodel / ohe-train           Train OHE neural network
    makeoheprediction / ohe-predict   Predict with OHE model

  3D Voxel Modeling:
    smi2voxel / voxel-fromsmi          SMILES -> RDKit voxels (OpenDX)
    makevoxeldb / voxel-makedb         Build 3D voxel database
    make3dcnnmodel / voxel-train       Train 3D CNN model
    make3dcnnprediction / voxel-predict Predict with 3D CNN model
    makecsvdataset / voxel-makecsv     Export voxel database to CSV

  Visualization:
    vis2dcnn / img2dvis                Visualize 2D CNN activations
    vis3dcnn / vis-3dcnn               Visualize 3D CNN activations

Run 'dmn <subcommand> --help' for details on each subcommand.
"""

    if not argv or argv[0] in ("-h", "--help"):
        print(description)
        print(epilog)
        return 0

    if argv[0] in ("-v", "--version"):
        print(f"DeepMolecularNetwork v{__version__}")
        return 0

    subcommand = argv[0].lower()
    if subcommand not in COMMANDS_REGISTRY:
        print(f"ERROR: Unknown subcommand '{argv[0]}'.")
        print(epilog)
        return 1

    handler = _load_command(subcommand)
    return handler(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
