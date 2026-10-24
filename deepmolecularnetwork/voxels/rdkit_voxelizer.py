#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""Generate multichannel 3D voxel grids (OpenDX) from SMILES using RDKit.

This module provides a fully open-source replacement for the external
APBS / MolDesc tool-chain previously required to build voxel databases.

Pipeline for every molecule:

1. Parse the SMILES, add explicit hydrogens and embed a 3D conformer with
   ETKDGv3, then relax it with MMFF94 (UFF fallback).
2. Compute per-atom properties:

   * ``charge``  - Gasteiger-Marsili partial charges (Gaussian smeared density)
   * ``esp``     - soft-core Coulomb electrostatic potential from the
     Gasteiger charges, clipped to ``[-1, 1]`` e/Angstrom
   * ``steric``  - van der Waals occupancy in ``[0, 1]``
   * ``logp``    - Wildman-Crippen atomic logP contributions
   * ``mr``      - Wildman-Crippen atomic molar refractivity contributions
   * ``tpsa``    - Ertl topological polar surface area atomic contributions

3. Translate the molecule to its centroid and rotate it onto its principal
   axes (PCA) so that every molecule is placed in a canonical orientation.
4. Sample every property on a cubic grid centred on the origin and write one
   OpenDX file per channel named ``<molecule>.<channel>.dx``.

The resulting directory can be consumed directly by ``dmn voxel-makedb``.
"""

import argparse
from collections.abc import Callable, Iterable, Sequence
from multiprocessing import Pool
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

from deepmolecularnetwork.voxels.opendx import write_open_dx

AVAILABLE_CHANNELS: tuple[str, ...] = ("charge", "esp", "steric", "logp", "mr", "tpsa")
DEFAULT_CHANNELS: tuple[str, ...] = ("charge", "esp", "steric", "logp")

# Soft-core radius (Angstrom) used to avoid the Coulomb singularity at nuclei.
_ESP_SOFTCORE = 1.0
# Clipping value for the electrostatic potential channel (e/Angstrom).
_ESP_CLIP = 1.0


def _require_rdkit() -> Any:
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without rdkit
        raise ImportError(
            "RDKit is required for SMILES voxelization. "
            "Install it with: pip install 'deepmolecularnetwork[chem]'"
        ) from exc
    return Chem


def sanitize_name(name: str) -> str:
    """Return a filesystem- and voxel-DB-safe molecule identifier.

    The voxel database derives the molecule name from the part of the file name
    preceding the first ``.``; dots, path separators and whitespace are therefore
    replaced with underscores.
    """
    return re.sub(r"[^A-Za-z0-9_\-+]", "_", name.strip())


def embed_molecule(smiles: str, random_seed: int = 42) -> Any:
    """Build a 3D, hydrogen-complete RDKit molecule from a SMILES string.

    Args:
        smiles: Input SMILES.
        random_seed: Seed for the ETKDG embedding (reproducibility).

    Returns:
        RDKit ``Mol`` with one 3D conformer and explicit hydrogens.

    Raises:
        ValueError: If the SMILES cannot be parsed or embedded.
    """
    chem = _require_rdkit()
    from rdkit.Chem import AllChem

    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"3D embedding failed for SMILES: {smiles!r}")

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:  # pragma: no cover - force field failures keep ETKDG geometry
        pass
    return mol


def compute_atom_properties(mol: Any) -> dict[str, np.ndarray]:
    """Compute per-atom RDKit properties used as voxel channel weights.

    Args:
        mol: RDKit molecule (explicit hydrogens recommended).

    Returns:
        Mapping ``property name -> array of shape (n_atoms,)`` containing
        ``charge``, ``logp``, ``mr``, ``tpsa`` and ``radius`` (vdW radius).
    """
    chem = _require_rdkit()
    from rdkit.Chem import AllChem, rdMolDescriptors

    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array(
        [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()], dtype=np.float64
    )
    charges = np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0)

    crippen = np.array(rdMolDescriptors._CalcCrippenContribs(mol), dtype=np.float64)
    tpsa = np.array(rdMolDescriptors._CalcTPSAContribs(mol), dtype=np.float64)
    ptable = chem.GetPeriodicTable()
    radii = np.array([ptable.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()], dtype=np.float64)

    return {
        "charge": charges,
        "logp": crippen[:, 0],
        "mr": crippen[:, 1],
        "tpsa": tpsa,
        "radius": radii,
    }


def canonical_orientation(coords: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Centre coordinates and rotate them onto their principal axes.

    The axes are ordered by decreasing variance and their signs are fixed by the
    skewness of the projected coordinates so that the result is deterministic
    and invariant to the input rotation/translation.

    Args:
        coords: Atomic coordinates of shape ``(n_atoms, 3)``.
        weights: Optional non-negative atomic weights (e.g. masses).

    Returns:
        Transformed coordinates of shape ``(n_atoms, 3)``.
    """
    xyz = np.asarray(coords, dtype=np.float64)
    w = np.ones(len(xyz)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    centred = xyz - (w[:, None] * xyz).sum(axis=0)
    if len(xyz) < 2:
        return centred

    cov = (centred * w[:, None]).T @ centred
    eigval, eigvec = np.linalg.eigh(cov)
    eigvec = eigvec[:, np.argsort(eigval)[::-1]]
    projected = centred @ eigvec
    skew = (w[:, None] * projected**3).sum(axis=0)
    signs = np.where(skew < 0, -1.0, 1.0)
    projected *= signs
    # Keep a right-handed frame.
    if np.linalg.det(eigvec * signs) < 0:
        projected[:, 2] *= -1.0
    return projected


def grid_axis(grid_size: int, spacing: float) -> np.ndarray:
    """Return 1D grid coordinates centred on the origin."""
    return (np.arange(grid_size, dtype=np.float64) - (grid_size - 1) / 2.0) * spacing


def voxelize(
    coords: np.ndarray,
    properties: dict[str, np.ndarray],
    channels: Sequence[str] = DEFAULT_CHANNELS,
    grid_size: int = 24,
    spacing: float = 1.0,
) -> dict[str, np.ndarray]:
    """Sample atomic properties on a cubic grid.

    Gaussian-smeared channels use a per-atom width ``sigma = 0.5 * r_vdW``.

    Args:
        coords: Atomic coordinates ``(n_atoms, 3)`` (already oriented).
        properties: Output of :func:`compute_atom_properties`.
        channels: Channels to generate (subset of :data:`AVAILABLE_CHANNELS`).
        grid_size: Number of grid points per axis.
        spacing: Grid spacing in Angstrom.

    Returns:
        Mapping ``channel -> array of shape (grid_size, grid_size, grid_size)``.
    """
    unknown = set(channels) - set(AVAILABLE_CHANNELS)
    if unknown:
        raise ValueError(f"Unknown channels {sorted(unknown)}; choose from {AVAILABLE_CHANNELS}")

    axis = grid_axis(grid_size, spacing)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    xyz = np.asarray(coords, dtype=np.float64)
    d2 = ((points[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=-1)  # (n_points, n_atoms)
    sigma = 0.5 * properties["radius"]
    gauss = np.exp(-d2 / (2.0 * sigma[None, :] ** 2))

    shape = (grid_size, grid_size, grid_size)
    grids: dict[str, np.ndarray] = {}
    for ch in channels:
        if ch == "steric":
            values = 1.0 - np.prod(1.0 - gauss, axis=1)
        elif ch == "esp":
            dist = np.sqrt(d2 + _ESP_SOFTCORE**2)
            values = np.clip(
                (properties["charge"][None, :] / dist).sum(axis=1), -_ESP_CLIP, _ESP_CLIP
            )
        else:
            values = gauss @ properties[ch]
        grids[ch] = values.reshape(shape)
    return grids


def smiles_to_voxels(
    smiles: str,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    grid_size: int = 24,
    spacing: float = 1.0,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Convert a SMILES string into multichannel voxel grids."""
    mol = embed_molecule(smiles, random_seed=random_seed)
    props = compute_atom_properties(mol)
    masses = np.array([a.GetMass() for a in mol.GetAtoms()], dtype=np.float64)
    coords = canonical_orientation(mol.GetConformer().GetPositions(), masses)
    return voxelize(coords, props, channels=channels, grid_size=grid_size, spacing=spacing)


def write_voxels_dx(
    grids: dict[str, np.ndarray], name: str, outdir: str | Path, spacing: float
) -> list[str]:
    """Write every voxel channel to ``<outdir>/<name>.<channel>.dx``."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ch, grid in grids.items():
        origin = [float(grid_axis(grid.shape[i], spacing)[0]) for i in range(3)]
        path = out / f"{name}.{ch}.dx"
        write_open_dx(grid, origin, [spacing] * 3, str(path))
        paths.append(str(path))
    return paths


def read_smiles_file(path: str | Path, max_mols: int | None = None) -> list[tuple[str, str]]:
    """Read a ``SMILES<tab|space>name`` file into a list of ``(smiles, name)``."""
    entries: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as fi:
        for i, line in enumerate(fi):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("\t") if "\t" in stripped else stripped.split()
            name = parts[1] if len(parts) > 1 else f"mol{i}"
            entries.append((parts[0].strip(), sanitize_name(name)))
            if max_mols is not None and len(entries) >= max_mols:
                break
    return entries


def _process_entry(task: tuple[str, str, Sequence[str], int, float, int, str]) -> tuple[str, str]:
    smiles, name, channels, grid_size, spacing, seed, outdir = task
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    try:
        grids = smiles_to_voxels(smiles, channels, grid_size, spacing, seed)
        write_voxels_dx(grids, name, outdir, spacing)
        return name, ""
    except Exception as exc:
        return name, str(exc)


def voxelize_smiles_file(
    smi_file: str | Path,
    outdir: str | Path,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    grid_size: int = 24,
    spacing: float = 1.0,
    random_seed: int = 42,
    max_mols: int | None = None,
    nproc: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Voxelize every molecule of a SMILES file.

    Returns:
        Tuple ``(succeeded names, {failed name: error message})``.
    """
    entries = read_smiles_file(smi_file, max_mols)
    tasks: Iterable[Any] = [
        (smi, name, tuple(channels), grid_size, spacing, random_seed, str(outdir))
        for smi, name in entries
    ]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    nproc = nproc or os.cpu_count() or 1
    ok: list[str] = []
    failed: dict[str, str] = {}
    if nproc > 1:
        with Pool(processes=nproc) as pool:
            results = list(pool.imap(_process_entry, tasks, chunksize=4))
    else:
        results = [_process_entry(task) for task in tasks]
    for name, err in results:
        if err:
            failed[name] = err
        else:
            ok.append(name)
    if progress is not None:
        progress(f"Voxelized {len(ok)} molecules, {len(failed)} failures")
    return ok, failed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: SMILES file -> multichannel OpenDX voxel files."""
    p = argparse.ArgumentParser(
        description=(
            "Generate multichannel 3D voxel grids (OpenDX) from SMILES with RDKit "
            "(Gasteiger charges, electrostatic potential, sterics, Crippen, TPSA)."
        )
    )
    p.add_argument("--smi", default=None, type=str, help="SMILES file (SMILES<TAB>name)")
    p.add_argument("--outdir", default=None, type=str, help="Output directory for .dx files")
    p.add_argument(
        "--channels",
        default=",".join(DEFAULT_CHANNELS),
        type=str,
        help=f"Comma-separated channels from {','.join(AVAILABLE_CHANNELS)} "
        f"(default: {','.join(DEFAULT_CHANNELS)})",
    )
    p.add_argument("--grid_size", default=24, type=int, help="Grid points per axis (default: 24)")
    p.add_argument("--spacing", default=1.0, type=float, help="Grid spacing in A (default: 1.0)")
    p.add_argument("--max_mols", default=None, type=int, help="Process only the first N molecules")
    p.add_argument("--nproc", default=None, type=int, help="Worker processes (default: all CPUs)")
    p.add_argument("--random_state", default=42, type=int, help="Conformer embedding seed")
    p.add_argument("--failed_out", default=None, type=str, help="Optional CSV of failed molecules")
    args = p.parse_args(argv)

    if args.smi is None or args.outdir is None:
        p.print_help()
        print("\nERROR! Both --smi and --outdir must be provided.")
        return 1

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    unknown = set(channels) - set(AVAILABLE_CHANNELS)
    if unknown:
        print(f"ERROR! Unknown channels: {sorted(unknown)}")
        return 1

    ok, failed = voxelize_smiles_file(
        args.smi,
        args.outdir,
        channels=channels,
        grid_size=args.grid_size,
        spacing=args.spacing,
        random_seed=args.random_state,
        max_mols=args.max_mols,
        nproc=args.nproc,
        progress=print,
    )
    if args.failed_out is not None:
        with open(args.failed_out, "w", encoding="utf-8") as fo:
            fo.write("Molecule,Error\n")
            for name, err in failed.items():
                fo.write(f'{name},"{err}"\n')
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
