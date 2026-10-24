#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""Unit tests for the RDKit 3D voxelization module."""

from pathlib import Path
import pytest
import numpy as np

try:
    import rdkit  # noqa: F401

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from deepmolecularnetwork.voxels.rdkit_voxelizer import (
    sanitize_name,
    canonical_orientation,
    grid_axis,
    embed_molecule,
    compute_atom_properties,
    voxelize,
    smiles_to_voxels,
    write_voxels_dx,
    read_smiles_file,
    voxelize_smiles_file,
    main,
)
from deepmolecularnetwork.voxels.opendx import read_open_dx


def test_sanitize_name():
    assert sanitize_name("Mol.1/A") == "Mol_1_A"
    assert sanitize_name("  A-4  ") == "A-4"
    assert sanitize_name("compound#5") == "compound_5"


def test_grid_axis():
    axis = grid_axis(4, 1.0)
    assert len(axis) == 4
    np.testing.assert_allclose(axis, np.array([-1.5, -0.5, 0.5, 1.5]))
    assert np.isclose(np.mean(axis), 0.0)


def test_canonical_orientation():
    coords = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 1.0],
            [-1.0, 0.0, -2.0],
            [4.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    oriented = canonical_orientation(coords)
    assert oriented.shape == coords.shape
    # Centroid should be zero
    np.testing.assert_allclose(oriented.mean(axis=0), np.zeros(3), atol=1e-12)


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not installed")
class TestRDKitVoxelization:
    """Tests requiring RDKit."""

    def test_embed_molecule_and_properties(self):
        mol = embed_molecule("CCO", random_seed=42)
        assert mol is not None
        assert mol.GetNumConformers() == 1
        assert mol.GetNumAtoms() > 3  # explicit Hs added

        props = compute_atom_properties(mol)
        assert "charge" in props
        assert "esp" not in props  # computed at voxelize
        assert "steric" not in props
        assert "logp" in props
        assert "mr" in props
        assert "tpsa" in props
        assert "radius" in props
        assert len(props["charge"]) == mol.GetNumAtoms()

    def test_voxelize_and_smiles_to_voxels(self):
        mol = embed_molecule("CCO", random_seed=42)
        props = compute_atom_properties(mol)
        coords = canonical_orientation(mol.GetConformer().GetPositions())
        v_direct = voxelize(
            coords,
            props,
            channels=("charge", "steric"),
            grid_size=8,
            spacing=1.5,
        )
        assert set(v_direct.keys()) == {"charge", "steric"}
        assert v_direct["charge"].shape == (8, 8, 8)

        grids = smiles_to_voxels(
            "CCO", channels=("charge", "steric", "logp"), grid_size=8, spacing=1.5
        )
        assert set(grids.keys()) == {"charge", "steric", "logp"}
        for ch, arr in grids.items():
            assert arr.shape == (8, 8, 8)
            assert not np.isnan(arr).any()

        # Steric values should be bounded between 0 and 1
        assert grids["steric"].min() >= 0.0
        assert grids["steric"].max() <= 1.0

    def test_voxelize_invalid_channel(self):
        with pytest.raises(ValueError, match="Unknown channels"):
            voxelize(np.zeros((1, 3)), {}, channels=["not_a_channel"])

    def test_write_voxels_dx(self, tmp_path: Path):
        grids = {
            "charge": np.zeros((4, 4, 4), dtype=np.float32),
            "esp": np.ones((4, 4, 4), dtype=np.float32),
        }
        written = write_voxels_dx(grids, "mol1", tmp_path, spacing=1.0)
        assert len(written) == 2
        for p in written:
            assert Path(p).is_file()
            arr, origin, delta = read_open_dx(p)
            assert arr.shape == (4, 4, 4)
            assert len(origin) == 3
            assert delta == [1.0, 1.0, 1.0]

    def test_read_smiles_file_and_voxelize(self, tmp_path: Path):
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO\tethanol\nCC\tethane\n# comment\n")
        entries = read_smiles_file(smi_file)
        assert len(entries) == 2
        assert entries[0] == ("CCO", "ethanol")

        outdir = tmp_path / "dx_out"
        ok, failed = voxelize_smiles_file(
            smi_file,
            outdir,
            channels=("charge", "steric"),
            grid_size=6,
            spacing=1.0,
            nproc=1,
        )
        assert len(ok) == 2
        assert len(failed) == 0
        assert (outdir / "ethanol.charge.dx").is_file()
        assert (outdir / "ethanol.steric.dx").is_file()

    def test_main_cli(self, tmp_path: Path):
        smi_file = tmp_path / "sample.smi"
        smi_file.write_text("c1ccccc1\tbenzene\n")
        outdir = tmp_path / "dx"

        # Missing args
        ret = main([])
        assert ret == 1

        # Invalid channel
        ret_inv = main(
            ["--smi", str(smi_file), "--outdir", str(outdir), "--channels", "invalid_ch"]
        )
        assert ret_inv == 1

        # Valid run
        failed_csv = tmp_path / "failed.csv"
        ret_ok = main(
            [
                "--smi",
                str(smi_file),
                "--outdir",
                str(outdir),
                "--channels",
                "charge,steric",
                "--grid_size",
                "8",
                "--spacing",
                "1.2",
                "--nproc",
                "1",
                "--failed_out",
                str(failed_csv),
            ]
        )
        assert ret_ok == 0
        assert (outdir / "benzene.charge.dx").is_file()
        assert failed_csv.is_file()
