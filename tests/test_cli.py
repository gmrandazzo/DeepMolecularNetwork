#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""Unit tests for DeepMolecularNetwork CLI entry points and command dispatching."""

import os
import sys
from pathlib import Path
import tempfile
import pytest

from deepmolecularnetwork.cli import main as cli_main
from deepmolecularnetwork.tabular.makeTABModel import main as tab_model_main
from deepmolecularnetwork.tabular.makeTABPrediction import main as tab_predict_main
from deepmolecularnetwork.ohe.model import main as ohe_model_main
from deepmolecularnetwork.ohe.predict import main as ohe_predict_main
from deepmolecularnetwork.OneHotEncodingModeling.makeOHEDatabase import main as ohe_db_main
from deepmolecularnetwork.SMILESModeling.SMI2OHEMatrix import main as smi2ohe_main
from deepmolecularnetwork.voxels.database import main as voxel_db_main
from deepmolecularnetwork.voxels.make_csv_dataset import main as voxel_makecsv_main
from deepmolecularnetwork.voxels.model import main as voxel_model_main
from deepmolecularnetwork.voxels.predict import main as voxel_predict_main
from deepmolecularnetwork.voxels.visualize import main as voxel_vis_main
from deepmolecularnetwork.images2d.visualize import main as img2dvis_main


class TestUnifiedCLI:
    """Tests for the master dmn CLI dispatcher."""

    def test_cli_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify dmn --help outputs usage information and returns 0."""
        ret = cli_main(["--help"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "DeepMolecularNetwork (DMN)" in captured.out
        assert "Primary Subcommands:" in captured.out

    def test_cli_empty_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify dmn without arguments displays usage and returns 0."""
        ret = cli_main([])
        assert ret == 0
        captured = capsys.readouterr()
        assert "DeepMolecularNetwork" in captured.out

    def test_cli_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify dmn --version outputs version and returns 0."""
        ret = cli_main(["--version"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "DeepMolecularNetwork v" in captured.out

    def test_cli_unknown_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify unknown subcommand returns 1 and error message."""
        ret = cli_main(["non-existent-subcommand"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "ERROR: Unknown subcommand" in captured.out

    @pytest.mark.parametrize(
        "subcmd",
        [
            "tab-makedb",
            "makednndb",
            "tab-train",
            "maketabmodel",
            "tab-predict",
            "maketabprediction",
            "ohe-train",
            "makeohemodel",
            "ohe-predict",
            "makeoheprediction",
            "voxel-makedb",
            "makevoxeldb",
            "voxel-train",
            "make3dcnnmodel",
            "voxel-predict",
            "make3dcnnprediction",
            "voxel-makecsv",
            "makecsvdataset",
            "vis-2dcnn",
            "vis2dcnn",
            "img2dvis",
            "vis-3dcnn",
            "vis3dcnn",
        ],
    )
    def test_cli_subcommands_help(self, subcmd: str) -> None:
        """Verify all registered subcommands accept --help cleanly via dmn dispatcher."""
        with pytest.raises(SystemExit) as exc:
            cli_main([subcmd, "--help"])
        assert exc.value.code == 0


class TestIndividualCLIs:
    """Tests for direct entry points of each module."""

    def test_tab_model_cli_missing_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = tab_model_main([])
        assert ret == 1
        captured = capsys.readouterr()
        assert "ERROR!" in captured.out

    def test_tab_predict_cli_missing_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = tab_predict_main([])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_ohe_model_cli_missing_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = ohe_model_main([])
        assert ret == 1
        captured = capsys.readouterr()
        assert "ERROR!" in captured.out

    def test_ohe_predict_cli_missing_args(self) -> None:
        with pytest.raises(SystemExit) as exc:
            ohe_predict_main([])
        assert exc.value.code == 2

    def test_ohe_db_cli_missing_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = ohe_db_main([])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_smi2ohe_cli_missing_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = smi2ohe_main([])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_smi2ohe_cli_execution(self, tmp_path: Path) -> None:
        smi_file = tmp_path / "molecules.smi"
        smi_file.write_text("CC(=O)Oc1ccccc1C(=O)O\tAspirin\nCCO\tEthanol\n")
        out_csv = tmp_path / "matrix.csv"

        ret = smi2ohe_main([str(smi_file), "30", str(out_csv)])
        assert ret == 0
        assert out_csv.exists()
        content = out_csv.read_text()
        assert "END" in content

    def test_voxel_db_cli_missing_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = voxel_db_main([])
        assert ret == 1
        captured = capsys.readouterr()
        assert "ERROR!" in captured.out

    def test_voxel_model_cli_missing_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = voxel_model_main([])
        assert ret == 1
        captured = capsys.readouterr()
        assert "ERROR!" in captured.out

    def test_voxel_predict_cli_missing_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = voxel_predict_main([])
        assert ret == 1
        captured = capsys.readouterr()
        assert "ERROR!" in captured.out

    def test_voxel_makecsv_cli_missing_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = voxel_makecsv_main([])
        assert ret == 1
        captured = capsys.readouterr()
        assert "ERROR!" in captured.out

    def test_voxel_vis_cli_missing_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = voxel_vis_main([])
        assert ret == 1
        captured = capsys.readouterr()
        assert "ERROR!" in captured.out

    def test_img2dvis_cli_missing_args(self) -> None:
        with pytest.raises(SystemExit) as exc:
            img2dvis_main([])
        assert exc.value.code == 2
