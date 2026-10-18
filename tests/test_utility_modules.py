"""
Tests for utility modules: molecule, miscfun, keras_additional_loss_functions,
most_descriptive_compound, dissimilar_compound_selection, modelhelpers.
"""

from pathlib import Path
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for headless test runs
import matplotlib.pyplot as plt  # noqa: E402

from deepmolecularnetwork.utility.molecule import (  # noqa: E402
    POINT,
    MOLECULE,
    get_vdw_radius,
    writepqr,
    acread,
    ac2pqr,
)
from deepmolecularnetwork.utility.miscfun import exp_pred_plot  # noqa: E402
from deepmolecularnetwork.utility.keras_additional_loss_functions import (  # noqa: E402
    rmse,
    score,
)
from deepmolecularnetwork.utility.most_descriptive_compound import MDC  # noqa: E402
from deepmolecularnetwork.utility.dissimilar_compound_selection import (  # noqa: E402
    DISC,
    _median,
)
from deepmolecularnetwork.utility.modelhelpers import (  # noqa: E402
    read_descriptor_order,
    load_model_compat,
    get_cuda_devices,
    get_validation_fnc,
    get_train_test_fnc,
    get_keras_model,
    get_load_model_fnc,
)


class TestMoleculeUtility:
    """Tests for POINT, MOLECULE and PQR/AC conversion."""

    def test_point_creation(self):
        p = POINT(1.0, 2.0, 3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_molecule_rotations_and_aliases(self):
        mol = MOLECULE(random_state=42)
        mol.name = "test_mol"
        mol.coord.append(POINT(1.0, 0.0, 0.0))
        mol.coord.append(POINT(0.0, 1.0, 0.0))
        mol.atomtype.extend(["C", "O"])
        mol.charge.extend([0.1, -0.1])
        mol.radius.extend([1.70, 1.52])

        assert mol.get_natoms() == 2
        # Test backward-compatible alias via __getattr__
        assert mol.getNatoms() == 2

        # Uniform rotation
        mol.uniform_rotation(nrot=4.0)
        assert mol.uni_rot is True
        assert mol.rnd_rot is False
        mol.next_rotation()
        assert mol.get_natoms() == 2

        # Random rotation
        mol.random_rotation()
        assert mol.rnd_rot is True
        mol.next_rotation()
        assert mol.get_natoms() == 2

        # Legacy aliases
        mol.uniformRotation(nrot=10.0)
        assert mol.uni_rot is True
        mol.randomRotation()
        assert mol.rnd_rot is True
        mol.nextRotation()

        with pytest.raises(AttributeError):
            _ = mol.non_existent_method

    def test_vdw_radius(self):
        assert get_vdw_radius("C") == 1.70
        assert get_vdw_radius("c") == 1.70
        assert get_vdw_radius("O") == 1.52
        assert get_vdw_radius("UnknownElement") == 9.99

    def test_ac_and_pqr_io(self, tmp_path):
        ac_file = tmp_path / "mol.ac"
        # Create minimal .ac formatted content
        ac_file.write_text(
            "ATOM 1 C UNL 1 1.000 2.000 3.000 0.1500 1.70\n"
            "ATOM 2 O UNL 1 4.000 5.000 6.000 -0.2000 1.52\n"
        )
        mol = acread(str(ac_file))
        assert mol.get_natoms() == 2
        assert mol.atomtype == ["C", "O"]
        assert mol.coord[0].x == 1.0
        assert mol.coord[1].y == 5.0

        pqr_out = tmp_path / "mol.pqr"
        writepqr(mol, str(pqr_out))
        assert pqr_out.exists()
        content = pqr_out.read_text()
        assert "ATOM      1   C" in content
        assert "ATOM      2   O" in content
        assert "TER\nEND" in content

        pqr_from_ac = tmp_path / "direct.pqr"
        ac2pqr(str(ac_file), str(pqr_from_ac))
        assert pqr_from_ac.exists()
        direct_content = pqr_from_ac.read_text()
        assert "ATOM      1   C" in direct_content


class TestMiscAndLosses:
    """Tests for miscfun and additional loss functions."""

    def test_exp_pred_plot(self, monkeypatch):
        # Mock plt.show to avoid blocking
        monkeypatch.setattr(plt, "show", lambda: None)
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 1.9, 3.1, 3.9, 5.1]
        exp_pred_plot(y_true, y_pred)

    def test_keras_additional_losses(self):
        import tensorflow as tf

        y_t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y_p = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        loss_rmse = rmse(y_t, y_p)
        assert float(np.sum(loss_rmse.numpy())) == 0.0

        loss_score = score(y_t, y_p)
        assert float(np.sum(loss_score.numpy())) == 0.0


class TestCompoundSelection:
    """Tests for MDC and DISC compound selection algorithms."""

    def test_mdc_selection(self):
        # 4 points forming a square distance matrix
        dmx = np.array(
            [
                [0.0, 1.0, 2.0, 1.5],
                [1.0, 0.0, 1.5, 2.0],
                [2.0, 1.5, 0.0, 1.0],
                [1.5, 2.0, 1.0, 0.0],
            ]
        )
        mdc = MDC(dmx, nobjects=2)
        selected = mdc.select()
        assert len(selected) == 2
        assert len(set(selected)) == 2
        assert mdc.mdclist() == selected

    def test_disc_selection_variants(self):
        dmx = np.array(
            [
                [0.0, 1.0, 3.0, 4.0],
                [1.0, 0.0, 2.0, 5.0],
                [3.0, 2.0, 0.0, 1.5],
                [4.0, 5.0, 1.5, 0.0],
            ]
        )
        assert _median([1, 3, 2]) == 2
        assert _median([1, 2, 3, 4]) == 2.5

        for method in ["Max", "Min", "Med", "Sum"]:
            disc = DISC(dmx, method=method, nobjects=2)
            selected = disc.select()
            assert len(selected) == 2
            assert len(set(selected)) == 2


class TestModelHelpers:
    """Tests for modelhelpers helper functions."""

    def test_read_descriptor_order(self, tmp_path):
        mpath = tmp_path / "model_dir"
        mpath.mkdir()
        header_file = mpath / "odesc_header.csv"
        header_file.write_text("feat1\nfeat2\nfeat3\n")

        odesc = read_descriptor_order(str(mpath))
        assert odesc == ["feat1", "feat2", "feat3"]

        empty_odesc = read_descriptor_order(str(tmp_path / "nonexistent"))
        assert empty_odesc == []

    def test_get_cuda_devices(self):
        assert get_cuda_devices() is None

    def test_modelhelpers_fallbacks(self):
        assert get_validation_fnc() is None
        assert get_train_test_fnc() is None
        assert get_keras_model() is None
        loader = get_load_model_fnc()
        assert loader is load_model_compat

    def test_custom_model_module(self, tmp_path, monkeypatch):
        # Create a mock model.py to test positive imports in modelhelpers
        model_code = (
            "def validation(x): return x\n"
            "def traintest(n, ts, rs): return ([], [])\n"
            "def build_model(): return 'mock'\n"
            "def load_personal_model(p): return 'loaded'\n"
        )
        (tmp_path / "model.py").write_text(model_code)
        import sys

        monkeypatch.syspath_prepend(str(tmp_path))
        # Remove cached 'model' if any
        sys.modules.pop("model", None)

        assert get_validation_fnc() is not None
        assert get_train_test_fnc() is not None
        assert get_keras_model() is not None
        assert get_load_model_fnc() is not None

        sys.modules.pop("model", None)

    def test_framework_helpers(self, tmp_path):
        from deepmolecularnetwork.framework import (
            create_adam,
            save_model_compat,
            Sequential,
            Dense,
            is_keras_3,
        )

        opt = create_adam(lr=0.0005)
        assert opt is not None

        m = Sequential([Dense(1, input_shape=(2,))])
        out_p = save_model_compat(m, str(tmp_path / "saved_compat"))
        assert Path(out_p).exists()
        assert isinstance(is_keras_3(), bool)


class TestModelValidation:
    """Tests for modelvalidation splitters and cross-validation methods."""

    def test_validation_splits(self, tmp_path):
        from deepmolecularnetwork.utility.modelvalidation import (
            mdc_train_test_split,
            disc_train_test_split,
            train_test_split,
            cv_group_read,
            static_group_cv,
        )

        dict_target = {f"mol{i}": float(i) for i in range(10)}
        tr1, te1 = mdc_train_test_split(dict_target, n_objects=2)
        assert len(te1) == 2
        assert len(tr1) == 8

        tr2, te2 = disc_train_test_split(dict_target, n_objects=2)
        assert len(te2) == 2
        assert len(tr2) == 8

        tr3, te3 = train_test_split(list(dict_target.keys()), test_size=0.2, random_state=42)
        assert len(tr3) + len(te3) == 10

        group_file = tmp_path / "groups.csv"
        group_file.write_text("mol0,mol1\nmol2,mol3\n")
        d = cv_group_read(str(group_file))
        assert len(d) == 2
        splits = list(static_group_cv(d))
        assert len(splits) == 2
