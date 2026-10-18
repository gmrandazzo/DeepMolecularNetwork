"""
Tests for analytics (FeatureImportance) and tabular (DNNDB, ModelPredictor).
"""

from pathlib import Path
import numpy as np
import pytest

from deepmolecularnetwork.analytics.feature_importance import (
    FeatureImportance,
    calc_feat_imp_final_results,
    write_feature_importance,
    print_progress_bar,
    mse_metric,
    mae_metric,
)
from deepmolecularnetwork.tabular.DNNDB import (
    write_name_list,
    load_name_list,
    add_dnn_database,
    load_dnn_database,
)
from deepmolecularnetwork.tabular.makeTABPrediction import ModelPredictor


class DummyModel:
    """Mock model implementing a linear predict interface."""

    def predict(self, X, **kwargs):
        # Predict sum of features
        return np.sum(X, axis=1, keepdims=True)


class TestFeatureImportance:
    """Tests for Breiman-Fisher feature importance calculation."""

    def test_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert mse_metric(y_true, y_pred) == 0.0
        assert mae_metric(y_true, y_pred) == 0.0

        # Progress bar smoke test
        print_progress_bar(1, 10)
        print_progress_bar(10, 10)

    def test_feature_importance_run(self, tmp_path):
        model = DummyModel()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([[3.5], [6.5], [10.5]])
        features = ["f1", "f2"]

        fi = FeatureImportance(model, X, y, features, iterations=3, random_state=42)
        # Test backward-compatible alias via __getattr__
        perm = fi.FeaturePermutation(X, fid=0)
        assert perm.shape == X.shape

        results = fi.calculate(verbose=0)
        assert "f1" in results
        assert "f2" in results
        assert len(results["f1"]["mae"]) == 3

        stats = calc_feat_imp_final_results([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(stats) == 5
        assert stats[0] == 1.0  # min
        assert stats[4] == 5.0  # max

        out_file = tmp_path / "fi_report.csv"
        write_feature_importance(results, str(out_file))
        assert out_file.exists()
        content = out_file.read_text()
        assert "Feature,Min(MSE)" in content
        assert "f1" in content


class TestDNNDatabase:
    """Tests for DNNDB database serialization."""

    def test_name_list_io(self, tmp_path):
        names = ["compound_A", "compound_B", "compound_C"]
        fout = tmp_path / "names.txt"
        write_name_list(names, str(fout))
        loaded = load_name_list(str(fout))
        assert loaded == names

    def test_dnn_db_pipeline(self, tmp_path):
        base_dir = tmp_path / "test_db"
        names = ["c1", "c2"]
        features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        add_dnn_database(str(base_dir), names, features)
        dbpath = f"{base_dir}.dnndb"
        assert Path(dbpath).is_dir()

        d = load_dnn_database(dbpath)
        assert "db0" in d
        loaded_names, loaded_mx, rows, cols = d["db0"]
        assert loaded_names == names
        np.testing.assert_allclose(loaded_mx, features)
        assert int(rows) == 2
        assert int(cols) == 2


class TestModelPredictor:
    """Tests for ModelPredictor feature alignment."""

    def test_model_predictor_gen_data(self, tmp_path):
        model_dir = tmp_path / "dummy_model"
        model_dir.mkdir()
        odesc_file = model_dir / "odesc_header.csv"
        odesc_file.write_text("feature1\nfeature2\n")

        desc_csv = tmp_path / "desc.csv"
        desc_csv.write_text("Molecule,feature1,feature2\nmol1,1.5,2.5\nmol2,3.5,4.5\n")

        mp = ModelPredictor(str(model_dir), str(desc_csv))
        assert mp.odesc == ["feature1", "feature2"]
        aligned = mp.gen_data()
        assert aligned.shape == (2, 2)
        # Test backward-compatible alias via __getattr__
        assert np.array_equal(aligned, mp.GenData())

        # Test predict method
        from deepmolecularnetwork.framework import Sequential, Dense, save_model_compat

        m = Sequential([Dense(1, input_shape=(2,))])
        m.compile(optimizer="adam", loss="mse")
        save_model_compat(m, str(model_dir / "0.h5"))
        pout = tmp_path / "pred_out.csv"
        mp.predict(str(pout))
        assert pout.exists()

    def test_model_predictor_missing_descriptors(self, tmp_path):
        model_dir = tmp_path / "model_missing"
        model_dir.mkdir()
        odesc_file = model_dir / "odesc_header.csv"
        odesc_file.write_text("feature1\nfeature2\nmissing_feat\n")

        desc_csv = tmp_path / "desc.csv"
        desc_csv.write_text("Molecule,feature1,feature2\nmol1,1.5,2.5\n")

        with pytest.raises(SystemExit):
            ModelPredictor(str(model_dir), str(desc_csv))

    def test_model_predictor_getattr_error(self, tmp_path):
        model_dir = tmp_path / "model_attr"
        model_dir.mkdir()
        (model_dir / "odesc_header.csv").write_text("f1\n")
        desc_csv = tmp_path / "desc.csv"
        desc_csv.write_text("Molecule,f1\nmol1,1.0\n")

        mp = ModelPredictor(str(model_dir), str(desc_csv))
        with pytest.raises(AttributeError):
            _ = mp.non_existent_method

    def test_make_tab_prediction_main(self, monkeypatch):
        from deepmolecularnetwork.tabular.makeTABPrediction import main
        import sys

        monkeypatch.setattr(sys, "argv", ["makeTABPrediction.py"])
        assert main() == 0

    def test_load_model_compat_and_devices(self, tmp_path):
        from deepmolecularnetwork.utility.modelhelpers import load_model_compat, get_cuda_devices
        from deepmolecularnetwork.framework import Sequential, Dense, save_model_compat

        m = Sequential([Dense(1, input_shape=(2,))])
        m_file = tmp_path / "saved_mod.h5"
        save_model_compat(m, str(m_file))

        l1 = load_model_compat(str(m_file))
        assert l1 is not None

        # Load without extension
        m_noext = tmp_path / "saved_mod"
        l2 = load_model_compat(str(m_noext))
        assert l2 is not None

        assert get_cuda_devices() is None

    def test_dnn_resnet_layer(self):
        from deepmolecularnetwork.tabular.dnnresnet import dnn_resnet_layer
        from deepmolecularnetwork.framework import Input

        inp = Input(shape=(10,))
        x1 = dnn_resnet_layer(inp, nunits=8, activation="relu", batch_normalization=True)
        assert x1 is not None
        x2 = dnn_resnet_layer(inp, nunits=8, activation=None, batch_normalization=False)
        assert x2 is not None


class TestNNTrainPipeline:
    """Tests for NNTrain simplerun balanced mode and runcv."""

    def test_nntrain_balanced_and_cv(self, tmp_path):
        from deepmolecularnetwork.tabular.makeTABModel import NNTrain

        X_raw = {f"m{i}": np.array([float(i), float(i * 2)], dtype=np.float32) for i in range(8)}
        target = {f"m{i}": np.array([float(i % 2)], dtype=np.float32) for i in range(8)}
        header = ["f1", "f2"]

        nn = NNTrain(X_raw, target, header)
        nn.verbose = 0

        # Balanced batch mode test
        nn.simplerun(
            batch_size_=4,
            batch_mode_=1,
            num_epochs=1,
            ndense_layers=1,
            nunits=8,
            random_state=42,
        )

        # Cross validation test
        cvout = tmp_path / "cv_res.csv"
        mout = tmp_path / "cv_models"
        fimp = tmp_path / "fimp.csv"
        nn.runcv(
            batch_size_=4,
            batch_mode_=0,
            num_epochs=1,
            ndense_layers=1,
            nunits=8,
            cvout=str(cvout),
            n_splits=2,
            n_repeats=1,
            random_state=42,
            mout=str(mout),
            fimpfile=str(fimp),
        )
        assert cvout.exists()
        assert (mout / "odesc_header.csv").exists()
