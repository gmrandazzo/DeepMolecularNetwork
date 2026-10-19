"""Smoke tests with synthetic data verifying end-to-end execution of all algorithm methods.

Covers:
- Tabular algorithms: NNTrain (simplerun random/balanced, runcv, gen_data, make_matrix,
  data_generator, make_prediction), ModelPredictor, create_dnn_database.
- One-Hot Encoding & SMILES: SMILES2MX, OHEDatabase, NNTrain (unimodal and multimodal),
  ModelPredictor, example_build_model, example_build_2d_data_model, build_gridsearch_model.
- 2D Image algorithms: make_2d_cnn_model, load_and_preprocess_image,
  extract_layer_activations, create_activation_grid, visualize_layers.
- 3D Voxel algorithms: OpenDX IO, VoxelObject, VoxelDatabase, Voxels3DModel,
  simplerun, build_3d_cnn_model, build_3d_resnet_model, Resnet3DBuilder.
- Analytics & Utilities: FeatureImportance, mdc_selection, disc_selection,
  Molecule, Point, exp_pred_plot.
"""

from pathlib import Path
import matplotlib
import numpy as np
from PIL import Image
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from deepmolecularnetwork.analytics.feature_importance import (  # noqa: E402
    FeatureImportance,
    calc_feat_imp_final_results,
    write_feature_importance,
)
from deepmolecularnetwork.framework import (  # noqa: E402
    Sequential,
    create_adam,
    layers,
    save_model_compat,
)
from deepmolecularnetwork.images2d.model import make_2d_cnn_model  # noqa: E402
from deepmolecularnetwork.images2d.visualize import (  # noqa: E402
    create_activation_grid,
    extract_layer_activations,
    load_and_preprocess_image,
    visualize_layers,
)
from deepmolecularnetwork.ohe.database import (  # noqa: E402
    DBFile,
    OHEDatabase,
    load_one_hot_encoding_csv_dir,
    read_one_hot_encoding_csv,
)
from deepmolecularnetwork.ohe.model import (  # noqa: E402
    NNTrain as OHENNTrain,
    build_gridsearch_model as build_ohe_gridsearch_model,
    example_build_2d_data_model,
    example_build_2DData_model,
    example_build_model as build_ohe_example_model,
)
from deepmolecularnetwork.ohe.predict import (  # noqa: E402
    ModelPredictor as OHEModelPredictor,
)
from deepmolecularnetwork.ohe.smiles2matrix import SMILES2MX  # noqa: E402
from deepmolecularnetwork.tabular.DNNDB import (  # noqa: E402
    add_dnn_database,
    load_dnn_database,
)
from deepmolecularnetwork.tabular.makeDNNDB import create_dnn_database  # noqa: E402
from deepmolecularnetwork.tabular.makeTABModel import (  # noqa: E402
    NNTrain as TabularNNTrain,
    build_gridsearch_model as build_tab_gridsearch_model,
    example_build_model as build_tab_example_model,
)
from deepmolecularnetwork.tabular.makeTABPrediction import (  # noqa: E402
    ModelPredictor as TabularModelPredictor,
)
from deepmolecularnetwork.utility.dissimilar_compound_selection import DISC  # noqa: E402
from deepmolecularnetwork.utility.miscfun import exp_pred_plot  # noqa: E402
from deepmolecularnetwork.utility.modelvalidation import (  # noqa: E402
    RepeatedKFold,
    TrainTestSplit,
)
from deepmolecularnetwork.utility.molecule import MOLECULE, POINT  # noqa: E402
from deepmolecularnetwork.utility.most_descriptive_compound import MDC  # noqa: E402
from deepmolecularnetwork.voxels.model import (  # noqa: E402
    Voxels3DModel,
    simplerun as voxels_simplerun,
)
from deepmolecularnetwork.voxels.model_builder import (  # noqa: E402
    build_model as build_voxels_3d_model,
    resnet3d_model,
)
from deepmolecularnetwork.voxels.opendx import read_open_dx, write_open_dx  # noqa: E402
from deepmolecularnetwork.voxels.resnet3d import Resnet3DBuilder  # noqa: E402
from deepmolecularnetwork.voxels.voxels import (  # noqa: E402
    VoxelObject,
    load_voxel_database,
    save_voxel_database,
)


class TestTabularAlgorithmsSmoke:
    """Smoke tests for all Tabular neural network algorithms and utilities."""

    @pytest.fixture
    def synthetic_tabular_data(self):
        """Generate synthetic tabular features, targets, and descriptors."""
        np.random.seed(42)
        keys = [f"mol_{i}" for i in range(12)]
        header = ["desc_1", "desc_2", "desc_3", "desc_4"]
        X_raw = {k: np.random.uniform(0.0, 10.0, size=len(header)).astype(np.float32) for k in keys}
        target = {k: np.array([float(idx % 3)], dtype=np.float32) for idx, k in enumerate(keys)}
        return keys, header, X_raw, target

    def test_nntrain_simplerun_and_batch_modes(self, tmp_path, synthetic_tabular_data):
        """Smoke test NNTrain simplerun with both random (0) and balanced (1) batch modes."""
        keys, header, X_raw, target = synthetic_tabular_data
        trainer = TabularNNTrain(X_raw, target, header)
        trainer.verbose = 0

        # Smoke test batch generators and matrix assembly
        feat_batch, tgt_batch = trainer.gen_data(keys[:4])
        assert feat_batch.shape == (4, len(header))
        assert tgt_batch.shape == (4, 1)

        # Test make_matrix with random and balanced batch mode
        b_feat = np.zeros((4, len(header)), dtype=np.float32)
        b_tgt = np.zeros((4, 1), dtype=np.float32)
        assert trainer.make_matrix(keys, 4, b_feat, b_tgt, batch_mode=0) == 0
        assert trainer.make_matrix(keys, 4, b_feat, b_tgt, batch_mode=1) == 0

        # Test data_generator yield
        gen = trainer.data_generator(keys, batch_size=2, batch_mode=0)
        gx, gy = next(gen)
        assert gx.shape == (2, len(header))
        assert gy.shape == (2, 1)

        # Simplerun with random batch mode
        m_file_random = tmp_path / "model_random.keras"
        res_rand = trainer.simplerun(
            batch_size_=4,
            batch_mode_=0,
            num_epochs=1,
            ndense_layers=1,
            nunits=8,
            random_state=42,
            model_output=str(m_file_random),
        )
        assert res_rand is not None
        model_rand = res_rand["model"]
        assert m_file_random.exists()

        # Simplerun with balanced batch mode
        m_file_balanced = tmp_path / "model_balanced.keras"
        res_bal = trainer.simplerun(
            batch_size_=4,
            batch_mode_=1,
            num_epochs=1,
            ndense_layers=1,
            nunits=8,
            random_state=42,
            model_output=str(m_file_balanced),
        )
        assert res_bal is not None
        assert m_file_balanced.exists()

        # Test make_prediction on keys
        preds = trainer.make_prediction(model_rand, keys[:3])
        assert len(preds) == 3
        for k in keys[:3]:
            assert k in preds

    def test_nntrain_runcv_smoke(self, tmp_path, synthetic_tabular_data):
        """Smoke test RepeatedKFold cross-validation pipeline in Tabular NNTrain."""
        keys, header, X_raw, target = synthetic_tabular_data
        trainer = TabularNNTrain(X_raw, target, header)
        trainer.verbose = 0

        cvout = tmp_path / "cv_summary.csv"
        mout = tmp_path / "cv_models"
        fimp = tmp_path / "cv_fimp.csv"

        trainer.runcv(
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
        assert fimp.exists()
        assert (mout / "odesc_header.csv").exists()

    def test_tabular_model_builders_smoke(self):
        """Smoke test standalone model builders for tabular neural nets."""
        m1 = build_tab_example_model(nfeatures=5, nunits=16, ndense_layers=2, ntargets=1)
        assert m1.output_shape == (None, 1)

        m2 = build_tab_gridsearch_model(
            nfeatures=5,
            ndense_layers=1,
            nunits=16,
            activation="relu",
            dropout=0.2,
        )
        assert m2.output_shape == (None, 1)

    def test_tabular_model_predictor_and_dnndb_smoke(self, tmp_path, synthetic_tabular_data):
        """Smoke test Tabular ModelPredictor and DNNDB database creation."""
        keys, header, X_raw, _ = synthetic_tabular_data

        # 1. Test makeDNNDB creation from CSV
        csv_file = tmp_path / "tabular_descriptors.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("Molecule," + ",".join(header) + "\n")
            for k in keys:
                vals = ",".join(str(v) for v in X_raw[k])
                f.write(f"{k},{vals}\n")

        db_dir = tmp_path / "dnn_database"
        create_dnn_database(str(csv_file), str(db_dir))
        actual_db_dir = f"{db_dir}.dnndb"
        assert (Path(actual_db_dir) / "db.info").exists()

        loaded_db = load_dnn_database(actual_db_dir)
        assert "db0" in loaded_db
        assert len(loaded_db["db0"][0]) == len(keys)

        # Test direct add_dnn_database
        add_dnn_database(str(tmp_path / "extra_dnn"), ["m1"], np.array([[1.0, 2.0]]))
        assert (tmp_path / "extra_dnn.dnndb" / "db.info").exists()

        # 2. Train a small model and predict using Tabular ModelPredictor
        model_dir = tmp_path / "predictor_models"
        model_dir.mkdir()
        odesc_file = model_dir / "odesc_header.csv"
        odesc_file.write_text("\n".join(header) + "\n")

        model = Sequential(
            [
                layers.Input(shape=(len(header),)),
                layers.Dense(8, activation="relu"),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer=create_adam(0.01), loss="mse")
        m_path = model_dir / "0.keras"
        save_model_compat(model, str(m_path))

        predictor = TabularModelPredictor(
            mpath=str(model_dir),
            csv_descriptors=str(csv_file),
        )
        pred_csv = tmp_path / "predictions.csv"
        predictor.predict(pout=str(pred_csv))
        assert pred_csv.exists()
        assert pred_csv.stat().st_size > 0


class TestOneHotEncodingAlgorithmsSmoke:
    """Smoke tests for One-Hot Encoding and SMILES representation pipelines."""

    @pytest.fixture
    def synthetic_ohe_environment(self, tmp_path):
        """Create synthetic SMILES file and OHEDatabase directory."""
        smiles_dict = {
            "ethanol": "CCO",
            "acetic_acid": "CC(=O)O",
            "benzene": "c1ccccc1",
            "ethylamine": "CCN",
            "acetone": "CC(=O)C",
            "propanol": "CCCO",
        }
        smi_file = tmp_path / "molecules.smi"
        with open(smi_file, "w", encoding="utf-8") as f:
            for name, smi in smiles_dict.items():
                f.write(f"{smi}\t{name}\n")

        # Create OHE database directory
        s2m = SMILES2MX(row_padding=32)
        db_dir = tmp_path / "ohe_db"
        db_dir.mkdir()
        for name, smi in smiles_dict.items():
            arr = s2m.smi2array(smi)
            np.save(db_dir / f"{name}.npy", arr)

        target = {name: float(i * 1.5) for i, name in enumerate(smiles_dict.keys())}
        dx = {
            name: np.array([float(len(smi)), float(i)], dtype=np.float32)
            for i, (name, smi) in enumerate(smiles_dict.items())
        }
        return smi_file, db_dir, target, dx

    def test_smiles2matrix_and_ohe_database_smoke(self, tmp_path, synthetic_ohe_environment):
        """Smoke test SMILES2MX and OHEDatabase sampling and CSV IO."""
        smi_file, db_dir, _, _ = synthetic_ohe_environment

        s2m = SMILES2MX(row_padding=32)
        arr = s2m.smi2array("CCO")
        shape = s2m.getshape()
        assert arr.ndim == 2
        assert arr.shape == shape

        # Test OHEDatabase loading and sampling
        db = OHEDatabase()
        db.load_ohe_db(str(db_dir))
        assert len(db.X) == 6
        assert db.input_shape is not None

        sample_dict = db.get_random_sample(3)
        assert len(sample_dict) == 3

        # Test DBFile direct access
        db_file = DBFile(db_dir / "ethanol.npy")
        assert db_file.get_shape() == shape
        assert db_file.getData().shape == shape

        # Test CSV export and CSV directory loading
        csv_dir = tmp_path / "csv_ohe"
        csv_dir.mkdir()
        csv_path = csv_dir / "ethanol.csv"
        np.savetxt(csv_path, arr, delimiter=",", fmt="%d")

        read_arr = read_one_hot_encoding_csv(str(csv_path))
        np.testing.assert_array_equal(read_arr[0], arr)

        dataset, in_shape = load_one_hot_encoding_csv_dir(str(csv_dir))
        assert "ethanol" in dataset

        db_csv = OHEDatabase()
        db_csv.load_ohe_db_from_csv_dir(str(csv_dir))
        assert "ethanol" in db_csv.X

    def test_ohe_model_builders_smoke(self):
        """Smoke test standalone builders for OHE models."""
        dshape = (32, 28)
        m1 = build_ohe_example_model(dshape, nfilters=4, nunits=8)
        assert m1 is not None

        # Both snake_case and CamelCase aliases
        m2 = example_build_2d_data_model(dshape, input_shape2=2, nfilters=4, nunits=8)
        m2_alias = example_build_2DData_model(dshape, input_shape2=2, nfilters=4, nunits=8)
        assert m2 is not None
        assert m2_alias is not None

        m3 = build_ohe_gridsearch_model(dshape, n_conv=1, nfilters=4, ndense_layers=1, nunits=8)
        assert m3 is not None

    def test_ohe_nntrain_unimodal_and_multimodal_smoke(self, tmp_path, synthetic_ohe_environment):
        """Smoke test OHE NNTrain simplerun in unimodal and multimodal modes."""
        _, db_dir, target, dx = synthetic_ohe_environment

        # 1. Unimodal simplerun (OHE matrix only)
        trainer_unimodal = OHENNTrain(str(db_dir), target)
        trainer_unimodal.verbose = 0
        feats, tgts, keys = trainer_unimodal.gen_data(list(target.keys())[:4])
        assert feats.ndim == 4
        assert len(tgts) == 4
        assert len(keys) == 4

        m_uni_path = tmp_path / "ohe_unimodal.keras"
        model_uni, metrics_uni = trainer_unimodal.simplerun(
            batch_size=2,
            num_epochs=1,
            nfilters=4,
            nunits=8,
            random_state=42,
            mout=str(m_uni_path),
        )
        assert model_uni is not None
        assert "mse" in metrics_uni
        assert "mae" in metrics_uni
        assert "r2" in metrics_uni
        assert m_uni_path.exists()

        # 2. Multimodal simplerun (OHE matrix + tabular descriptors)
        trainer_multimodal = OHENNTrain(str(db_dir), target, dx=dx, n_descs=2)
        trainer_multimodal.verbose = 0
        m_feats, m_tgts, m_keys = trainer_multimodal.gen_data(list(target.keys())[:4])
        assert isinstance(m_feats, list)
        assert len(m_feats) == 2
        assert m_feats[0].ndim == 4  # OHE image
        assert m_feats[1].ndim == 2  # Descriptors
        assert len(m_tgts) == 4

        m_multi_path = tmp_path / "ohe_multimodal.keras"
        model_multi, metrics_multi = trainer_multimodal.simplerun(
            batch_size=2,
            num_epochs=1,
            nfilters=4,
            nunits=8,
            random_state=42,
            mout=str(m_multi_path),
        )
        assert model_multi is not None
        assert "mse" in metrics_multi
        assert m_multi_path.exists()

    def test_ohe_model_predictor_smoke(self, tmp_path, synthetic_ohe_environment):
        """Smoke test OHE ModelPredictor inference engine."""
        smi_file, db_dir, target, dx = synthetic_ohe_environment

        # Train a tiny unimodal model and save it
        trainer = OHENNTrain(str(db_dir), target)
        trainer.verbose = 0
        m_path = tmp_path / "pred_ohe_model.keras"
        trainer.simplerun(
            batch_size=2,
            num_epochs=1,
            nfilters=4,
            nunits=8,
            mout=str(m_path),
        )

        # Predict using OHE ModelPredictor
        predictor = OHEModelPredictor(
            mpath=str(m_path),
            smiles_path=str(smi_file),
            padding=32,
        )
        pout = tmp_path / "ohe_preds.csv"
        stats = predictor.predict(pout=str(pout))
        assert len(stats) > 0
        for k, v in stats.items():
            assert "mean" in v
            assert "std" in v
            assert "min" in v
            assert "max" in v
        assert pout.exists()


class TestImages2DAlgorithmsSmoke:
    """Smoke tests for 2D convolutional images and activation visualization."""

    def test_2d_cnn_model_smoke(self):
        """Smoke test 2D CNN model construction, training, and inference on synthetic images."""
        model = make_2d_cnn_model(input_shape=(32, 32, 3), n_classes=1, learning_rate=0.001)
        assert model.input_shape == (None, 32, 32, 3)

        x_synthetic = np.random.uniform(0.0, 1.0, size=(4, 32, 32, 3)).astype(np.float32)
        y_synthetic = np.array([[1.0], [0.0], [1.0], [0.0]], dtype=np.float32)

        history = model.fit(x_synthetic, y_synthetic, epochs=1, batch_size=2, verbose=0)
        assert "loss" in history.history

        preds = model.predict(x_synthetic, verbose=0)
        assert preds.shape == (4, 1)

    def test_2d_visualization_pipeline_smoke(self, tmp_path):
        """Smoke test image loading, activation extraction, grid generation, and visualization."""
        # Create synthetic test image and save to disk
        img_arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        img_path = tmp_path / "synthetic_molecule.png"
        img.save(img_path)

        # Test load_and_preprocess_image
        tensor = load_and_preprocess_image(str(img_path), target_size=(32, 32))
        assert tensor.shape == (1, 32, 32, 3)
        assert tensor.dtype == np.float32

        # Build a small model to test layer activations
        model = make_2d_cnn_model(input_shape=(32, 32, 3), n_classes=1)
        names, acts = extract_layer_activations(model, tensor, n_layers=2)
        assert len(names) == len(acts) == 2

        # Test create_activation_grid
        grid = create_activation_grid(acts[0], images_per_row=4)
        assert grid is not None
        assert grid.ndim == 2

        # Test visualize_layers with file output
        save_prefix = str(tmp_path / "act_vis")
        visualize_layers(model, tensor, n_layers=2, images_per_row=4, save_path=save_prefix)
        generated_files = list(tmp_path.glob("act_vis_*.png"))
        assert len(generated_files) >= 1


class TestVoxels3DAlgorithmsSmoke:
    """Smoke tests for 3D Voxel modeling, OpenDX format, and 3D ResNets."""

    @pytest.fixture
    def synthetic_voxels_environment(self, tmp_path):
        """Prepare synthetic OpenDX files and voxel database."""
        db_dir = tmp_path / "voxels_db"
        db_dir.mkdir()
        mol_keys = ["m1", "m2", "m3", "m4"]
        voxels_data = np.random.uniform(0.0, 1.0, size=(2, 16, 16, 16)).astype(np.float32)

        for k in mol_keys:
            save_voxel_database(str(db_dir), k, voxels_data)

        target_file = tmp_path / "targets_3d.csv"
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("Molecule,Target\n")
            for i, k in enumerate(mol_keys):
                f.write(f"{k},{float(i + 1.0)}\n")

        return db_dir, target_file, mol_keys

    def test_opendx_and_voxel_objects_smoke(self, tmp_path):
        """Smoke test OpenDX file IO and VoxelObject data handling."""
        grid = np.random.uniform(0.0, 5.0, size=(4, 4, 4)).astype(np.float32)
        origin = [0.0, 0.0, 0.0]
        delta = [1.0, 1.0, 1.0]
        dx_file = tmp_path / "grid.dx"

        write_open_dx(grid, origin, delta, str(dx_file))
        assert dx_file.exists()

        g_read, o_read, d_read = read_open_dx(str(dx_file))
        np.testing.assert_allclose(g_read, grid, atol=1e-5)
        np.testing.assert_allclose(o_read, origin, atol=1e-5)
        np.testing.assert_allclose(d_read, delta, atol=1e-5)

        vobj = VoxelObject([str(dx_file)])
        assert vobj.name == "grid"
        assert len(vobj.voxels) == 1
        assert vobj.voxels[0].shape == (4, 4, 4)

        vdb_dir = tmp_path / "test_vdb"
        vdb_dir.mkdir()
        save_voxel_database(str(vdb_dir), "mol_a", grid)
        loaded_vdb = load_voxel_database(str(vdb_dir))
        assert "mol_a" in loaded_vdb

    def test_voxels_model_builders_smoke(self):
        """Smoke test 3D CNN, 3D ResNet, and Resnet3DBuilder architectures."""
        m_cnn = build_voxels_3d_model("channels_last", (16, 16, 16, 1), 1, 8, 4)
        assert m_cnn is not None

        m_res = resnet3d_model((16, 16, 16, 1))
        assert m_res is not None

        m_r18 = Resnet3DBuilder.build_resnet_18((16, 16, 16, 1), 1)
        assert m_r18 is not None

    def test_voxels3d_model_pipeline_smoke(self, synthetic_voxels_environment):
        """Smoke test Voxels3DModel dataset generation and training simplerun."""
        db_dir, target_file, mol_keys = synthetic_voxels_environment

        vmodel = Voxels3DModel(str(target_file), str(db_dir), random_state=42)
        assert sorted(vmodel.get_available_keys()) == sorted(mol_keys)

        x_ds, y_ds = vmodel.make_dataset(mol_keys[:2], nrotations=1)
        assert len(x_ds) == 2
        assert len(y_ds) == 2

        # Smoke test voxels simplerun
        try:
            voxels_simplerun(
                db=str(db_dir),
                csv_target=str(target_file),
                csv_descriptors=None,
                num_epochs=1,
                n_rot_train=1,
                train_steps_per_epoch_=1,
                n_rotation_test=1,
                test_steps_per_epoch_=1,
                ndense_layers=1,
                nunits=8,
                nfilters=4,
                random_state=42,
            )
        finally:
            statconf = Path("statconf.csv")
            if statconf.exists():
                statconf.unlink()


class TestAnalyticsAndUtilityAlgorithmsSmoke:
    """Smoke tests for Analytics (FeatureImportance) and Utility algorithms."""

    def test_feature_importance_synthetic_smoke(self, tmp_path):
        """Smoke test Breiman-Fisher permutation feature importance on synthetic data."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float32)
        # Target linearly dependent on features with an offset to avoid zero initial error
        y = (2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + 0.2).reshape(-1, 1)
        features = ["feat_a", "feat_b", "feat_c"]

        class LinearDummyModel:
            def predict(self, data, **kwargs):
                return (2.0 * data[:, 0] - 1.5 * data[:, 1] + 0.5 * data[:, 2]).reshape(-1, 1)

        model = LinearDummyModel()
        fi = FeatureImportance(model, X, y, features, iterations=2, random_state=42)
        results = fi.calculate(verbose=0)
        assert len(results) == 3
        for f in features:
            assert f in results
            assert len(results[f]) == 2

        feat_mae_quantiles = calc_feat_imp_final_results(results["feat_a"]["mae"])
        assert len(feat_mae_quantiles) == 5

        out_csv = tmp_path / "fi_results.csv"
        write_feature_importance(results, str(out_csv))
        assert out_csv.exists()

    def test_compound_selection_smoke(self):
        """Smoke test MDC and DISC diverse compound selection algorithms."""
        d_mat = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, 0.0, 1.5, 2.5, 3.5],
                [2.0, 1.5, 0.0, 1.2, 2.2],
                [3.0, 2.5, 1.2, 0.0, 1.1],
                [4.0, 3.5, 2.2, 1.1, 0.0],
            ]
        )

        # MDC selection
        mdc = MDC(d_mat, nobjects=2)
        mdc_selected = mdc.select()
        assert len(mdc_selected) == 2
        assert len(set(mdc_selected)) == 2

        # DISC selection
        disc = DISC(d_mat, method="Max", nobjects=2)
        disc_selected = disc.select()
        assert len(disc_selected) == 2
        assert len(set(disc_selected)) == 2

    def test_molecule_and_point_smoke(self):
        """Smoke test 3D Molecule and Point geometric representation."""
        p1 = POINT(1.0, 2.0, 3.0)
        p2 = POINT(4.0, 5.0, 6.0)
        mol = MOLECULE()
        mol.name = "test_mol"
        mol.coord.append(p1)
        mol.coord.append(p2)
        assert mol.get_natoms() == 2
        assert mol.name == "test_mol"

    def test_exp_pred_plot_smoke(self, monkeypatch):
        """Smoke test experimental vs predicted plotting utility."""
        monkeypatch.setattr(plt, "show", lambda: None)
        y_exp = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.2, 3.8]
        exp_pred_plot(y_exp, y_pred)

    def test_model_validation_splitters_smoke(self):
        """Smoke test TrainTestSplit and RepeatedKFold splitters."""
        items = ["a", "b", "c", "d", "e", "f"]
        train_items, test_items = TrainTestSplit(items, test_size=0.5, random_state=42)
        assert len(train_items) + len(test_items) == len(items)

        splits = list(RepeatedKFold(items, n_splits=2, n_repeats=1, random_state=42))
        assert len(splits) == 2
