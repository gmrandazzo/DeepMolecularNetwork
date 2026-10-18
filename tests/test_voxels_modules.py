"""
Tests for 3D Voxel modules: opendx, voxels, resnet3d, model_builder.
"""

from pathlib import Path
import numpy as np
import pytest

from deepmolecularnetwork.voxels.opendx import (
    read_open_dx,
    write_open_dx,
    readOpenDX,
    writeOpenDX,
)
from deepmolecularnetwork.voxels.voxels import (
    VoxelObject,
    save_voxel_database,
    load_voxel_database,
)
from deepmolecularnetwork.voxels.resnet3d import Resnet3DBuilder
from deepmolecularnetwork.voxels.model_builder import rmse, rsquared
from deepmolecularnetwork.framework import K


class TestOpenDXAndVoxelObject:
    """Tests for OpenDX reading/writing and VoxelObject class."""

    @pytest.fixture
    def sample_dx(self, tmp_path):
        voxel = np.ones((4, 4, 4), dtype=np.float32) * 2.5
        origin = [0.0, 0.0, 0.0]
        delta = [1.0, 1.0, 1.0]
        dx_path = tmp_path / "sample.dx"
        write_open_dx(voxel, origin, delta, str(dx_path))
        return str(dx_path)

    def test_opendx_modern_and_legacy_api(self, tmp_path):
        voxel = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
        origin = [1.0, 2.0, 3.0]
        delta = [0.5, 0.5, 0.5]
        out_file = tmp_path / "out.dx"

        # Modern snake_case
        write_open_dx(voxel, origin, delta, str(out_file))
        v_read, o_read, d_read = read_open_dx(str(out_file))
        np.testing.assert_allclose(v_read, voxel, atol=1e-5)
        np.testing.assert_allclose(o_read, origin, atol=1e-5)
        np.testing.assert_allclose(d_read, delta, atol=1e-5)

        # Legacy CamelCase
        out_legacy = tmp_path / "out_legacy.dx"
        writeOpenDX(voxel, origin, delta, str(out_legacy))
        v_leg, o_leg, d_leg = readOpenDX(str(out_legacy))
        np.testing.assert_allclose(v_leg, voxel, atol=1e-5)

    def test_voxel_object_pipeline(self, sample_dx):
        vobj = VoxelObject([sample_dx])
        assert vobj.name == "sample"
        assert len(vobj.voxels) == 1
        assert vobj.voxels[0].shape == (4, 4, 4)

        # Flatten test
        flat = vobj.to_flatten()
        assert flat.shape == (64,)
        # Test backward-compatible alias via __getattr__
        flat_leg = vobj.toFlatten()
        assert np.array_equal(flat, flat_leg)

        # Single voxel rotation
        rot_single = vobj.voxel_rotation(vobj.voxels)
        assert len(rot_single) == 1
        assert rot_single[0].shape == (4, 4, 4)

    def test_voxel_database_io(self, tmp_path):
        db_dir = tmp_path / "vdb_dir"
        db_dir.mkdir()
        voxels_data = np.ones((2, 4, 4, 4), dtype=np.float32)

        save_voxel_database(str(db_dir), "mol1", voxels_data)
        loaded = load_voxel_database(str(db_dir))
        assert "mol1" in loaded
        assert len(loaded["mol1"]) == 1
        np.testing.assert_allclose(loaded["mol1"][0], voxels_data)


class TestResnet3DAndModelBuilder:
    """Tests for 3D ResNet architectures and custom loss functions."""

    def test_resnet3d_builder(self):
        input_shape = (8, 8, 8, 1)
        model = Resnet3DBuilder.build_resnet_18(input_shape, 1)
        assert model is not None
        assert model.input_shape == (None, 8, 8, 8, 1)
        assert model.output_shape == (None, 1)

    def test_model_builder_losses(self):
        import tensorflow as tf

        y_true = tf.constant([[1.0], [2.0], [3.0]])
        y_pred = tf.constant([[1.1], [1.9], [3.1]])

        loss_rmse = rmse(y_true, y_pred)
        assert loss_rmse is not None
        r2 = rsquared(y_true, y_pred)
        assert r2 is not None

    def test_model_architectures_smoke(self):
        from deepmolecularnetwork.voxels.model_builder import (
            build_model,
            build_2d_data_model,
            build_2d_data_model_old,
            build_shapenet10,
            build_fcn_model,
            build_custom_model,
            build_model_orig,
            build_model_,
            build_model_garbage,
            build_model_okg25s25,
            build_model_not_well_working,
            resnet_model,
            resnet3d_model,
        )

        input_shape = (16, 16, 16, 1)
        m1 = build_model("channels_last", input_shape, 1, 16, 4)
        assert m1 is not None
        m2 = build_2d_data_model("channels_last", input_shape, 5, 1, 16, 4)
        assert m2 is not None
        m3 = build_shapenet10("channels_last", input_shape, 1, 16, 4)
        assert m3 is not None
        m4 = build_fcn_model("channels_last", input_shape, 1, 16, 4)
        assert m4 is not None
        m5 = build_custom_model("channels_last", input_shape, 16, 4)
        assert m5 is not None
        m6 = build_model_orig("channels_last", input_shape, 1, 16, 4)
        assert m6 is not None
        m7 = resnet_model(input_shape)
        assert m7 is not None
        m8 = build_model_garbage("channels_last", input_shape, 1, 16, 4)
        assert m8 is not None
        m9 = build_model_okg25s25("channels_last", input_shape, 1, 16, 4)
        assert m9 is not None
        m10 = build_model_not_well_working("channels_last", input_shape, 1, 16, 4)
        assert m10 is not None
        m11 = resnet3d_model(input_shape)
        assert m11 is not None
        input_32 = (32, 32, 32, 1)
        m12 = build_model_("channels_last", input_32, 1, 16, 4)
        assert m12 is not None
        m13 = build_2d_data_model_old("channels_last", input_32, 5, 1, 16, 4)
        assert m13 is not None


class TestVoxels3DModelPipeline:
    """Tests for Voxels3DModel initialization and dataset generation."""

    def test_voxels_3d_model(self, tmp_path):
        from deepmolecularnetwork.voxels.model import Voxels3DModel

        db_dir = tmp_path / "vdb"
        db_dir.mkdir()
        voxels_data = np.ones((2, 4, 4, 4), dtype=np.float32)
        save_voxel_database(str(db_dir), "mol1", voxels_data)
        save_voxel_database(str(db_dir), "mol2", voxels_data)

        target_file = tmp_path / "target.csv"
        target_file.write_text("Molecule,Activity\nmol1,1.5\nmol2,2.5\n")

        vmodel = Voxels3DModel(str(target_file), str(db_dir), random_state=42)
        assert sorted(vmodel.get_available_keys()) == ["mol1", "mol2"]
        # Legacy alias test
        assert sorted(vmodel.GetAvailableKeys()) == ["mol1", "mol2"]

        X, y = vmodel.make_dataset(["mol1", "mol2"], nrotations=1)
        assert len(X) == 2
        assert len(y) == 2

    def test_voxels_3d_simplerun(self, tmp_path):
        from deepmolecularnetwork.voxels.model import simplerun

        db_dir = tmp_path / "vdb_sr"
        db_dir.mkdir()
        voxels_data = np.ones((2, 16, 16, 16), dtype=np.float32)
        save_voxel_database(str(db_dir), "mol1", voxels_data)
        save_voxel_database(str(db_dir), "mol2", voxels_data)
        save_voxel_database(str(db_dir), "mol3", voxels_data)
        save_voxel_database(str(db_dir), "mol4", voxels_data)

        target_file = tmp_path / "target_sr.csv"
        target_file.write_text("Molecule,Activity\nmol1,1.0\nmol2,2.0\nmol3,3.0\nmol4,4.0\n")

        try:
            simplerun(
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
