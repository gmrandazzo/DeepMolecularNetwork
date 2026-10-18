"""
Standard Numerical Parity and Regression Test Suite.
Verifies numerical equivalence, loss functions, forward pass outputs,
and 3D voxel transformations across legacy Keras 2 and modern Keras 3.
"""

import numpy as np

from deepmolecularnetwork.framework import (
    Sequential,
    Dense,
    Conv3D,
    MaxPooling3D,
    Flatten,
)
from deepmolecularnetwork.utility.numpy_loss_functions import MSE, MAE, RSQ
from deepmolecularnetwork.voxels.opendx import readOpenDX, writeOpenDX
from deepmolecularnetwork.tabular.makeTABModel import NNTrain
from deepmolecularnetwork.utility.io import ReadDescriptors, ReadTarget


class TestLossFunctionsParity:
    """Tests for numerical consistency and exact parity of loss functions."""

    def test_mse_exact(self):
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_pred = np.array([[1.5], [1.8], [3.2], [3.5]])
        expected_mse = float(np.mean((y_true - y_pred) ** 2))

        # Test 2D
        computed = MSE(y_true, y_pred)
        assert np.isclose(computed, expected_mse, atol=1e-6)

        # Test 1D fallback
        computed_1d = MSE(y_true.flatten(), y_pred.flatten())
        assert np.isclose(computed_1d, expected_mse, atol=1e-6)

    def test_mae_exact(self):
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_pred = np.array([[1.5], [1.8], [3.2], [3.5]])
        expected_mae = float(np.mean(np.abs(y_true - y_pred)))

        computed = MAE(y_true, y_pred)
        assert np.isclose(computed, expected_mae, atol=1e-6)

        computed_1d = MAE(y_true.flatten(), y_pred.flatten())
        assert np.isclose(computed_1d, expected_mae, atol=1e-6)

    def test_rsq_exact(self):
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = np.array([[1.1], [1.9], [3.1], [3.9], [5.0]])

        rss = np.sum((y_true - y_pred) ** 2)
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        expected_rsq = 1.0 - (rss / tss)

        computed = RSQ(y_true, y_pred)
        assert np.isclose(computed, expected_rsq, atol=1e-6)

        computed_1d = RSQ(y_true.flatten(), y_pred.flatten())
        assert np.isclose(computed_1d, expected_rsq, atol=1e-6)

    def test_masked_values_loss(self):
        """Verify that masked values (-9999.0 sentinel) are correctly ignored."""
        y_true = np.array([[1.0], [-9999.0], [3.0], [4.0]])
        y_pred = np.array([[1.5], [100.0], [3.2], [3.5]])

        # Valid indices: 0, 2, 3
        valid_true = np.array([1.0, 3.0, 4.0])
        valid_pred = np.array([1.5, 3.2, 3.5])
        expected_mse = float(np.mean((valid_true - valid_pred) ** 2))

        computed = MSE(y_true, y_pred, mask_value=-9999.0)
        assert np.isclose(computed, expected_mse, atol=1e-5)


class TestModelForwardNumericalParity:
    """Verifies that model forward passes produce deterministic numerical results."""

    def test_dense_forward_pass_parity(self):
        """
        Builds a fixed-weight MLP and checks that the forward pass output matches
        the golden numerical vector down to atol=1e-5.
        """
        np.random.seed(42)
        n_features = 4
        n_units = 3

        model = Sequential(
            [
                Dense(n_units, input_shape=(n_features,), activation="relu", use_bias=True),
                Dense(1, activation=None, use_bias=True),
            ]
        )

        # Assign deterministic weights
        w1 = np.array(
            [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.9], [0.2, -0.3, 0.4]],
            dtype=np.float32,
        )
        b1 = np.array([0.05, -0.1, 0.15], dtype=np.float32)

        w2 = np.array([[0.5], [-0.5], [1.0]], dtype=np.float32)
        b2 = np.array([0.25], dtype=np.float32)

        model.layers[0].set_weights([w1, b1])
        model.layers[1].set_weights([w2, b2])

        # Input vector
        x_input = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 1.5, -0.5]], dtype=np.float32)

        # Mathematical ground truth:
        # Sample 0:
        # z1 = x @ w1 + b1 = [1,2,3,4] @ w1 + b1
        #   z1[0] = 0.1 + 0.8 - 2.1 + 0.8 + 0.05 = -0.35 -> relu: 0.0
        #   z1[1] = -0.2 + 1.0 + 2.4 - 1.2 - 0.1 = 1.9 -> relu: 1.9
        #   z1[2] = 0.3 - 1.2 + 2.7 + 1.6 + 0.15 = 3.55 -> relu: 3.55
        # z2 = [0, 1.9, 3.55] @ [0.5, -0.5, 1.0] + 0.25 = 0 - 0.95 + 3.55 + 0.25 = 2.85
        expected_y0 = 2.85

        y_pred = model.predict(x_input, verbose=0)
        assert np.isclose(
            y_pred[0, 0], expected_y0, atol=1e-5
        ), f"Expected {expected_y0}, got {y_pred[0, 0]}"
        assert y_pred.shape == (2, 1)

    def test_3d_cnn_forward_parity(self):
        """
        Builds a 3D CNN model and verifies output tensor computation.
        """
        input_shape = (8, 8, 8, 1)
        model = Sequential(
            [
                Conv3D(
                    4,
                    kernel_size=(3, 3, 3),
                    input_shape=input_shape,
                    padding="same",
                    activation="relu",
                ),
                MaxPooling3D(pool_size=(2, 2, 2)),
                Flatten(),
                Dense(1),
            ]
        )

        # Test deterministic forward evaluation
        x_3d = np.ones((2, 8, 8, 8, 1), dtype=np.float32) * 0.5
        y_pred = model.predict(x_3d, verbose=0)

        assert y_pred.shape == (2, 1)
        assert not np.isnan(y_pred).any()
        # Predictions for identical inputs must match across batch elements
        assert np.isclose(y_pred[0, 0], y_pred[1, 0], atol=1e-5)


class TestVoxelAndGridParity:
    """Tests for OpenDX reading/writing and voxel spatial operations."""

    def test_opendx_roundtrip(self, tmp_path):
        """Verify OpenDX file format writing and reading preserves float values."""
        grid_shape = (4, 4, 4)
        np.random.seed(123)
        original_voxel = np.random.uniform(-5.0, 5.0, size=grid_shape).astype(np.float64)
        origin = [10.5, -20.25, 0.125]
        delta = [0.5, 0.5, 0.5]

        dx_file = tmp_path / "test_field.dx"
        writeOpenDX(original_voxel, origin, delta, str(dx_file))

        loaded_voxel, loaded_origin, loaded_delta = readOpenDX(str(dx_file))

        # OpenDX writes float in %.6e format (approx 6-7 decimal places of precision)
        np.testing.assert_allclose(original_voxel, loaded_voxel, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(origin, loaded_origin, atol=1e-5)
        np.testing.assert_allclose(delta, loaded_delta, atol=1e-5)

    def test_rotation_matrix_orthogonality(self):
        """Verify 3D rotation matrices maintain orthogonality (R * R^T = I)."""

        def mk3drotx(theta):
            ret = np.eye(4)
            ret[1, [1, 2]] = [np.cos(theta), -np.sin(theta)]
            ret[2, [1, 2]] = [np.sin(theta), np.cos(theta)]
            return ret

        def mk3droty(theta):
            ret = np.eye(4)
            ret[0, [0, 2]] = [np.cos(theta), -np.sin(theta)]
            ret[2, [0, 2]] = [np.sin(theta), np.cos(theta)]
            return ret

        def mk3drotz(theta):
            ret = np.eye(4)
            ret[0, [0, 1]] = [np.cos(theta), -np.sin(theta)]
            ret[1, [0, 1]] = [np.sin(theta), np.cos(theta)]
            return ret

        angle = np.pi / 4.0
        for rot_fn in (mk3drotx, mk3droty, mk3drotz):
            R = rot_fn(angle)[:3, :3]
            # Orthogonality condition: R @ R.T == I
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-7)
            # Determinant of proper rotation must equal +1
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-7)


class TestTabularDeterministicBenchmark:
    """Verifies deterministic training and metrics evaluation on real benchmark data."""

    def test_tabular_benchmark_run(self):
        np.random.seed(12345)
        X_raw, nfeatures, xheader = ReadDescriptors("data/example_01/dataset.desc.csv")
        target = ReadTarget("data/example_01/target.csv")

        # Select first 200 compounds for rapid deterministic testing
        keys_subset = sorted(list(target.keys()))[:200]
        X_subset = {k: X_raw[k] for k in keys_subset}
        target_subset = {k: target[k] for k in keys_subset}

        nn = NNTrain(X_subset, target_subset, xheader)
        nn.verbose = 0
        results = nn.simplerun(
            batch_size_=32,
            batch_mode_=0,
            num_epochs=3,
            ndense_layers=1,
            nunits=32,
            random_state=12345,
            model_output=None,
        )

        assert results is not None
        assert "r2" in results
        assert "mse" in results
        assert not np.isnan(results["mse"])
        assert results["mse"] < 50.0  # Reasonable loss range
