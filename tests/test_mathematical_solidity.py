"""Senior Machine Learning Engineer Mathematical Rigor & Numerical Solidity Suite.

Verifies mathematical invariants, metric space axioms, boundary conditions,
numerical stability under extreme scaling, SO(3) group properties, and
statistical metric soundness across DeepMolecularNetwork modules.
"""

import math
import numpy as np
import pytest

from deepmolecularnetwork.ohe.smiles2matrix import DEFAULT_SMILE_CHARS, SMILES2MX
from deepmolecularnetwork.utility.dissimilar_compound_selection import DISC, _median
from deepmolecularnetwork.utility.modelvalidation import RepeatedKFold
from deepmolecularnetwork.utility.most_descriptive_compound import MDC
from deepmolecularnetwork.utility.numpy_loss_functions import MAE, MSE, RSQ, gen_mask
from deepmolecularnetwork.voxels.voxels import VoxelObject


class TestLossMetricsMathematicalSolidity:
    """Rigorous mathematical tests for loss and regression quality metrics."""

    def test_rsq_axiomatic_properties(self):
        """Verify R² / Q² obeys fundamental regression theory axioms."""
        y_true = np.array([2.5, 3.0, 4.5, 6.0, 8.0, 9.5], dtype=np.float64)

        # 1. Perfect predictor: R² ≡ 1.0
        assert np.isclose(RSQ(y_true, y_true), 1.0, atol=1e-7)

        # 2. Trivial baseline: always predicting the mean y_bar ≡ 0.0
        y_mean_pred = np.full_like(y_true, np.mean(y_true))
        assert np.isclose(RSQ(y_true, y_mean_pred), 0.0, atol=1e-7)

        # 3. Worse than trivial baseline: R² < 0.0
        y_anti_pred = 2.0 * np.mean(y_true) - y_true
        r2_anti = RSQ(y_true, y_anti_pred)
        assert r2_anti < 0.0

        # 4. Invariance to affine shift: adding constant C leaves R² identical
        c = 1000.0
        assert np.isclose(RSQ(y_true + c, y_mean_pred + c), 0.0, atol=1e-6)

        # 5. Invariance to uniform scaling: multiplying by alpha leaves R² identical
        alpha = 3.14159
        assert np.isclose(RSQ(alpha * y_true, alpha * y_anti_pred), r2_anti, atol=1e-6)

    def test_zero_variance_target_stability(self):
        """Test constant target vectors where Total Sum of Squares (TSS) = 0."""
        # When all true values are identical, TSS is zero.
        y_const = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.1, 4.9, 5.0, 5.2])

        # Must not raise ZeroDivisionError and return bounded float
        val = RSQ(y_const, y_pred)
        assert np.isfinite(val)
        assert val == 0.0

    def test_metric_inequalities(self):
        """Verify Cauchy-Schwarz and Jensen's inequality: MAE <= RMSE <= sqrt(N)*MAE."""
        rng = np.random.default_rng(seed=42)
        y_true = rng.normal(loc=10.0, scale=2.5, size=100)
        y_pred = y_true + rng.normal(loc=0.0, scale=1.0, size=100)

        mae_val = MAE(y_true, y_pred)
        mse_val = MSE(y_true, y_pred)
        rmse_val = math.sqrt(mse_val)

        # Jensen's inequality for convex square function:
        assert mae_val <= rmse_val + 1e-9

        # Bound on max discrepancy:
        n = len(y_true)
        assert rmse_val <= math.sqrt(n) * mae_val + 1e-9

        # Symmetry: Loss(a, b) == Loss(b, a)
        assert np.isclose(MAE(y_true, y_pred), MAE(y_pred, y_true), atol=1e-7)
        assert np.isclose(MSE(y_true, y_pred), MSE(y_pred, y_true), atol=1e-7)

    def test_masked_loss_mathematical_precision(self):
        """Verify masked loss ignores missing values with zero numerical leakage."""
        sentinel = -9999.0
        y_true = np.array([1.0, sentinel, 3.0, sentinel, 5.0])
        y_pred = np.array([1.5, 999.0, 3.5, -42.0, 4.5])

        # Ground truth unmasked: y_true_valid = [1.0, 3.0, 5.0], y_pred_valid = [1.5, 3.5, 4.5]
        # Differences = [-0.5, -0.5, 0.5] -> MAE = 0.5, MSE = 0.25
        computed_mae = MAE(y_true, y_pred, mask_value=sentinel)
        computed_mse = MSE(y_true, y_pred, mask_value=sentinel)

        assert np.isclose(computed_mae, 0.5, atol=1e-7)
        assert np.isclose(computed_mse, 0.25, atol=1e-7)

        # Ensure gen_mask produces binary indicator vector:
        mask = gen_mask(y_true, mask_value=sentinel)
        assert mask == [1.0, 0.0, 1.0, 0.0, 1.0]

    def test_floating_point_extreme_ranges(self):
        """Verify numerical stability under tiny and huge floating-point scales."""
        # Tiny numbers (e.g. binding affinities in nanomolar / molar scale 1e-9)
        tiny_true = np.array([1e-9, 2e-9, 3e-9, 4e-9])
        tiny_pred = np.array([1.1e-9, 1.9e-9, 3.1e-9, 3.9e-9])
        assert np.isfinite(MSE(tiny_true, tiny_pred))
        assert np.isfinite(RSQ(tiny_true, tiny_pred))

        # Huge numbers (e.g. molecular weights / partition coefficients in large units 1e6)
        huge_true = np.array([1e6, 2e6, 3e6, 4e6])
        huge_pred = np.array([1.05e6, 1.95e6, 3.05e6, 3.95e6])
        assert np.isfinite(MSE(huge_true, huge_pred))
        assert np.isfinite(RSQ(huge_true, huge_pred))


class TestSpatialGeometryAndSO3GroupTheory:
    """Tests verifying 3D group theory (SO(3)), orthogonality, and isometries."""

    @pytest.mark.parametrize("angle_deg", [0.0, 30.0, 45.0, 90.0, 180.0, 270.0, -60.0])
    def test_so3_orthogonality_and_determinant(self, angle_deg):
        """Every 3D rotation matrix must satisfy R^T * R = I and det(R) = +1."""
        rad = math.radians(angle_deg)
        r_x = VoxelObject.mk3drotx(rad)
        r_y = VoxelObject.mk3droty(rad)
        r_z = VoxelObject.mk3drotz(rad)

        for R4 in [r_x, r_y, r_z]:
            # Full 4x4 homogeneous matrix checks
            assert np.allclose(R4.T @ R4, np.eye(4), atol=1e-9)
            assert np.isclose(np.linalg.det(R4), 1.0, atol=1e-9)

            # Pure SO(3) 3x3 rotation block
            R3 = R4[:3, :3]
            assert np.allclose(R3.T @ R3, np.eye(3), atol=1e-9)
            assert np.isclose(np.linalg.det(R3), 1.0, atol=1e-9)

    def test_isometry_euclidean_distance_invariance(self):
        """Rotations are rigid isometries: ||R*p1 - R*p2|| == ||p1 - p2||."""
        rng = np.random.default_rng(seed=123)
        p1 = rng.uniform(-10.0, 10.0, size=3)
        p2 = rng.uniform(-10.0, 10.0, size=3)
        original_dist = np.linalg.norm(p1 - p2)

        # Composite rotation
        rx = VoxelObject.mk3drotx(0.75)
        ry = VoxelObject.mk3droty(1.20)
        rz = VoxelObject.mk3drotz(-0.45)
        R3 = (rx @ ry @ rz)[:3, :3]

        p1_rot = R3 @ p1
        p2_rot = R3 @ p2
        rotated_dist = np.linalg.norm(p1_rot - p2_rot)

        assert np.isclose(original_dist, rotated_dist, atol=1e-8)
        assert np.isclose(np.dot(p1_rot, p2_rot), np.dot(p1, p2), atol=1e-8)

    def test_rotation_group_composition_and_translation(self):
        """Single-axis rotations compose additively: R(a) * R(b) == R(a + b)."""
        a, b = 0.35, 0.85
        r_a = VoxelObject.mk3drotz(a)
        r_b = VoxelObject.mk3drotz(b)
        r_combined = r_a @ r_b
        r_expected = VoxelObject.mk3drotz(a + b)

        assert np.allclose(r_combined, r_expected, atol=1e-9)

        # Inverse rotation is transpose: R(a) * R(-a) == I
        r_neg = VoxelObject.mk3drotz(-a)
        assert np.allclose(r_a @ r_neg, np.eye(4), atol=1e-9)

        # Translation group additive composition: T(x1) * T(x2) == T(x1 + x2)
        t1 = VoxelObject.mk3dtr(1.0, -2.0, 3.5)
        t2 = VoxelObject.mk3dtr(4.0, 2.0, -1.5)
        t_expected = VoxelObject.mk3dtr(5.0, 0.0, 2.0)
        assert np.allclose(t1 @ t2, t_expected, atol=1e-9)


class TestChemicalSpaceMetricSpaceAxioms:
    """Verifies distance matrix metric space axioms and deterministic sampling."""

    def test_euclidean_distance_metric_axioms(self):
        """Distance matrices must satisfy identity, non-negativity, symmetry, and triangle ineq."""
        rng = np.random.default_rng(seed=999)
        coords = rng.uniform(-5.0, 5.0, size=(10, 4))  # 10 compounds, 4 descriptors
        n = len(coords)

        # Compute Euclidean distance matrix D
        dmx = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                dmx[i, j] = np.linalg.norm(coords[i] - coords[j])

        # 1. Identity of indiscernibles: D_ii == 0
        assert np.allclose(np.diag(dmx), 0.0, atol=1e-12)

        # 2. Non-negativity: D_ij >= 0
        assert np.all(dmx >= -1e-12)

        # 3. Symmetry: D_ij == D_ji
        assert np.allclose(dmx, dmx.T, atol=1e-12)

        # 4. Triangle inequality: D_ik <= D_ij + D_jk
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert dmx[i, k] <= dmx[i, j] + dmx[j, k] + 1e-9

    def test_mdc_and_disc_selection_soundness(self):
        """Test Most Descriptive Compound and Dissimilarity selection on metric spaces."""
        rng = np.random.default_rng(seed=555)
        coords = rng.normal(loc=0.0, scale=1.0, size=(12, 3))
        dmx = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

        # MDC: selects informative subset without duplicates
        mdc = MDC(dmx, nobjects=4)
        selected_mdc = mdc.select()
        assert len(selected_mdc) == 4
        assert len(set(selected_mdc)) == 4  # All indices unique
        assert all(0 <= idx < 12 for idx in selected_mdc)

        # DISC: tests all linkage methods (Max, Min, Med, Sum)
        for method in ["Max", "Min", "Med", "Sum"]:
            disc = DISC(dmx, nobjects=3, method=method)
            selected_disc = disc.select()
            assert len(selected_disc) == 3
            assert len(set(selected_disc)) == 3
            assert all(0 <= idx < 12 for idx in selected_disc)

    def test_median_odd_and_even_soundness(self):
        """Test exactness of internal median helper on odd and even length distributions."""
        assert _median([1, 5, 2, 8, 7]) == 5.0
        assert _median([1, 2, 3, 4]) == 2.5
        assert _median([42]) == 42.0

    def test_repeated_kfold_partition_invariants(self):
        """K-fold splits must be mutually disjoint and collectively exhaustive."""
        items = [f"MOL_{i}" for i in range(25)]
        n_splits = 5
        n_repeats = 2

        splits = list(RepeatedKFold(items, n_splits=n_splits, n_repeats=n_repeats, random_state=42))
        assert len(splits) == n_splits * n_repeats

        for train_keys, val_keys, test_keys in splits:
            s_train = set(train_keys)
            s_val = set(val_keys)
            s_test = set(test_keys)

            # Mutual disjointness:
            assert s_train.isdisjoint(s_val)
            assert s_train.isdisjoint(s_test)
            assert s_val.isdisjoint(s_test)

            # Collective exhaustiveness:
            union_all = s_train.union(s_val).union(s_test)
            assert union_all == set(items)


class TestOneHotRepresentationMathematicalInvariants:
    """Verifies discrete geometry and simplex constraints of molecular representations."""

    def test_simplex_and_sparsity_constraints(self):
        """Every active column in a one-hot representation must lie on the probability simplex."""
        s2m = SMILES2MX(row_padding=24)
        smi = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        tokens = s2m.parse(smi)
        arr = s2m.smi2array(smi)

        # Shape invariant: (vocabulary_size, row_padding)
        vocab_size = len(DEFAULT_SMILE_CHARS)
        assert arr.shape == (vocab_size, 24)

        # 1. Simplex constraint: for each parsed token position, sum along vocabulary = 1
        num_tokens = len(tokens)
        active_cols_sum = np.sum(arr[:, :num_tokens], axis=0)
        assert np.all(active_cols_sum == 1)

        # 2. Sparsity / Padding invariant: columns beyond token length must be exactly zero
        padded_cols_sum = np.sum(arr[:, num_tokens:])
        assert padded_cols_sum == 0

        # 3. Binary values only: arr in {0, 1}
        assert np.all(np.isin(arr, [0.0, 1.0]))
