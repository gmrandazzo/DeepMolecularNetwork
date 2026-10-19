"""Unit tests for 2D molecular image CNN model and visualization helpers."""

import numpy as np

from deepmolecularnetwork.images2d.model import make_2d_cnn_model
from deepmolecularnetwork.images2d.visualize import (
    create_activation_grid,
    extract_layer_activations,
)


def test_make_2d_cnn_model_forward_pass():
    """Test 2D CNN model construction, compilation, and inference."""
    model = make_2d_cnn_model(input_shape=(32, 32, 3), n_classes=1, learning_rate=1e-4)
    dummy_img = np.random.rand(2, 32, 32, 3).astype(np.float32)
    preds = model(dummy_img)
    assert preds.shape == (2, 1)

    # Multi-class variant
    multi_model = make_2d_cnn_model(input_shape=(32, 32, 3), n_classes=3)
    preds_multi = multi_model(dummy_img)
    assert preds_multi.shape == (2, 3)


def test_extract_activations_and_grid_creation():
    """Test extracting intermediate layer outputs and formatting activation grid."""
    model = make_2d_cnn_model(input_shape=(32, 32, 3), n_classes=1)
    dummy_img = np.random.rand(1, 32, 32, 3).astype(np.float32)

    layer_names, activations = extract_layer_activations(model, dummy_img, n_layers=2)
    assert len(layer_names) == 2
    assert len(activations) == 2
    assert activations[0].ndim == 4

    grid = create_activation_grid(activations[0], images_per_row=4)
    assert grid is not None
    assert grid.ndim == 2
    assert grid.dtype == np.uint8
