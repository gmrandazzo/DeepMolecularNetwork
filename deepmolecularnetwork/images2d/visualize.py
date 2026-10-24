#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""Visualization utilities for 2D convolutional network feature maps."""

import argparse
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deepmolecularnetwork.framework import load_model_compat, models


def load_and_preprocess_image(
    image_path: str, target_size: tuple[int, int] = (150, 150)
) -> np.ndarray:
    """Load image from disk and preprocess to normalized tensor."""
    with Image.open(image_path) as img:
        img_resized = img.convert("RGB").resize(target_size)
        img_tensor = np.asarray(img_resized, dtype=np.float32)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0
    return img_tensor


def extract_layer_activations(
    model: Any, img_tensor: np.ndarray, n_layers: int = 8
) -> tuple[list[str], list[np.ndarray]]:
    """Compute intermediate activations for the first n_layers of the model."""
    conv_or_pool_layers = [
        layer
        for layer in model.layers
        if "conv" in layer.name.lower() or "pool" in layer.name.lower()
    ]
    target_layers = (conv_or_pool_layers if conv_or_pool_layers else model.layers)[:n_layers]
    layer_outputs = [layer.output for layer in target_layers]
    layer_names = [layer.name for layer in target_layers]

    activation_model = models.Model(inputs=model.inputs, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor, verbose=0)
    if not isinstance(activations, list):
        activations = [activations]
    return layer_names, activations


def create_activation_grid(
    layer_activation: np.ndarray, images_per_row: int = 16
) -> np.ndarray | None:
    """Render a 4D layer activation into a 2D composite image grid."""
    if layer_activation.ndim != 4:
        return None
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = max(1, n_features // images_per_row)
    grid = np.zeros((size * n_cols, images_per_row * size), dtype=np.uint8)

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_idx = col * images_per_row + row
            if channel_idx < n_features:
                channel = layer_activation[0, :, :, channel_idx].copy()
                mean = channel.mean()
                std = channel.std() + 1e-7
                channel -= mean
                channel /= std
                channel *= 64
                channel += 128
                channel = np.clip(channel, 0, 255).astype(np.uint8)
                grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel

    return grid


def visualize_layers(
    model: Any,
    img_tensor: np.ndarray,
    n_layers: int = 8,
    images_per_row: int = 16,
    save_path: str | None = None,
) -> None:
    """Generate and display or save activation grid maps."""
    layer_names, activations = extract_layer_activations(model, img_tensor, n_layers=n_layers)

    for layer_name, layer_act in zip(layer_names, activations):
        grid = create_activation_grid(layer_act, images_per_row=images_per_row)
        if grid is None:
            continue
        plt.figure(figsize=(10, 10))
        plt.title(layer_name)
        plt.imshow(grid, cmap="viridis")
        plt.axis("off")
        if save_path:
            out_file = f"{save_path}_{layer_name}.png"
            plt.savefig(out_file, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for 2D layer visualization."""
    p = argparse.ArgumentParser(
        description="Visualize convolutional activations for 2D molecular images"
    )
    p.add_argument(
        "--model", required=True, type=str, help="Path to trained model file (.keras or .h5)"
    )
    p.add_argument("--img", required=True, type=str, help="Path to input image file")
    p.add_argument(
        "--save", default=None, type=str, help="Optional output prefix to save visualization plots"
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    model = load_model_compat(args.model)
    target_size = (150, 150)
    in_shape = getattr(model, "input_shape", None)
    if in_shape is not None and len(in_shape) >= 3:
        h = in_shape[1]
        w = in_shape[2]
        if h is not None and w is not None:
            target_size = (int(w), int(h))

    img_tensor = load_and_preprocess_image(args.img, target_size=target_size)
    visualize_layers(model, img_tensor, save_path=args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
