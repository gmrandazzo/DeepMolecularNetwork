#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""2D Convolutional Neural Network architectures for molecular images."""

from typing import Any

from deepmolecularnetwork.framework import create_adam, layers, models


def make_2d_cnn_model(
    input_shape: tuple[int, int, int] = (150, 150, 3),
    n_classes: int = 1,
    learning_rate: float = 1e-4,
) -> Any:
    """Build and compile a 2D CNN architecture for image classification.

    Args:
        input_shape: Dimensions (height, width, channels) of input images.
        n_classes: Number of prediction targets/classes. Defaults to 1 (binary classification).
        learning_rate: Initial optimizer learning rate.

    Returns:
        Compiled Keras Sequential model.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(128, activation="relu"),
            layers.Dense(n_classes, activation="sigmoid" if n_classes == 1 else "softmax"),
        ]
    )

    optimizer = create_adam(learning_rate=learning_rate)
    loss = "binary_crossentropy" if n_classes == 1 else "sparse_categorical_crossentropy"
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return model


# Alias for legacy compatibility
make_model = make_2d_cnn_model
