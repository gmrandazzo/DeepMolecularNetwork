#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""Neural network model architectures and training pipelines for One-Hot Encoded data."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from deepmolecularnetwork.framework import (
    Model,
    Sequential,
    create_adam,
    layers,
    save_model_compat,
)
from deepmolecularnetwork.ohe.database import OHEDatabase
from deepmolecularnetwork.utility.modelhelpers import GetKerasModel, GetTrainTestFnc
from deepmolecularnetwork.utility.modelvalidation import TrainTestSplit

logger = logging.getLogger(__name__)


def example_build_model(
    dshape: tuple[int, ...],
    nfilters: int = 32,
    nunits: int = 64,
    learning_rate: float = 0.001,
) -> Any:
    """Build a single-branch 2D convolutional model for one-hot encoded matrix input."""
    input_shape = (dshape[0], dshape[1], 1)
    model = Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(nfilters, kernel_size=(1, 4), strides=(1, 1), padding="same"),
            layers.Activation("relu"),
            layers.Conv2D(nfilters, kernel_size=(1, 2), strides=(1, 1), padding="same"),
            layers.Activation("relu"),
            layers.Conv2D(nfilters, kernel_size=(1, 2), strides=(1, 1), padding="same"),
            layers.Activation("relu"),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(nunits),
            layers.Activation("relu"),
            layers.Dense(nunits),
            layers.Activation("relu"),
            layers.Dense(1),
        ]
    )
    optimizer = create_adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mse", "mae"])
    return model


def example_build_2d_data_model(
    dshape: tuple[int, ...],
    input_shape2: int,
    nfilters: int = 32,
    nunits: int = 64,
    learning_rate: float = 0.0001,
) -> Any:
    """Build a dual-branch neural network model accepting OHE matrix and tabular descriptors."""
    input_shape1 = (dshape[0], dshape[1], 1)

    in1 = layers.Input(shape=input_shape1, name="ohe_input")
    m1 = layers.Conv2D(nfilters, kernel_size=(1, 6), strides=(1, 2), padding="same")(in1)
    m1 = layers.Activation("relu")(m1)
    m1 = layers.Conv2D(nfilters, kernel_size=(1, 6), strides=(1, 2), padding="same")(m1)
    m1 = layers.Activation("relu")(m1)
    m1 = layers.AveragePooling2D(pool_size=(1, 2))(m1)
    m1 = layers.Dropout(0.3)(m1)
    m1 = layers.Flatten()(m1)

    in2 = layers.Input(shape=(input_shape2,), name="tabular_input")
    m2 = layers.Dense(nunits)(in2)
    m2 = layers.Activation("relu")(m2)
    m2 = layers.Dense(nunits)(m2)
    m2 = layers.Activation("relu")(m2)

    concat = layers.Concatenate()([m1, m2])
    out = layers.Dense(nunits)(concat)
    out = layers.Activation("relu")(out)
    out = layers.Dense(1)(out)

    fmodel = Model(inputs=[in1, in2], outputs=out)
    optimizer = create_adam(learning_rate=learning_rate)
    fmodel.compile(optimizer=optimizer, loss="mse", metrics=["mse", "mae"])
    return fmodel


# Backward compatibility alias
example_build_2DData_model = example_build_2d_data_model


def build_gridsearch_model(
    dshape: tuple[int, ...],
    n_conv: int = 2,
    nfilters: int = 32,
    ndense_layers: int = 2,
    nunits: int = 64,
    learning_rate: float = 0.0001,
) -> Any:
    """Configurable architecture builder for hyperparameter grid search."""
    input_shape = (dshape[0], dshape[1], 1)
    model = Sequential()
    model.add(layers.Input(shape=input_shape))

    for _ in range(n_conv):
        model.add(layers.Conv2D(nfilters, kernel_size=(1, 4), strides=(1, 1), padding="same"))
        model.add(layers.Activation("relu"))

    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())

    for _ in range(ndense_layers):
        model.add(layers.Dense(nunits))
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(0.1))

    model.add(layers.Dense(1))
    optimizer = create_adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mse", "mae"])
    return model


class NNTrain:
    """Training pipeline for One-Hot Encoded and multi-modal molecular networks."""

    _LEGACY_ALIASES = {
        "GenData": "gen_data",
    }

    def __getattr__(self, name: str) -> Any:
        if name in self._LEGACY_ALIASES:
            return getattr(self, self._LEGACY_ALIASES[name])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __init__(
        self,
        ohedb_path: str | Path,
        target: dict[str, Any],
        dx: dict[str, Any] | None = None,
        n_descs: int | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            ohedb_path: Path to OHEDatabase directory with .npy files.
            target: Mapping of molecule key to target value(s).
            dx: Optional dictionary of tabular descriptors.
            n_descs: Number of tabular descriptors.
        """
        self.db = OHEDatabase()
        self.db.load_ohe_db(ohedb_path)
        self.target = target
        self.tgtshape = 1
        if target:
            first_val = next(iter(target.values()))
            if isinstance(first_val, (list, tuple, np.ndarray)):
                self.tgtshape = len(first_val)

        self.dx = dx
        if n_descs is not None:
            self.n_descs = n_descs
        elif dx:
            self.n_descs = len(next(iter(dx.values())))
        else:
            self.n_descs = 0
        self.verbose = 1

    def _gen_multimodal_data(
        self, keys: list[str]
    ) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
        assert self.dx is not None
        ret_keys: list[str] = []
        batch_x_ohe: list[np.ndarray] = []
        batch_x_desc: list[Any] = []
        batch_y: list[Any] = []

        for key in keys:
            if key not in self.target or key not in self.dx or key not in self.db.X:
                continue
            tval = self.target[key]
            cfeat = self.dx[key]
            smiohes = self.db.X[key].getData()
            smiohe = smiohes if getattr(smiohes, "ndim", 2) == 2 else smiohes[0]
            batch_y.append(tval)
            batch_x_ohe.append(smiohe)
            batch_x_desc.append(cfeat)
            ret_keys.append(key)

        arr_ohe = np.asarray(batch_x_ohe, dtype=np.float32)
        if arr_ohe.ndim == 3:
            arr_ohe = arr_ohe[:, :, :, np.newaxis]
        arr_desc = np.asarray(batch_x_desc, dtype=np.float32)
        arr_y = np.asarray(batch_y, dtype=np.float32)
        return [arr_ohe, arr_desc], arr_y, ret_keys

    def _gen_unimodal_data(self, keys: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        ret_keys: list[str] = []
        batch_features: list[np.ndarray] = []
        batch_target: list[Any] = []

        for key in keys:
            if key not in self.target or key not in self.db.X:
                continue
            tval = self.target[key]
            smiohes = self.db.X[key].getData()
            smiohe = smiohes if getattr(smiohes, "ndim", 2) == 2 else smiohes[0]
            batch_target.append(tval)
            batch_features.append(smiohe)
            ret_keys.append(key)

        arr_feat = np.asarray(batch_features, dtype=np.float32)
        if arr_feat.ndim == 3:
            arr_feat = arr_feat[:, :, :, np.newaxis]
        arr_y = np.asarray(batch_target, dtype=np.float32)
        return arr_feat, arr_y, ret_keys

    def gen_data(
        self, keys: list[str]
    ) -> tuple[np.ndarray | list[np.ndarray], np.ndarray, list[str]]:
        """Extract batches of features and targets for given molecule keys.

        Args:
            keys: List of molecule keys to assemble.

        Returns:
            Tuple of (features, target_array, valid_keys).
        """
        if self.dx is not None:
            return self._gen_multimodal_data(keys)
        return self._gen_unimodal_data(keys)

    def simplerun(
        self,
        batch_size: int = 32,
        num_epochs: int = 50,
        nfilters: int = 32,
        nunits: int = 64,
        random_state: int = 42,
        mout: str | None = None,
    ) -> tuple[Any, dict[str, float]]:
        """Perform a single train/validation split run."""
        ttfn = GetTrainTestFnc() or TrainTestSplit
        all_keys = list(self.target.keys())
        train_keys, test_keys = ttfn(all_keys, test_size=0.20, random_state=random_state)

        if self.db.input_shape is None:
            raise ValueError(
                "Database input shape is not defined. Ensure OHEDatabase is properly loaded."
            )
        input_shape = self.db.input_shape

        custom_builder = GetKerasModel()
        if self.dx is not None:
            if custom_builder is not None:
                model = custom_builder(input_shape, self.n_descs, nfilters, nunits)
            else:
                model = example_build_2d_data_model(input_shape, self.n_descs, nfilters, nunits)
        else:
            if custom_builder is not None:
                model = custom_builder(input_shape, nfilters, nunits)
            else:
                model = example_build_model(input_shape, nfilters, nunits)

        x_train, y_train, _ = self.gen_data(train_keys)
        x_test, y_test, _ = self.gen_data(test_keys)

        model.fit(
            x_train,
            y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=self.verbose,
        )

        ypred_test = model.predict(x_test)
        metrics = {
            "mse": float(mse(y_test, ypred_test)),
            "mae": float(mae(y_test, ypred_test)),
            "r2": float(r2_score(y_test, ypred_test)),
        }

        if mout is not None:
            save_model_compat(model, mout)

        return model, metrics
