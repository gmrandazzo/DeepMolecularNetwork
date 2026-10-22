#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""Model prediction module and CLI for 3D CNN Voxel models."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

from deepmolecularnetwork.utility.io import ReadDescriptors
from deepmolecularnetwork.utility.modelhelpers import LoadKerasModels
from deepmolecularnetwork.voxels.voxels import LoadVoxelDatabase


class ModelPredictor:
    """Make predictions using trained 3D Convolutional Neural Network models."""

    def __init__(self, mpath: str, db: str, csv_descriptors: str | None = None):
        """Initialize ModelPredictor with model path, voxel database, and optional descriptors."""
        self.mpath = mpath
        self.voxels = LoadVoxelDatabase(db)
        if len(self.voxels) == 0:
            raise ValueError(f"No voxels loaded from database: {db}")

        print("Max Conformations %d " % (len(list(self.voxels.values())[0])))
        if csv_descriptors is not None:
            self.other_descriptors, self.nfeatures, self.header = ReadDescriptors(csv_descriptors)
        else:
            self.other_descriptors = None
            self.nfeatures = 0
            self.header = None

        self.input_shape, self.conv3d_chtype = self.get_input_voxel_shape()

        all_keys = list(self.voxels.keys())
        if self.other_descriptors is not None:
            desc_keys = set(self.other_descriptors.keys())
            self.keys = [k for k in all_keys if k in desc_keys]
        else:
            self.keys = all_keys

    def get_input_voxel_shape(self) -> tuple[tuple[int, ...], str]:
        """Compute the input voxel grid shape and channel dimension format."""
        input_shape: tuple[int, ...] = ()
        for voxel in self.voxels.values():
            if len(input_shape) == 0:
                input_shape = voxel[0][0].shape
            else:
                for i in range(len(input_shape)):
                    if voxel[0][0].shape[i] != input_shape[i]:
                        raise ValueError(
                            f"Input shape mismatch: expected {input_shape}, got {voxel[0][0].shape}"
                        )

        shape_list = list(input_shape)
        if len(shape_list) == 3:
            shape_list.append(1)
            conv3d_chtype = "channels_last"
        else:
            conv3d_chtype = "channels_first"

        print(f"Input shape: {shape_list}")
        print("Channel type: %s" % (conv3d_chtype))
        return tuple(shape_list), conv3d_chtype

    def voxel_data_generator(
        self, random_voxel_rotations: int
    ) -> np.ndarray | list[np.ndarray]:
        """Generate test batches with rototranslations for all molecules."""
        if "last" in self.conv3d_chtype:
            batch_features = np.zeros(
                (0, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            )
        else:
            batch_features = np.zeros(
                (0, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
            )

        batch_desc_features = np.zeros((0, self.nfeatures)) if self.other_descriptors else None

        random.seed(datetime.now().microsecond)
        selected_confs: set[str] = set()

        for key in self.keys:
            max_conf = len(self.voxels[key])
            max_rot = len(self.voxels[key][0])
            total_available = max_rot * max_conf

            if random_voxel_rotations >= total_available:
                for c in range(max_conf):
                    for i in range(max_rot):
                        voxel = self.voxels[key][c][i]
                        batch_features = np.vstack([batch_features, [voxel]])
                        if self.other_descriptors is not None and batch_desc_features is not None:
                            batch_desc_features = np.vstack(
                                [batch_desc_features, self.other_descriptors[key]]
                            )
            else:
                for _ in range(random_voxel_rotations):
                    rconf = random.randint(0, max_conf - 1)
                    rrot = random.randint(0, max_rot - 1)
                    tag = f"{key}-{rconf}-{rrot}"
                    while tag in selected_confs:
                        rconf = random.randint(0, max_conf - 1)
                        rrot = random.randint(0, max_rot - 1)
                        tag = f"{key}-{rconf}-{rrot}"
                    selected_confs.add(tag)

                    voxel = self.voxels[key][rconf][rrot]
                    batch_features = np.vstack([batch_features, [voxel]])
                    if self.other_descriptors is not None and batch_desc_features is not None:
                        batch_desc_features = np.vstack(
                            [batch_desc_features, self.other_descriptors[key]]
                        )

        if "last" in self.conv3d_chtype and batch_features.ndim == 4:
            batch_features = batch_features.reshape(
                batch_features.shape[0],
                batch_features.shape[1],
                batch_features.shape[2],
                batch_features.shape[3],
                1,
            )

        if batch_desc_features is not None:
            return [batch_features, batch_desc_features]
        return batch_features

    def predict(self, n_rotations: int = 100, pout: str = "predictions.csv") -> dict[str, list[float]]:
        """Run predictions on the voxel database using the models found at self.mpath."""
        predictions: dict[str, list[float]] = {key: [] for key in self.keys}
        print("N. rotations: %d" % (n_rotations))
        x_topred = self.voxel_data_generator(n_rotations)

        models = LoadKerasModels(self.mpath)
        for model in models:
            y_pred = list(model.predict(x_topred))

            k = 0
            c = 0
            for i in range(len(y_pred)):
                val = float(np.squeeze(y_pred[i]))
                predictions[self.keys[k]].append(val)
                if c == n_rotations - 1:
                    k += 1
                    c = 0
                else:
                    c += 1

        with open(pout, "w") as fo:
            fo.write("Molecule,Mean,Std,Min,Max\n")
            for key in self.keys:
                if len(predictions[key]) > 0:
                    ypavg = float(np.mean(predictions[key]))
                    ystdev = float(np.std(predictions[key]))
                    y_min = float(np.min(predictions[key]))
                    y_max = float(np.max(predictions[key]))
                    fo.write(f"{key},{ypavg:.4f},{ystdev:.4f},{y_min:.4f},{y_max:.4f}\n")

        all_y_pout = f"all_y_{pout}" if not os.path.dirname(pout) else os.path.join(
            os.path.dirname(pout), f"all_y_{os.path.basename(pout)}"
        )
        with open(all_y_pout, "w") as fo:
            for key in self.keys:
                if len(predictions[key]) > 0:
                    preds_str = ",".join(f"{val:.4f}" for val in predictions[key])
                    fo.write(f"{key},{preds_str}\n")

        return predictions


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for 3D CNN Voxel model predictions."""
    p = argparse.ArgumentParser(
        description="Predict target values for molecules in a 3D Voxel database using trained 3D CNN models."
    )
    p.add_argument("--db", default=None, type=str, help="Path to 3D Voxel database")
    p.add_argument("--minp", default=None, type=str, help="Path to trained model or model directory")
    p.add_argument(
        "--desc_csv",
        default=None,
        type=str,
        help="Optional molecular descriptors CSV file for multimodal models",
    )
    p.add_argument(
        "--n_rotations",
        default=100,
        type=int,
        help="Number of random rotations to evaluate per molecule (default: 100)",
    )
    p.add_argument("--pout", default=None, type=str, help="Output CSV prediction path")

    args = p.parse_args(argv)

    if args.pout is None or args.db is None or args.minp is None:
        p.print_help()
        print("\nERROR! --db, --minp, and --pout are required.")
        return 1

    mp = ModelPredictor(args.minp, args.db, args.desc_csv)
    mp.predict(args.n_rotations, args.pout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
