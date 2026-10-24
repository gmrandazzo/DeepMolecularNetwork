#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""Model predictor for One-Hot Encoded and multi-modal molecular networks."""

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

from deepmolecularnetwork.ohe.smiles2matrix import SMILES2MX
from deepmolecularnetwork.utility.io import ReadDescriptors
from deepmolecularnetwork.utility.modelhelpers import LoadKerasModels

logger = logging.getLogger(__name__)


def _infer_padding_from_model(mpath: str | Path) -> int | None:
    """Infer SMILES sequence padding from loaded Keras model input shape."""
    try:
        for m in LoadKerasModels(str(mpath)):
            in_shape = None
            if hasattr(m, "inputs") and m.inputs:
                in_shape = m.inputs[0].shape
            elif hasattr(m, "input_shape"):
                in_shape = m.input_shape
            if in_shape is not None and len(in_shape) >= 3:
                pad_val = in_shape[2]
                if pad_val is not None and int(pad_val) > 0:
                    return int(pad_val)
            break
    except Exception:
        pass
    return None


class ModelPredictor:
    """Inference engine for models trained on SMILES/OHE and tabular descriptors."""

    _LEGACY_ALIASES = {
        "ReadSMILES": "read_smiles",
        "GenData": "gen_data",
    }

    def __getattr__(self, name: str) -> Any:
        if name in self._LEGACY_ALIASES:
            return getattr(self, self._LEGACY_ALIASES[name])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __init__(
        self,
        mpath: str | Path,
        smiles_path: str | Path,
        csv_descriptors: str | Path | None = None,
        padding: int | None = None,
    ) -> None:
        """Initialize ModelPredictor.

        Args:
            mpath: Path to saved model file or directory of models.
            smiles_path: Path to SMILES text file (tab or whitespace separated).
            csv_descriptors: Optional CSV containing molecular descriptors.
            padding: SMILES sequence padding length (auto-inferred if None).
        """
        self.mpath = mpath
        if padding is None:
            inferred = _infer_padding_from_model(mpath)
            actual_padding = inferred if inferred is not None else 256
        else:
            actual_padding = padding

        self.s2m = SMILES2MX(actual_padding)
        self.X, self.input_shape = self.read_smiles(smiles_path)

        if csv_descriptors is not None:
            self.desc, self.nfeatures, self.header = ReadDescriptors(str(csv_descriptors))
            common_keys = [k for k in self.X.keys() if k in self.desc]
            self.keys = common_keys
        else:
            self.desc = None
            self.nfeatures = 0
            self.header = []
            self.keys = list(self.X.keys())

    def read_smiles(self, smiles_list: str | Path) -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
        """Read a SMILES file into a dictionary of 2D OHE matrices."""
        dataset: dict[str, np.ndarray] = {}
        with open(smiles_list, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split("\t") if "\t" in stripped else stripped.split()
                if len(parts) >= 2:
                    smi, name = parts[0], parts[1]
                else:
                    smi, name = parts[0], parts[0]
                dataset[name] = self.s2m.smi2array(smi)

        return dataset, self.s2m.getshape()

    def gen_data(self) -> np.ndarray | list[np.ndarray]:
        """Prepare batch tensor(s) for prediction."""
        batch_ohe = [self.X[k] for k in self.keys]
        arr_ohe = np.asarray(batch_ohe, dtype=np.float32)[:, :, :, np.newaxis]

        if self.desc is not None and self.nfeatures > 0:
            batch_desc = [self.desc[k] for k in self.keys]
            arr_desc = np.asarray(batch_desc, dtype=np.float32)
            return [arr_ohe, arr_desc]

        return arr_ohe

    def predict(self, pout: str | Path | None = None) -> dict[str, dict[str, float]]:
        """Compute predictions across loaded ensemble models.

        Args:
            pout: Optional path to write summary CSV predictions.

        Returns:
            Dictionary mapping molecule key to summary statistics (mean, std, min, max).
        """
        models = list(LoadKerasModels(str(self.mpath)))
        if not models:
            raise FileNotFoundError(f"No valid models found at '{self.mpath}'")

        x_topred = self.gen_data()
        raw_predictions: dict[str, list[float]] = {k: [] for k in self.keys}

        for model in models:
            preds = model.predict(x_topred, verbose=0)
            preds_flat = np.ravel(preds)
            for key, val in zip(self.keys, preds_flat):
                raw_predictions[key].append(float(val))

        summary: dict[str, dict[str, float]] = {}
        for key, vals in raw_predictions.items():
            if vals:
                summary[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }

        if pout is not None:
            with open(pout, "w", encoding="utf-8") as fo:
                fo.write("Molecule,Mean,Std,Min,Max\n")
                for key, stat in summary.items():
                    m, s, mn, mx = stat["mean"], stat["std"], stat["min"], stat["max"]
                    fo.write(f"{key},{m:.4f},{s:.4f},{mn:.4f},{mx:.4f}\n")

        return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for OHE ModelPredictor."""
    p = argparse.ArgumentParser(description="Predict molecular targets using OHE CNN models")
    p.add_argument("--smiles", required=True, type=str, help="Path to SMILES file")
    p.add_argument("--minp", required=True, type=str, help="Path to model file or directory")
    p.add_argument("--desc_csv", default=None, type=str, help="Optional CSV descriptors file")
    p.add_argument(
        "--padding",
        default=None,
        type=int,
        help="SMILES sequence padding length (default: auto-inferred from model or 256)",
    )
    p.add_argument("--pout", default=None, type=str, help="Prediction output CSV path")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    predictor = ModelPredictor(
        args.minp, args.smiles, csv_descriptors=args.desc_csv, padding=args.padding
    )
    predictor.predict(args.pout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
