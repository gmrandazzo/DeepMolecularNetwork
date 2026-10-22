#!/usr/bin/env python3
# (c) 2018-2023 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import argparse
from pathlib import Path
import sys
import numpy as np

from deepmolecularnetwork.utility.io import ReadDescriptors
from deepmolecularnetwork.utility.modelhelpers import LoadKerasModels, ReadDescriptorOrder


class ModelPredictor:
    def __init__(self, mpath, csv_descriptors):
        self.mpath = mpath
        self.odesc = ReadDescriptorOrder(mpath)
        self.desc, self.nfeatures, self.header = ReadDescriptors(csv_descriptors)
        self.keys = list(self.desc.keys())
        # check that all the descriptors
        # are present in the previous loaded desc
        nf = []
        self.dmap = {}
        for i, col in enumerate(self.odesc):
            if col in self.header:
                self.dmap[col] = i
            else:
                nf.append(col)
        if len(nf) > 0:
            print("Error! Missing some descriptors")
            print("Prediction not possible!")
            print(nf)
            sys.exit(1)

    def gen_data(self):
        """Generate aligned feature array for prediction."""
        batch_features = []
        for key in self.desc.keys():
            x = [0.0] * len(self.odesc)
            for i in range(len(self.desc[key])):
                if self.header[i] in self.dmap:
                    p = self.dmap[self.header[i]]
                    x[p] = self.desc[key][i]
            batch_features.append(x)
        return np.array(batch_features)

    _LEGACY_ALIASES = {
        "GenData": "gen_data",
    }

    def __getattr__(self, name: str):
        if name in self._LEGACY_ALIASES:
            return getattr(self, self._LEGACY_ALIASES[name])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def predict(self, pout):
        predictions = {key: [] for key in self.keys}

        x_topred = self.gen_data()
        for model in LoadKerasModels(self.mpath):
            y_pred = list(model.predict(x_topred))
            for i, pred in enumerate(y_pred):
                predictions[self.keys[i]].append(pred)

        pout_path = Path(pout)
        with open(pout_path, "w") as fo:
            for key, preds in predictions.items():
                if len(preds) > 0:
                    ypavg = float(np.mean(preds))
                    ystdev = float(np.std(preds))
                    y_min = float(np.min(preds))
                    y_max = float(np.max(preds))
                    fo.write(f"{key},{ypavg:.4f},{ystdev:.4f},{y_min:.4f},{y_max:.4f}\n")

        all_y_path = pout_path.parent / f"all_y_{pout_path.name}"
        with open(all_y_path, "w") as fo:
            for key, preds in predictions.items():
                if len(preds) > 0:
                    formatted_preds = [f"{float(np.squeeze(val)):.4f}" for val in preds]
                    fo.write(f"{key}," + ",".join(formatted_preds) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Make predictions on molecular descriptors using trained Tabular DNN models."
    )
    p.add_argument("--desc_csv", default=None, type=str, help="Molecule descriptors CSV")
    p.add_argument(
        "--minp", default=None, type=str, help="Path to trained model or model directory"
    )
    p.add_argument("--pout", default=None, type=str, help="Prediction output CSV")
    args = p.parse_args(argv)

    if args.pout is None or args.desc_csv is None or args.minp is None:
        p.print_help()
        print("\nUsage:")
        print("--desc_csv [Molecular descriptors file]")
        print("--minp [keras model out]")
        print("--pout [default None]")
        return 0

    mp = ModelPredictor(args.minp, args.desc_csv)
    mp.predict(args.pout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
