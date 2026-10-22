#!/usr/bin/env python3
# (c) 2018-2023 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import argparse
from math import ceil
from pathlib import Path
import random
import time

import numpy as np

from deepmolecularnetwork.analytics.feature_importance import (
    FeatureImportance,
    WriteFeatureImportance,
)
from deepmolecularnetwork.framework import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    K,
    LeakyReLU,
    ModelCheckpoint,
    Sequential,
    TensorBoard,
    create_adam,
    save_model_compat,
)
from deepmolecularnetwork.utility.io import (
    ReadDescriptors,
    ReadTarget,
    WriteCrossValidationOutput,
)
from deepmolecularnetwork.utility.modelhelpers import (
    GetKerasModel,
    GetLoadModelFnc,
    GetTrainTestFnc,
    GetValidationFnc,
)
from deepmolecularnetwork.utility.modelvalidation import RepeatedKFold, TrainTestSplit
from deepmolecularnetwork.utility.numpy_loss_functions import MSE, RSQ

ERR_MOL_NOT_FOUND = "Molecule %s not found"


def example_build_model(nfeatures, nunits, ndense_layers, ntargets):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(nfeatures,)))
    model.add(Dense(nunits, activation="relu"))
    model.add(Dropout(0.15))
    for _ in range(ndense_layers):
        model.add(Dense(nunits, activation="relu"))
        model.add(Dropout(0.1))
    model.add(Dense(nunits, activation="relu"))
    model.add(Dense(ntargets))
    model.compile(loss="mse", optimizer=create_adam(learning_rate=0.00005), metrics=["mse", "mae"])
    return model


def build_gridsearch_model(nfeatures, ndense_layers, nunits, activation, dropout):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(nfeatures,)))

    for _ in range(ndense_layers):
        model.add(Dense(nunits))
        if activation == "relu":
            model.add(Activation("relu"))
        elif activation == "leakyrelu":
            model.add(LeakyReLU())
        else:
            print("No activation %s found" % (activation))

        if dropout == "on":
            model.add(Dropout(0.1))

    model.add(Dense(1))
    model.compile(loss="mse", optimizer=create_adam(learning_rate=0.001), metrics=["mse", "mae"])
    return model


class NNTrain:
    def __init__(self, x_raw, target, xheader=None):
        self.x_raw = x_raw
        self.xheader = xheader
        self.target = target
        self.nfeatures = len(list(x_raw.values())[0])
        try:
            self.ntargets = len(list(target.values())[0])
        except (TypeError, IndexError):
            self.ntargets = 1
        self.ntotobj = len(list(self.x_raw.keys()))
        print("#Inst.: %d  #Feat.: %d  #Tar.: %d" % (self.ntotobj, self.nfeatures, self.ntargets))
        self.verbose = 0
        self.batch_features = None
        self.batch_target = None

    _LEGACY_ALIASES = {
        "X_raw": "x_raw",
        "GenData": "gen_data",
        "makeMatrix": "make_matrix",
        "DataGenerator": "data_generator",
        "makePrediction": "make_prediction",
    }

    def __getattr__(self, name: str):
        if name in self._LEGACY_ALIASES:
            return getattr(self, self._LEGACY_ALIASES[name])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        if name == "X_raw":
            self.__dict__["x_raw"] = value
        else:
            super().__setattr__(name, value)

    def gen_data(self, keys):
        batch_features = np.zeros((len(keys), self.nfeatures))
        batch_target = np.zeros((len(keys), self.ntargets))
        i = 0
        for key in keys:
            try:
                t = self.target[key]
                x = self.x_raw[key]
                batch_target[i] = np.copy(t)
                batch_features[i] = np.copy(x)
            except KeyError:
                print(ERR_MOL_NOT_FOUND % (key))
            i += 1
        return batch_features, batch_target

    def _make_matrix_random(self, keys, nobjs, batch_features, batch_target):
        for i in range(nobjs):
            key = keys[random.randint(0, len(keys) - 1)]
            try:
                batch_target[i] = np.copy(self.target[key])
                batch_features[i] = np.copy(self.x_raw[key])
            except KeyError:
                print(ERR_MOL_NOT_FOUND % (key))

    def _make_matrix_balanced(self, keys, nobjs, batch_features, batch_target):
        btask = [0 for _ in range(self.ntargets)]
        key_size = len(keys) - 1
        for i in range(nobjs):
            if sum(btask) == self.ntargets:
                btask = [0 for _ in range(self.ntargets)]
                j = random.randint(0, self.ntargets - 1)
            else:
                j = btask.index(min(btask))

            while True:
                key = keys[random.randint(0, key_size)]
                if int(np.fabs(self.target[key][j] - 9999.0)) != 0:
                    break
            try:
                batch_target[i] = np.copy(self.target[key])
                batch_features[i] = np.copy(self.x_raw[key])
            except KeyError:
                print(ERR_MOL_NOT_FOUND % (key))

    def make_matrix(self, keys, nobjs, batch_features, batch_target, batch_mode=0):
        if batch_mode == 0:
            self._make_matrix_random(keys, nobjs, batch_features, batch_target)
        else:
            self._make_matrix_balanced(keys, nobjs, batch_features, batch_target)
        return 0

    def data_generator(self, keys, batch_size, batch_mode=0):
        self.batch_features = np.zeros((batch_size, self.nfeatures))
        self.batch_target = np.zeros((batch_size, self.ntargets))
        while True:
            self.make_matrix(keys, batch_size, self.batch_features, self.batch_target, batch_mode)
            yield self.batch_features, self.batch_target

    def make_prediction(self, model, keys):
        predictions = {}
        x = [self.x_raw[key] for key in keys]
        x = np.array(x)
        xp = model.predict(x, verbose=0)
        for i, key in enumerate(keys):
            predictions[key] = xp[i]
        return predictions

    def simplerun(
        self,
        batch_size_,
        batch_mode_,
        num_epochs,
        ndense_layers,
        nunits,
        random_state,
        model_output=None,
    ):
        ttfn = GetTrainTestFnc()
        if ttfn is None:
            ttfn = TrainTestSplit
        else:
            print("Using custom train/test split function")

        train_keys, test_keys = ttfn(
            list(self.target.keys()), test_size=0.20, random_state=random_state
        )

        print("Train set size: %d Test set size %d" % (len(train_keys), len(test_keys)))

        if model_output is not None and Path(model_output).is_file():
            model = GetLoadModelFnc()(model_output)
        else:
            model_ = GetKerasModel()
            if model_ is None:
                model = example_build_model(self.nfeatures, nunits, ndense_layers, self.ntargets)
            else:
                model = model_(self.nfeatures, nunits, ndense_layers)
        print(model.summary())

        train_steps_per_epoch = ceil(len(train_keys) / float(batch_size_))
        train_generator = self.data_generator(train_keys, batch_size_, batch_mode_)
        x_test, y_test = self.gen_data(test_keys)

        b = batch_size_
        log_dir_ = "./logs/#b%d_#e%d_#u%d_#dl%d_" % (b, num_epochs, nunits, ndense_layers)
        log_dir_ += time.strftime("%Y%m%d%H%M%S")

        callbacks_ = [
            TensorBoard(log_dir=log_dir_, histogram_freq=0, write_graph=False, write_images=False)
        ]

        model.fit(
            train_generator,
            steps_per_epoch=train_steps_per_epoch,
            epochs=num_epochs,
            verbose=self.verbose,
            validation_data=(x_test, y_test),
            callbacks=callbacks_,
        )

        y_recalc_train = self.make_prediction(model, train_keys)
        y_pred_test = self.make_prediction(model, test_keys)

        ytrain_recalc = [y_recalc_train[key] for key in train_keys]
        ytrain_true = [self.target[key] for key in train_keys]

        ytest_pred = [y_pred_test[key] for key in test_keys]
        ytest_true = [self.target[key] for key in test_keys]

        r2_val = RSQ(ytrain_true, ytrain_recalc)
        q2_val = RSQ(ytest_pred, ytest_true)
        mse_val = MSE(ytest_pred, ytest_true)
        print(f"R2: {r2_val:.4f} Q2: {q2_val:.4f} MSE: {mse_val:.4f}")

        with open("%s_pred.csv" % time.strftime("%Y%m%d%H%M%S"), "w") as fo:
            for i, key in enumerate(test_keys):
                fo.write("%s" % (key))
                for j in range(len(ytest_true[i])):
                    fo.write(f",{ytest_true[i][j]:f},{ytest_pred[i][j]:f}")
                fo.write("\n")

        if model_output is not None:
            save_model_compat(model, model_output)

        return {
            "r2": r2_val,
            "q2": q2_val,
            "mse": mse_val,
            "model": model,
        }

    def _train_fold_model(
        self,
        train_keys,
        val_keys,
        batch_size_,
        batch_mode_,
        num_epochs,
        ndense_layers,
        nunits,
        cvout,
        cv_,
        model_output,
    ):
        K.clear_session()
        train_steps_per_epoch = ceil(len(train_keys) / float(batch_size_))
        train_generator = self.data_generator(train_keys, batch_size_, batch_mode_)
        x_val, y_val = self.gen_data(val_keys)

        model_ = GetKerasModel()
        if model_ is None:
            model = example_build_model(self.nfeatures, nunits, ndense_layers, self.ntargets)
        else:
            model = model_(self.nfeatures, nunits, ndense_layers, self.ntargets)

        print(model.summary())
        dname = cvout.replace(".csv", "")
        log_dir_ = "./logs/cv%d_%s_#b%d_#e%d_#u%d_#dl%d_%s" % (
            cv_,
            dname,
            batch_size_,
            num_epochs,
            nunits,
            ndense_layers,
            time.strftime("%Y%m%d%H%M%S"),
        )
        callbacks_ = [
            TensorBoard(log_dir=log_dir_, histogram_freq=0, write_graph=False, write_images=False),
            ModelCheckpoint(model_output, monitor="val_loss", verbose=0, save_best_only=True),
        ]

        model.fit(
            train_generator,
            steps_per_epoch=train_steps_per_epoch,
            epochs=num_epochs,
            verbose=self.verbose,
            validation_data=(x_val, y_val),
            callbacks=callbacks_,
        )
        return model

    def _prepare_mout_path(self, mout):
        mout_path = (
            Path(mout) if mout is not None else Path(f"{time.strftime('%Y%m%d%H%M%S')}_model")
        )
        last_model = None
        if mout_path.exists():
            p = list(mout_path.glob("**/*.h5")) + list(mout_path.glob("**/*.keras"))
            mids = [int(x.stem) for x in p if x.stem.isdigit()]
            last_model = max(mids) if mids else None
        else:
            mout_path.mkdir(exist_ok=True, parents=True)
            with open(mout_path / "odesc_header.csv", "w") as f:
                for dname in self.xheader:
                    f.write(f"{dname}\n")
        return mout_path, last_model

    def _get_validation_splitter(self, n_splits, n_repeats, random_state):
        valfn = GetValidationFnc()
        if valfn is None:
            return RepeatedKFold(
                list(self.target.keys()),
                n_splits,
                n_repeats,
                random_state=random_state,
                test_size=0.2,
            )
        print("Using custom validation split function")
        return valfn(list(self.target.keys()))

    def _eval_fold_feature_importance(self, model_, test_keys, feat_imp):
        x_test, y_test = self.gen_data(test_keys)
        fimp = FeatureImportance(model_, x_test, y_test, self.xheader)
        fires = fimp.calculate(verbose=1)
        for key, metrics in fires.items():
            feat_imp[key]["mae"].extend(metrics["mae"])
            feat_imp[key]["mse"].extend(metrics["mse"])

    def runcv(
        self,
        batch_size_,
        batch_mode_,
        num_epochs,
        ndense_layers,
        nunits,
        cvout,
        n_splits=5,
        n_repeats=10,
        random_state=None,
        mout=None,
        fimpfile=None,
    ):
        print("N. instances: %d" % (len(self.target)))
        mout_path, last_model = self._prepare_mout_path(mout)

        feat_imp = (
            {feat_name: {"mae": [], "mse": []} for feat_name in self.xheader} if fimpfile else {}
        )
        predictions = {key: [] for key in self.target.keys()}
        recalc = {key: [] for key in self.target.keys()}

        valfn = self._get_validation_splitter(n_splits, n_repeats, random_state)

        cv_ = 0
        for train_keys, val_keys, test_keys in valfn:
            model_output = str(mout_path.absolute() / f"{cv_}.h5")
            print(
                "Train set size: %d Val set size %d Test set size: %d"
                % (len(train_keys), len(val_keys), len(test_keys))
            )
            if last_model is None:
                self._train_fold_model(
                    train_keys,
                    val_keys,
                    batch_size_,
                    batch_mode_,
                    num_epochs,
                    ndense_layers,
                    nunits,
                    cvout,
                    cv_,
                    model_output,
                )
            elif last_model - 1 == cv_:
                last_model = None

            model_ = GetLoadModelFnc()(model_output)
            y_recalc_train = self.make_prediction(model_, train_keys)
            y_pred_val = self.make_prediction(model_, val_keys)
            y_pred_test = self.make_prediction(model_, test_keys)

            for key in train_keys:
                recalc[key].append(y_recalc_train[key])
            for key in test_keys:
                predictions[key].append(y_pred_test[key])

            r2 = RSQ([self.target[k] for k in train_keys], [y_recalc_train[k] for k in train_keys])
            q2 = RSQ([self.target[k] for k in test_keys], [y_pred_test[k] for k in test_keys])
            tr2 = RSQ([self.target[k] for k in val_keys], [y_pred_val[k] for k in val_keys])
            print(f"Train R2: {r2:.4f} Test Q2: {q2:.4f} Val: R2: {tr2:.4f}\n")

            if fimpfile is not None:
                self._eval_fold_feature_importance(model_, test_keys, feat_imp)
            cv_ += 1

        WriteCrossValidationOutput(cvout, self.target, predictions, recalc)

        if fimpfile is not None:
            WriteFeatureImportance(feat_imp, fimpfile)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train Tabular Deep Neural Network models (regression or classification)."
    )
    parser.add_argument("--xmatrix", default=None, type=str, help="input data matrix")
    parser.add_argument("--ytarget", default=None, type=str, help="input data matrix")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size")
    parser.add_argument(
        "--batch_mode", default=0, type=int, help="Batch mode: 0=random; 1=balanced"
    )
    parser.add_argument("--random_state", default=123458976, type=int, help="Random state")
    parser.add_argument("--gsout", default=None, type=str, help="Grid Search output")
    parser.add_argument("--n_splits", default=5, type=int, help="Number of kfold splits")
    parser.add_argument("--n_repeats", default=20, type=int, help="Number of repetitions")
    parser.add_argument("--cvout", default=None, type=str, help="Cross validation output")
    parser.add_argument("--nunits", default=32, type=int, help="Number of neurons")
    parser.add_argument("--ndense_layers", default=1, type=int, help="Number of dense layers")
    parser.add_argument("--mout", default=None, type=str, help="Model out path")
    parser.add_argument("--featimp", default=None, type=str, help="Feature Importance file")
    parser.add_argument("--verbose", default=0, type=int, help="Set verbosity 0, 1, 2")
    parser.add_argument("--mtype", default=0, type=int, help="Regression: 0, classification: 1")

    args = parser.parse_args(argv)
    if args.xmatrix is None or args.ytarget is None or args.batch_size is None:
        parser.print_help()
        print("\nERROR! Please specify --xmatrix, --ytarget, and --batch_size.")
        return 1

    np.random.seed(args.random_state)
    random.seed(args.random_state)
    x_raw, _, xheader = ReadDescriptors(args.xmatrix)
    target = ReadTarget(args.ytarget)
    nn = NNTrain(x_raw, target, xheader)
    nn.verbose = args.verbose
    if args.cvout is None and args.gsout is None:
        nn.simplerun(
            args.batch_size,
            args.batch_mode,
            args.epochs,
            args.ndense_layers,
            args.nunits,
            args.random_state,
            args.mout,
        )
    elif args.cvout is not None and args.gsout is None:
        nn.runcv(
            args.batch_size,
            args.batch_mode,
            args.epochs,
            args.ndense_layers,
            args.nunits,
            args.cvout,
            args.n_splits,
            args.n_repeats,
            args.random_state,
            args.mout,
            args.featimp,
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
