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
from sys import argv
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


def example_build_model(nfeatures, nunits, ndense_layers, ntargets):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(nfeatures,)))
    model.add(Dense(nunits, activation="relu"))
    model.add(Dropout(0.15))
    for i in range(ndense_layers):
        model.add(Dense(nunits, activation="relu"))
        model.add(Dropout(0.1))
    model.add(Dense(nunits, activation="relu"))
    model.add(Dense(ntargets))
    model.compile(loss="mse", optimizer=create_adam(learning_rate=0.00005), metrics=["mse", "mae"])
    return model


def build_gridsearch_model(nfeatures, ndense_layers, nunits, activation, dropout):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(nfeatures,)))

    for i in range(ndense_layers):
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
    def __init__(self, X_raw, target, xheader=None):
        self.X_raw = X_raw
        self.xheader = xheader
        self.target = target
        self.nfeatures = len(list(X_raw.values())[0])
        try:
            self.ntargets = len(list(target.values())[0])
        except (TypeError, IndexError):
            self.ntargets = 1
        self.ntotobj = len(list(self.X_raw.keys()))
        print("#Inst.: %d  #Feat.: %d  #Tar.: %d" % (self.ntotobj, self.nfeatures, self.ntargets))
        self.verbose = 0
        self.batch_features = None
        self.batch_target = None

    def GenData(self, keys):
        batch_features = np.zeros((len(keys), self.nfeatures))
        batch_target = np.zeros((len(keys), self.ntargets))
        i = 0
        for key in keys:
            try:
                t = self.target[key]
                x = self.X_raw[key]
                batch_target[i] = np.copy(t)
                batch_features[i] = np.copy(x)
            except KeyError:
                print("Molecule %s not found" % (key))
            i += 1
        return batch_features, batch_target

    def makeMatrix(self, keys, nobjs, batch_features, batch_target, batch_mode=0):
        if batch_mode == 0:
            for i in range(nobjs):
                key = keys[random.randint(0, len(keys) - 1)]
                try:
                    t = self.target[key]
                    x = self.X_raw[key]
                    batch_target[i] = np.copy(t)
                    batch_features[i] = np.copy(x)
                except KeyError:
                    print("Molecule %s not found" % (key))
        else:
            btask = [0 for _ in range(self.ntargets)]
            for i in range(nobjs):
                key = None
                key_size = len(keys) - 1
                if sum(btask) == self.ntargets:
                    for k in range(len(btask)):
                        btask[k] = 0
                    j = random.randint(0, self.ntargets - 1)
                else:
                    j = btask.index(min(btask))

                while True:
                    key = keys[random.randint(0, key_size)]
                    if int(np.fabs(self.target[key][j] - 9999.0)) == 0:
                        continue
                    else:
                        break
                try:
                    t = self.target[key]
                    x = self.X_raw[key]
                    batch_target[i] = np.copy(t)
                    batch_features[i] = np.copy(x)
                except KeyError:
                    print("Molecule %s not found" % (key))
        return 0

    def DataGenerator(self, keys, batch_size, batch_mode=0):
        self.batch_features = np.zeros((batch_size, self.nfeatures))
        self.batch_target = np.zeros((batch_size, self.ntargets))
        while True:
            self.makeMatrix(keys, batch_size, self.batch_features, self.batch_target, batch_mode)
            yield self.batch_features, self.batch_target

    def makePrediction(self, model, keys):
        predictions = {}
        x = []
        for key in keys:
            x.append(self.X_raw[key])
        x = np.array(x)
        xp = model.predict(x, verbose=0)
        for i in range(len(keys)):
            key = keys[i]
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

        model = None
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
        train_generator = self.DataGenerator(train_keys, batch_size_, batch_mode_)
        x_test, y_test = self.GenData(test_keys)

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

        y_recalc_train = self.makePrediction(model, train_keys)
        y_pred_test = self.makePrediction(model, test_keys)

        ytrain_recalc = []
        ytrain_true = []
        for key in train_keys:
            ytrain_recalc.append(y_recalc_train[key])
            ytrain_true.append(self.target[key])

        ytest_pred = []
        ytest_true = []
        for key in test_keys:
            ytest_pred.append(y_pred_test[key])
            ytest_true.append(self.target[key])

        r2_val = RSQ(ytrain_true, ytrain_recalc)
        q2_val = RSQ(ytest_pred, ytest_true)
        mse_val = MSE(ytest_pred, ytest_true)
        print(f"R2: {r2_val:.4f} Q2: {q2_val:.4f} MSE: {mse_val:.4f}")

        fo = open("%s_pred.csv" % time.strftime("%Y%m%d%H%M%S"), "w")
        for i in range(len(ytest_true)):
            fo.write("%s" % (test_keys[i]))
            for j in range(len(ytest_true[i])):
                fo.write(f",{ytest_true[i][j]:f},{ytest_pred[i][j]:f}")
            fo.write("\n")
        fo.close()

        if model_output is not None:
            save_model_compat(model, model_output)

        return {
            "r2": r2_val,
            "q2": q2_val,
            "mse": mse_val,
            "model": model,
        }

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

        mout_path = None
        if mout is not None:
            mout_path = Path(mout)
        else:
            mout_path = Path("%s_model" % (time.strftime("%Y%m%d%H%M%S")))

        last_model = None
        if mout_path.exists():
            p = list(Path(mout_path).glob("**/*.h5")) + list(Path(mout_path).glob("**/*.keras"))
            mids = [int(x.stem) for x in p if x.stem.isdigit()]
            if len(mids) > 0:
                last_model = max(mids)
            else:
                last_model = None
        else:
            mout_path.mkdir(exist_ok=True, parents=True)
            f = open("%s/odesc_header.csv" % (str(mout_path.absolute())), "w")
            for dname in self.xheader:
                f.write("%s\n" % (dname))
            f.close()

        feat_imp = {}
        if fimpfile is not None:
            for feat_name in self.xheader:
                feat_imp[feat_name] = {"mae": [], "mse": []}

        cv_ = 0
        predictions = {}
        recalc = {}
        for key in self.target.keys():
            predictions[key] = []
            recalc[key] = []

        valfn = GetValidationFnc()
        if valfn is None:
            valfn = RepeatedKFold(
                list(self.target.keys()),
                n_splits,
                n_repeats,
                random_state=random_state,
                test_size=0.2,
            )
        else:
            print("Using custom validation split function")
            valfn = valfn(list(self.target.keys()))

        for train_keys, val_keys, test_keys in valfn:
            K.clear_session()
            train_steps_per_epoch = ceil(len(train_keys) / float(batch_size_))
            train_generator = self.DataGenerator(train_keys, batch_size_, batch_mode_)
            x_test, y_test = self.GenData(test_keys)
            x_val, y_val = self.GenData(val_keys)
            print(
                "Train set size: %d Val set size %d Test set size: %d"
                % (len(train_keys), len(val_keys), len(test_keys))
            )
            model_output = "%s/%d.h5" % (str(mout_path.absolute()), cv_)

            if last_model is None:
                model = None
                model_ = GetKerasModel()
                if model_ is None:
                    model = example_build_model(
                        self.nfeatures, nunits, ndense_layers, self.ntargets
                    )
                else:
                    model = model_(self.nfeatures, nunits, ndense_layers, self.ntargets)

                print(model.summary())
                dname = cvout.replace(".csv", "")
                b = batch_size_
                log_dir_ = "./logs/cv%d_%s_#b%d_#e%d_#u%d_#dl%d_" % (
                    cv_,
                    dname,
                    b,
                    num_epochs,
                    nunits,
                    ndense_layers,
                )
                log_dir_ += time.strftime("%Y%m%d%H%M%S")
                callbacks_ = [
                    TensorBoard(
                        log_dir=log_dir_, histogram_freq=0, write_graph=False, write_images=False
                    ),
                    ModelCheckpoint(
                        model_output, monitor="val_loss", verbose=0, save_best_only=True
                    ),
                ]

                model.fit(
                    train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=num_epochs,
                    verbose=self.verbose,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks_,
                )
            else:
                if last_model - 1 == cv_:
                    last_model = None

            model_ = GetLoadModelFnc()(model_output)

            y_recalc_train = self.makePrediction(model_, train_keys)
            y_pred_val = self.makePrediction(model_, val_keys)
            y_pred_test = self.makePrediction(model_, test_keys)

            y_recalc = []
            y_true_recalc = []
            for key in train_keys:
                y_recalc.append(y_recalc_train[key])
                y_true_recalc.append(self.target[key])
                recalc[key].append(y_recalc_train[key])

            ypred_val = []
            ytrue_val = []
            for key in val_keys:
                ypred_val.append(y_pred_val[key])
                ytrue_val.append(self.target[key])

            ypred_test = []
            ytrue_test = []
            for key in test_keys:
                ypred_test.append(y_pred_test[key])
                ytrue_test.append(self.target[key])
                predictions[key].append(y_pred_test[key])

            r2 = RSQ(y_true_recalc, y_recalc)
            q2 = RSQ(ytrue_test, ypred_test)
            tr2 = RSQ(ytrue_val, ypred_val)
            print(f"Train R2: {r2:.4f} Test Q2: {q2:.4f} Val: R2: {tr2:.4f}\n")

            if fimpfile is not None:
                fimp = FeatureImportance(model, x_test, y_test, self.xheader)
                fires = fimp.Calculate(verbose=1)
                for key in fires.keys():
                    feat_imp[key]["mae"].extend(fires[key]["mae"])
                    feat_imp[key]["mse"].extend(fires[key]["mse"])
            cv_ += 1

        WriteCrossValidationOutput(cvout, self.target, predictions, recalc)

        if fimpfile is not None:
            WriteFeatureImportance(feat_imp, fimpfile)


def main():
    parser = argparse.ArgumentParser()
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

    args = parser.parse_args(argv[1:])
    if args.xmatrix is None or args.ytarget is None or args.batch_size is None:
        print("ERROR! Please specify input table to train!")
        print(
            f"\n Usage: {argv[0]} --xmatrix [input file] --ytarget [input file] "
            "--epochs [int] --batch_size [int]"
        )
    else:
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        X_raw, nfeatures_, xheader = ReadDescriptors(args.xmatrix)
        target = ReadTarget(args.ytarget)
        nn = NNTrain(X_raw, target, xheader)
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


if __name__ == "__main__":
    main()
