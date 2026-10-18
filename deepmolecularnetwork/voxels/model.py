"""
opendx file read/write
(c) 2018-2023 gmrandazzo@gmail.com
This file is part of DeepMolecularNetwork.
You can use,modify, and distribute it under
the terms of the GNU General Public Licenze, version 3.
See the file LICENSE for details
"""

from datetime import datetime
import os
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import r2_score

from deepmolecularnetwork.framework import (
    K,
    ModelCheckpoint,
    TensorBoard,
    is_keras_3,
    plot_model,
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
)
from deepmolecularnetwork.utility.modelvalidation import (
    CVGroupRead,
    RepeatedKFold,
    StaticGroupCV,
    TrainTestSplit,
)
from deepmolecularnetwork.utility.numpy_loss_functions import MAE, RSQ
from deepmolecularnetwork.voxels.model_builder import build_2DData_model, build_model
from deepmolecularnetwork.voxels.voxels import LoadVoxelDatabase


class Voxels3DModel:
    def __init__(
        self,
        csv_target: str,
        db: str,
        csv_descriptors: str | None = None,
        random_state: int | None = 42,
    ):
        """
        init function
        """
        self.voxels = LoadVoxelDatabase(db)
        print("Max Conformations %d " % (len(list(self.voxels.values())[0])))
        if csv_descriptors is not None:
            self.other_descriptors, self.nfeatures, self.header = ReadDescriptors(csv_descriptors)
        else:
            self.other_descriptors = None
            self.nfeatures = 0
            self.header = None
        self.target = ReadTarget(csv_target)
        self._rng = np.random.default_rng(random_state)
        self.input_shape, self.conv3d_chtype = self.get_input_voxel_shape()
        # voxel conformation selection statistics
        self.statvoxconf = {}
        for key in self.voxels.keys():
            max_conf = len(self.voxels[key])
            self.statvoxconf[key] = [[0] * len(self.voxels[key][i]) for i in range(max_conf)]

    _LEGACY_ALIASES = {
        "getInputVoxelShape": "get_input_voxel_shape",
        "GetAvailableKeys": "get_available_keys",
        "makeDataset": "make_dataset",
        "VoxelTrainGenerator": "voxel_train_generator",
        "VoxelTestSetGenerator": "voxel_test_set_generator",
        "FeaturePermutation": "feature_permutation",
    }

    def __getattr__(self, name: str):
        if name in self._LEGACY_ALIASES:
            return getattr(self, self._LEGACY_ALIASES[name])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_input_voxel_shape(self):
        input_shape = []
        for voxel in self.voxels.values():
            voxel = np.array(voxel)
            if len(input_shape) == 0:
                input_shape = voxel[0][0].shape
            else:
                for i in range(len(input_shape)):
                    if voxel[0][0].shape[i] != input_shape[i]:
                        print("Input shape Error!")
                        assert voxel[0][0].shape[i] == input_shape[i]
        input_shape = list(input_shape)
        if len(input_shape) == 3:
            input_shape.append(1)
            conv3d_chtype = "channels_last"
        else:
            conv3d_chtype = "channels_first"
        print(f"Input shape: {input_shape}")
        print("Channel type: %s" % (conv3d_chtype))
        return tuple(input_shape), conv3d_chtype

    def get_available_keys(self):
        """Return the intersection of keys present in all active modalities."""
        if self.other_descriptors is not None:
            common = (
                set(self.voxels.keys())
                & set(self.other_descriptors.keys())
                & set(self.target.keys())
            )
        else:
            common = set(self.voxels.keys()) & set(self.target.keys())
        return sorted(common)

    def _sample_conformation_rotation(self, key, selected_confs):
        max_conf = len(self.voxels[key])
        max_rot = len(self.voxels[key][0])
        while True:
            rconf = random.randint(0, max_conf - 1)
            rrot = random.randint(0, max_rot - 1)
            conf_tag = f"{key}-{rconf}-{rrot}"
            if conf_tag not in selected_confs:
                selected_confs.append(conf_tag)
                self.statvoxconf[key][rconf][rrot] += 1
                return rconf, rrot

    def _init_empty_features(self):
        if "last" in self.conv3d_chtype:
            return np.array([]).reshape(0, *self.input_shape[:3])
        return np.array([]).reshape(0, *self.input_shape[:4])

    def make_dataset(self, keys, nrotations):
        """
        Create a dataset given:
        keys: key names
        nrotations: number of rotations for data augmentation
        return X and y
        """
        batch_features = self._init_empty_features()
        batch_desc_features = (
            np.array([]).reshape(0, self.nfeatures) if self.other_descriptors is not None else None
        )

        batch_target = []
        selected_confs = []
        for _ in range(nrotations):
            for key in keys:
                rconf, rrot = self._sample_conformation_rotation(key, selected_confs)
                voxel = self.voxels[key][rconf][rrot]
                batch_features = np.vstack([batch_features, [voxel]])
                batch_target.append(self.target[key])
                if self.other_descriptors is not None:
                    batch_desc_features = np.vstack(
                        [batch_desc_features, self.other_descriptors[key]]
                    )

        if "last" in self.conv3d_chtype:
            batch_features = batch_features.reshape(*batch_features.shape[:4], 1)

        target_arr = np.array(batch_target)
        if batch_desc_features is not None:
            return [batch_features, batch_desc_features], target_arr
        return batch_features, target_arr

    def voxel_train_generator(self, keys, n_rot_repetitions=1):
        """Keras voxel data augmentation."""
        while True:
            random.seed(datetime.now().microsecond)
            X, y = self.make_dataset(keys, n_rot_repetitions)
            yield X, y

    def voxel_test_set_generator(self, keys, random_voxel_rotations):
        return self.make_dataset(keys, random_voxel_rotations)

    def feature_permutation(self, batch_features, fid=-1):
        batch_perm = batch_features.copy()
        if fid < self.nfeatures:
            self._rng.shuffle(batch_perm[1][:, fid])
        else:
            self._rng.shuffle(batch_perm[0][:])
        return batch_perm


Voxel3DModel = Voxels3DModel


def _prepare_simplerun_split(available_keys, random_state, fcvgroup=None, tid=None):
    if fcvgroup is not None:
        cvgroups = CVGroupRead(fcvgroup)
        tkey = int(tid) if tid is not None else random.choice(list(cvgroups.keys()))
        test_keys = cvgroups[tkey]
        train_keys = [k for key, grp in cvgroups.items() if key != tkey for k in grp]
        return train_keys, test_keys

    ttfn = GetTrainTestFnc()
    if ttfn is None:
        ttfn = TrainTestSplit
    else:
        print("Using custom train/test split function")
    return ttfn(available_keys, test_size=0.20, random_state=random_state)


def _build_voxel_model(vm, ndense_layers, nunits, nfilters):
    model_ = GetKerasModel()
    if vm.other_descriptors is None:
        if model_ is None:
            return build_model(vm.conv3d_chtype, vm.input_shape, ndense_layers, nunits, nfilters)
        return model_(vm.conv3d_chtype, vm.input_shape, ndense_layers, nunits, nfilters)

    if model_ is None:
        return build_2DData_model(
            vm.conv3d_chtype, vm.input_shape, vm.nfeatures, ndense_layers, nunits, nfilters
        )
    return model_(vm.conv3d_chtype, vm.input_shape, vm.nfeatures, ndense_layers, nunits, nfilters)


def _save_statconf(statvoxconf, filename="statconf.csv"):
    with open(filename, "w") as fo:
        for key, conf_mat in statvoxconf.items():
            vals = [str(val) for conf in conf_mat for val in conf]
            fo.write(f"{key},{','.join(vals)}\n")


def _setup_cv_method(available_keys, fcvgroup, n_splits_, n_repeats_, random_state):
    if fcvgroup is not None:
        cvgroups = CVGroupRead(fcvgroup)
        return StaticGroupCV(cvgroups)
    return RepeatedKFold(available_keys, n_splits_, n_repeats_, random_state, test_size=0.2)


def _prepare_cv_mout_path(mout, vm):
    if mout is None:
        return None
    mout_path = Path(f"{time.strftime('%Y%m%d%H%M%S')}_{mout}")
    mout_path.mkdir(exist_ok=True, parents=True)
    if vm.other_descriptors is not None:
        with open(mout_path / "odesc_header.csv", "w") as f:
            for item in vm.header:
                f.write(f"{item}\n")
    return mout_path


def _store_fold_predictions(target_dict, test_keys, ypred, n_rot_test):
    k = 0
    c = 0
    for i in range(len(ypred)):
        target_dict[test_keys[k]].append(list(ypred[i]))
        if c == n_rot_test - 1:
            k += 1
            c = 0
        else:
            c += 1


def _calculate_feature_importance(vm, model, x_val_, y_test_, ypred_test, feat_imp, iterations=20):
    e_orig = MAE(list(y_test_), list(ypred_test))
    for fid_ in range(vm.nfeatures):
        for _ in range(iterations):
            x_val_perm = vm.feature_permutation(x_val_, fid=fid_)
            ypred_perm = model.predict(x_val_perm)
            e_perm = MAE(list(y_test_), list(ypred_perm))
            feat_imp[fid_].append(e_perm / e_orig)

    for _ in range(iterations):
        x_val_perm = vm.feature_permutation(x_val_, fid=9999)
        ypred_perm = model.predict(x_val_perm)
        e_perm = MAE(list(y_test_), list(ypred_perm))
        feat_imp[-1].append(e_perm / e_orig)


def _save_feature_importance(vm, feat_imp, featimp_out):
    with open(str(featimp_out), "w") as fo:
        for i in range(vm.nfeatures):
            a = np.array(feat_imp[i])
            fo.write(
                "{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    vm.header[i],
                    a.min(),
                    np.percentile(a, 25),
                    np.percentile(a, 50),
                    np.percentile(a, 75),
                    a.max(),
                )
            )
        a = np.array(feat_imp[-1])
        fo.write(
            "{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                "qm_voxel_charge",
                a.min(),
                np.percentile(a, 25),
                np.percentile(a, 50),
                np.percentile(a, 75),
                a.max(),
            )
        )


def simplerun(
    db,
    csv_target,
    csv_descriptors,
    num_epochs,
    n_rot_train,
    train_steps_per_epoch_,
    n_rotation_test,
    test_steps_per_epoch_,
    ndense_layers,
    nunits,
    nfilters,
    random_state,
    **kwargs,
):
    outmodel = kwargs.get("outmodel", None)
    fcvgroup = kwargs.get("fcvgroup", None)
    tid = kwargs.get("tid", None)

    vm = Voxel3DModel(csv_target, db, csv_descriptors)
    available_keys = vm.get_available_keys()
    train_keys, test_keys = _prepare_simplerun_split(available_keys, random_state, fcvgroup, tid)

    print("Trainin set size: %d Validation set size %d" % (len(train_keys), len(test_keys)))
    train_generator = vm.voxel_train_generator(train_keys, n_rot_train)
    test_generator = vm.voxel_train_generator(test_keys, n_rot_train)

    model = _build_voxel_model(vm, ndense_layers, nunits, nfilters)
    try:
        if plot_model is not None:
            plot_model(model, to_file="model.png", show_shapes=True)
    except Exception:
        pass

    dname = os.path.basename(csv_target).replace(".csv", "") + os.path.basename(db)
    tstamp = time.strftime("%Y%m%d%H%M%S")
    log_dir_ = (
        f"./logs/{dname}_{num_epochs}_#rot{train_steps_per_epoch_}_#f{nfilters}"
        f"_#dl{ndense_layers}_#u{nunits}_{tstamp}"
    )
    callbacks_ = [
        TensorBoard(log_dir=log_dir_, histogram_freq=0, write_graph=False, write_images=False)
    ]

    fit_kwargs = {
        "epochs": num_epochs,
        "steps_per_epoch": train_steps_per_epoch_,
        "verbose": 1,
        "validation_data": test_generator,
        "validation_steps": test_steps_per_epoch_,
        "callbacks": callbacks_,
    }
    if not is_keras_3():
        fit_kwargs["use_multiprocessing"] = False

    model.fit(
        train_generator,
        **fit_kwargs,
    )

    x_test_, y_test_ = vm.voxel_test_set_generator(test_keys, n_rotation_test)
    y_pred_ = model.predict(x_test_)
    print("Test R2: %.4f" % (r2_score(y_test_, y_pred_)))

    _save_statconf(vm.statvoxconf)

    if outmodel is not None:
        model.save(outmodel)


def _train_or_load_cv_fold_model(vm, cv_, train_keys, val_keys, mout_path, csv_target, **kwargs):
    num_epochs = kwargs["num_epochs"]
    train_steps_per_epoch_ = kwargs["train_steps_per_epoch_"]
    n_rot_train = kwargs["n_rot_train"]
    n_rot_test = kwargs["n_rot_test"]
    test_steps_per_epoch_ = kwargs["test_steps_per_epoch_"]
    ndense_layers = kwargs["ndense_layers"]
    nunits = kwargs["nunits"]
    nfilters = kwargs["nfilters"]

    K.clear_session()
    model = _build_voxel_model(vm, ndense_layers, nunits, nfilters)
    print(model.summary())

    dname = os.path.basename(csv_target).replace(".csv", "")
    tstamp = time.strftime("%Y%m%d%H%M%S")
    log_dir_ = (
        f"./logs/cv{cv_}_{dname}_{num_epochs}_#rot{train_steps_per_epoch_}_#f{nfilters}"
        f"_#dl{ndense_layers}_#u{nunits}_{tstamp}"
    )
    model_outfile = f"{mout_path.absolute()}/{cv_}.h5" if mout_path else f"./{cv_}.h5"

    if Path(model_outfile).exists():
        print("The model %s exist.\nLoading the model and continue for prediction")
    else:
        callbacks_ = [
            TensorBoard(log_dir=log_dir_, histogram_freq=0, write_graph=False, write_images=False),
            ModelCheckpoint(model_outfile, monitor="val_loss", verbose=0, save_best_only=True),
        ]
        train_generator = vm.voxel_train_generator(train_keys, n_rot_train)
        val_generator = vm.voxel_train_generator(val_keys, n_rot_test)
        fit_kwargs = {
            "epochs": num_epochs,
            "steps_per_epoch": train_steps_per_epoch_,
            "verbose": 1,
            "validation_data": val_generator,
            "validation_steps": test_steps_per_epoch_,
            "callbacks": callbacks_,
        }
        if not is_keras_3():
            fit_kwargs["use_multiprocessing"] = True

        model.fit(
            train_generator,
            **fit_kwargs,
        )

    return GetLoadModelFnc()(model_outfile)


def cv(
    db,
    csv_target,
    csv_descriptors,
    n_splits_,
    n_repeats_,
    num_epochs,
    n_rot_train,
    train_steps_per_epoch_,
    n_rot_test,
    test_steps_per_epoch_,
    ndense_layers,
    nunits,
    **kwargs,
):
    nfilters = kwargs.get("nfilters", 32)
    random_state = kwargs.get("random_state", None)
    cvout = kwargs.get("cvout", None)
    fcvgroup = kwargs.get("fcvgroup", None)
    featimp_out = kwargs.get("featimp_out", None)
    mout = kwargs.get("mout", None)

    vm = Voxel3DModel(csv_target, db, csv_descriptors)
    available_keys = vm.get_available_keys()
    print("N. instances: %d" % (len(vm.target)))
    predictions = {key: [] for key in vm.target.keys()}
    valpredictions = {key: [] for key in vm.target.keys()}

    feat_imp = None
    if featimp_out is not None:
        if vm.other_descriptors is not None:
            feat_imp = [[] for _ in range(vm.nfeatures + 1)]
        else:
            print("Feature Importance calculation: DISABLED")

    mout_path = _prepare_cv_mout_path(mout, vm)
    cvmethod = _setup_cv_method(available_keys, fcvgroup, n_splits_, n_repeats_, random_state)

    cv_ = 0
    for train_keys, val_keys, test_keys in cvmethod:
        print(
            "Train set size: %d Val set size %d Test set size: %d"
            % (len(train_keys), len(val_keys), len(test_keys))
        )
        model = _train_or_load_cv_fold_model(
            vm,
            cv_,
            train_keys,
            val_keys,
            mout_path,
            csv_target,
            num_epochs=num_epochs,
            train_steps_per_epoch_=train_steps_per_epoch_,
            n_rot_train=n_rot_train,
            n_rot_test=n_rot_test,
            test_steps_per_epoch_=test_steps_per_epoch_,
            ndense_layers=ndense_layers,
            nunits=nunits,
            nfilters=nfilters,
        )
        x_train_, y_train_ = vm.voxel_test_set_generator(train_keys, n_rot_train)
        x_test_, y_test_ = vm.voxel_test_set_generator(test_keys, n_rot_test)
        x_val_, y_val_ = vm.voxel_test_set_generator(val_keys, n_rot_test)

        y_train_recalc = model.predict(x_train_)
        ypred_test = model.predict(x_test_)
        ypred_val = model.predict(x_val_)

        r2 = RSQ(y_train_, y_train_recalc)
        q2 = RSQ(y_test_, ypred_test)
        vr2 = RSQ(y_val_, ypred_val)
        print(f"Train R2: {r2:.4f} Test Q2: {q2:.4f} Val: R2: {vr2:.4f}\n")

        _store_fold_predictions(valpredictions, test_keys, ypred_val, n_rot_test)
        _store_fold_predictions(predictions, test_keys, ypred_test, n_rot_test)

        if feat_imp is not None:
            _calculate_feature_importance(vm, model, x_val_, y_test_, ypred_test, feat_imp)

        if mout_path is not None:
            model.save(f"{mout_path.absolute()}/{cv_}.h5")
        cv_ += 1

    if cvout is not None:
        WriteCrossValidationOutput(cvout, vm.target, predictions, None)

    if feat_imp is not None:
        _save_feature_importance(vm, feat_imp, featimp_out)

    return {key: np.mean(preds) for key, preds in predictions.items() if len(preds) > 0}
