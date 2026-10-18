#!/usr/bin/env python3
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import numpy as np


def print_progress_bar(iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    if iteration == total:
        print()


printProgressBar = print_progress_bar


def mse_metric(y_true, y_pred):
    """MEAN SQUARED ERROR"""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    return float(np.mean(np.square(y_p - y_t)))


def mae_metric(y_true, y_pred):
    """MEAN ABSOLUTE ERROR"""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    return float(np.mean(np.abs(y_p - y_t)))


MSE = mse_metric
MAE = mae_metric


class FeatureImportance:
    def __init__(self, model, X, y, featlst, f_perm=None, iterations=20, random_state=12345):
        self.model = model
        self.X = X
        self.y = y
        self.featlst = featlst
        self.f_perm = f_perm
        self.iterations = iterations
        self._rng = np.random.default_rng(random_state)

    def feature_permutation(self, batch_features, fid=-1):
        if self.f_perm is None:
            batch_perm = batch_features.copy()
            self._rng.shuffle(batch_perm[:, fid])
            return batch_perm
        return self.f_perm(batch_features, fid)

    _LEGACY_ALIASES = {
        "FeaturePermutation": "feature_permutation",
        "Calculate": "calculate",
    }

    def __getattr__(self, name: str):
        if name in self._LEGACY_ALIASES:
            return getattr(self, self._LEGACY_ALIASES[name])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def calculate(self, verbose=0):
        """
        Compute the feature importance according to
        the Breiman-Fisher-Rudin-Dominici-Algorithm.
        """
        i = 0
        if verbose == 1:
            print_progress_bar(
                0,
                len(self.featlst) * self.iterations,
                prefix="Feature Importance Progress:",
                suffix="Complete",
                length=50,
            )
        feature_importance_results = {}
        ypred = self.model.predict(self.X)
        e_orig_mse = mae_metric(self.y, ypred)
        e_orig_mae = mae_metric(self.y, ypred)
        for fid_ in range(len(self.featlst)):
            feat_name = self.featlst[fid_]
            feature_importance_results[feat_name] = {"mae": [], "mse": []}
            for _ in range(self.iterations):
                x_val_perm = self.feature_permutation(self.X, fid=fid_)
                ypred_perm = self.model.predict(x_val_perm)
                e_perm_mse = mse_metric(self.y, ypred_perm)
                e_perm_mae = mae_metric(self.y, ypred_perm)
                feature_importance_results[feat_name]["mae"].append(e_perm_mae / e_orig_mae)
                feature_importance_results[feat_name]["mse"].append(e_perm_mse / e_orig_mse)
                if verbose == 1:
                    print_progress_bar(
                        i,
                        len(self.featlst) * self.iterations,
                        prefix="Progress:",
                        suffix="Complete",
                        length=50,
                    )
                    i += 1
        return feature_importance_results


def calc_feat_imp_final_results(x):
    """
    Calculate min, first quantile, median, third quantile and max for an array
    """
    arr = np.array(x)
    min_x = arr.min()
    q1 = np.percentile(arr, 25)
    med_x = np.percentile(arr, 50)
    q3 = np.percentile(arr, 75)
    max_x = arr.max()
    return [min_x, q1, med_x, q3, max_x]


CalcFeatImpFinalResults = calc_feat_imp_final_results


def write_feature_importance(feature_importance_dict, fout):
    """
    Write output for the feature importance results
    """
    with open(fout, "w") as fo:
        fo.write("Feature,Min(MSE),Q1(MSE),MED(MSE),Q3(MSE),MAX(MSE),")
        fo.write("Min(MAE),Q1(MAE),MED(MAE),Q3(MAE),MAX(MAE)\n")
        for key in feature_importance_dict.keys():
            fo.write("%s," % (key))
            r_mse = calc_feat_imp_final_results(feature_importance_dict[key]["mse"])
            r_mae = calc_feat_imp_final_results(feature_importance_dict[key]["mae"])
            fo.write(
                "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(
                    r_mse[0], r_mse[1], r_mse[2], r_mse[3], r_mse[4]
                )
            )
            fo.write(
                "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    r_mae[0], r_mae[1], r_mae[2], r_mae[3], r_mae[4]
                )
            )


WriteFeatureImportance = write_feature_importance
