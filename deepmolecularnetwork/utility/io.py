"""
I/O utilities for descriptors, targets, and cross-validation reports.
(c) 2018-2023 gmrandazzo@gmail.com
This file is part of DeepMolecularNetwork.
You can use,modify, and distribute it under
the terms of the GNU General Public Licenze, version 3.
See the file LICENSE for details
"""

import numpy as np


def read_descriptors(csv_descriptors, sep=","):
    """
    Read Molecular Descriptors and get back as dictionary.
    Returns the dictionary, number of features, and header names.
    """
    d = {}
    header = []
    with open(csv_descriptors) as fi:
        for line in fi:
            if "Molecule" in line or "Object Names" in line:
                header.extend(str.split(line.strip(), sep)[1:])
            else:
                v = str.split(line.strip(), sep)
                d[v[0]] = np.array(list(v[1:])).astype(float)

    if not header:
        for i in range(len(list(d.values())[0])):
            header.append(f"Var{i + 1}")
    return d, len(list(d.values())[0]), header


def read_target(csv_target, sep=","):
    """Read CSV Target dictionary."""
    d = {}
    with open(csv_target) as fi:
        for line in fi:
            if "Molecule" in line or "Object Names" in line:
                continue
            v = str.split(line.strip(), sep)
            if v[0] not in d:
                d[v[0]] = []
            for item in v[1:]:
                d[v[0]].append(float(item))

    for key in d:
        d[key] = np.array(d[key]).astype(float)
    return d


def _format_prediction_columns(key, col, ground_true_dict, prediction_dict):
    """Format prediction statistics for a given key and column index."""
    ytrue = float(ground_true_dict[key][col]) if key in ground_true_dict else 0.0
    if key in prediction_dict:
        ypred = [float(pred[col]) for pred in prediction_dict[key]]
        freq = len(ypred)
        ypavg = float(np.mean(ypred))
        ypstdev = float(np.std(ypred))
        return f",{ytrue:.4f},{ypavg:.4f},{ypstdev:.4f},{ytrue - ypavg:.4f},{freq:d}"
    return f",{ytrue:.4f},0.0,0.0,0.0,0"


def _format_recalculated_columns(key, col, ground_true_dict, recalculated_dict):
    """Format recalculated statistics for a given key and column index."""
    if recalculated_dict and key in recalculated_dict:
        yrecalc = [float(recalc[col]) for recalc in recalculated_dict[key]]
        freq = len(yrecalc)
        yravg = float(np.mean(yrecalc))
        yrstdev = float(np.std(yrecalc))
        ytrue = float(ground_true_dict[key][col]) if key in ground_true_dict else 0.0
        return f",{yravg:.4f},{yrstdev:.4f},{ytrue - yravg:.4f},{freq:d}"
    return ",0.0,0.0,0.0,0"


def write_cross_validation_output(
    outcsv, ground_true_dict, prediction_dict, recalculated_dict=None
):
    """Write cross-validation predictions and statistics to CSV.

    ``recalculated_dict`` is optional; when ``None`` the recalculated columns are zero-filled.
    """
    ncols = len(list(ground_true_dict.values())[0])
    with open(outcsv, "w") as f:
        f.write("Molecule")
        for i in range(ncols):
            f.write(
                ",y%d_true,y%d_pred,y%d_pred_stdev,y%d_pred_res,y%d_pred_freq"
                % (i + 1, i + 1, i + 1, i + 1, i + 1)
            )
        for j in range(ncols):
            f.write(
                ",y%d_recalc,y%d_recalc_stdev,y%d_recalc_res,y%d_recalc_freq"
                % (j + 1, j + 1, j + 1, j + 1)
            )
        f.write("\n")

        all_keys = sorted(set(ground_true_dict.keys()) | set(prediction_dict.keys()))

        for key in all_keys:
            f.write(str(key))
            for col in range(ncols):
                f.write(_format_prediction_columns(key, col, ground_true_dict, prediction_dict))
            for col in range(ncols):
                f.write(_format_recalculated_columns(key, col, ground_true_dict, recalculated_dict))
            f.write("\n")


# Backward compatible CamelCase aliases
ReadDescriptors = read_descriptors
ReadTarget = read_target
WriteCrossValidationOutput = write_cross_validation_output
