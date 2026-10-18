"""
I/O utilities for descriptors, targets, and cross-validation reports.
(c) 2018-2023 gmrandazzo@gmail.com
This file is part of DeepMolecularNetwork.
You can use,modify, and distribute it under
the terms of the GNU General Public Licenze, version 3.
See the file LICENSE for details
"""

import numpy as np


def ReadDescriptors(csv_descriptors, sep=","):
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


def ReadTarget(csv_target, sep=","):
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


def WriteCrossValidationOutput(outcsv, ground_true_dict, prediction_dict, recalculated_dict):
    """Write cross-validation predictions and statistics to CSV."""
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

        keys = list(ground_true_dict.keys())
        keys.extend(list(prediction_dict.keys()))
        keys = list(set(keys))

        for key in keys:
            f.write(str(key))
            if key in prediction_dict:
                for j in range(ncols):
                    ypred = [float(pred[j]) for pred in prediction_dict[key]]
                    freq = len(ypred)
                    ypavg = float(np.mean(ypred))
                    ypstdev = float(np.std(ypred))
                    ytrue = float(ground_true_dict[key][j]) if key in ground_true_dict else 0.0
                    f.write(f",{ytrue:.4f},{ypavg:.4f},{ypstdev:.4f},{ytrue - ypavg:.4f},{freq:d}")
            else:
                for j in range(ncols):
                    ytrue = float(ground_true_dict[key][j]) if key in ground_true_dict else 0.0
                    f.write(f",{ytrue:.4f},0.0,0.0,0.0,0")

            if key in recalculated_dict:
                for j in range(ncols):
                    yrecalc = [float(recalc[j]) for recalc in recalculated_dict[key]]
                    freq = len(yrecalc)
                    yravg = float(np.mean(yrecalc))
                    yrstdev = float(np.std(yrecalc))
                    ytrue = float(ground_true_dict[key][j]) if key in ground_true_dict else 0.0
                    f.write(f",{yravg:.4f},{yrstdev:.4f},{ytrue - yravg:.4f},{freq:d}")
            else:
                for j in range(ncols):
                    f.write(",0.0,0.0,0.0,0")
            f.write("\n")


# Snake_case aliases
read_descriptors = ReadDescriptors
read_target = ReadTarget
write_cross_validation_output = WriteCrossValidationOutput
