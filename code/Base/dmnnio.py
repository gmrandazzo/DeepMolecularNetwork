#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import numpy as np


def ReadDescriptors(csv_descriptors, sep=","):
    """
    Read Molecular Descriptors and get back as dictionary.
    the function return also the number of features and the header names
    """
    d = {}
    header = []
    fi = open(csv_descriptors, "r")
    for line in fi:
        if "Molecule" in line or "Object Names" in line:
            header.extend(str.split(line.strip(), sep)[1:])
        else:
            v = str.split(line.strip(), sep)
            d[v[0]] = np.array(list(v[1:])).astype(float)
    fi.close()
    if len(header) == 0:
        for i in range(len(list(d.values())[0])):
            header.append("Var%d" % (i+1))
    return d, len(list(d.values())[0]), header


def ReadTarget(csv_target, sep=","):
    """
    Read CSV Target using different sepration types
    """
    fi = open(csv_target, "r")
    d = {}
    for line in fi:
        v = str.split(line.strip(), sep)
        if "Molecule" in line or "Object Names" in line:
            continue
        else:
            if v[0] not in d.keys():
                d[v[0]] = []
            for item in v[1:]:
                d[v[0]].append(float(item))
    fi.close()
    for key in d.keys():
        d[key] = np.array(d[key]).astype(float)
    return d


def WriteCrossValidationOutput(outcsv, 
                               ground_true_dict,
                               prediction_dict,
                               recalculated_dict):
    ncols = len(list(ground_true_dict.values())[0])
    f = open(outcsv, "w")
    f.write("Molecule")
    for i in range(ncols):
        f.write(",y%d_true,y%d_pred,y%d_pred_stdev,y%d_pred_res,y%d_pred_freq" 
% (i+1, i+1, i+1, i+1, i+1))
    for j in range(ncols):
        f.write(",y%d_recalc,y%d_recalc_stdev,y%d_recalc_res,y%d_recalc_freq" 
% (i+1, i+1, i+1, i+1))
    f.write("\n")
    
    keys = list(ground_true_dict.keys())
    keys.extend(list(prediction_dict.keys()))
    keys = list(set(keys))
    
    for key in keys:
        f.write("%s" % (key))
        if key in prediction_dict.keys():
            for j in range(ncols):
                freq = 0
                ypavg = 0.
                ypstdev = 0.
                ytrue = 0.
                ypred = []
                for k in range(j, len(prediction_dict[key]), ncols):
                    ypred.append(float(prediction_dict[key][k]))
                
                freq = len(ypred)
                ypavg = np.mean(ypred)
                ypstdev = np.std(ypred)
            
                if key in ground_true_dict.keys():
                    ytrue = float(ground_true_dict[key][j])
                
                f.write(",%.4f,%.4f,%.4f,%.4f,%d" % (ytrue,
                                                    ypavg,
                                                    ypstdev,
                                                    ytrue-ypavg,
                                                    freq))
        else:
            for j in range(ncols):
                ytrue = 0.
                if key in ground_true_dict.keys():
                    ytrue = ground_true_dict[key][j]
                f.write(",%.4f,0.0,0.0,0.0,0" % (ytrue))
        
        if key in recalculated_dict.keys():
            for j in range(ncols):
                freq = 0
                yravg = 0.
                yrstdev = 0.
                yrecalc = []
                for k in range(j, len(recalculated_dict[key]), ncols):
                    yrecalc.append(float(recalculated_dict[key][k]))
                freq = len(yrecalc)
                yravg = np.mean(yrecalc)
                yrstdev = np.std(yrecalc)
                ytrue = 0.
                if key in ground_true_dict.keys():
                    ytrue = float(ground_true_dict[key][j])
                f.write(",%.4f,%.4f,%.4f,%d" % (yravg,
                                                yrstdev,
                                                ytrue-yravg,
                                                freq))
        else:
            for j in range(ncols):
                f.write(",0.0,0.0,0.0,0")
        f.write("\n")
    f.close()
