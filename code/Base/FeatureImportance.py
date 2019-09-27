#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

from __future__ import print_function
import numpy as np

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
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
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    #print '\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)
    # Print New Line on Complete
    if iteration == total:
        print()

def MSE(y_true, y_pred):
    """
    MEAN SQUARED ERROR
    """
    e = 0.0
    for i in range(len(y_true)):
        e += np.square(y_pred[i]-y_true[i])
    return e/float(len(y_true))


def MAE(y_true, y_pred):
    """
    MEAN ABSOLUTE ERROR
    """
    e = 0.0
    for i in range(len(y_true)):
        e += np.abs(y_pred[i]-y_true[i])
    return e/float(len(y_true))


class FeatureImportance(object):
    def __init__(self, model, X, y, featlst, f_perm=None, iterations=20):
        self.model = model
        self.X = X
        self.y = y
        self.featlst = featlst
        self.f_perm = f_perm
        self.iterations = iterations

    def FeaturePermutation(self, batch_features, fid=-1):
        if self.f_perm is None:
            batch_perm = batch_features.copy()
            np.random.shuffle(batch_perm[:, fid])
            return batch_perm
        else:
            """
            For complex data function where permutation function should
            be rewritten
            """
            return self.f_perm(batch_features, fid)

    def Calculate(self, verbose=0):
        """
        Compute the feature importance according to
        the Breiman-Fisher-Rudin-Dominici-Algorithm
        Train a model f with a feature map X and a target vector y.
        Measure th error L(y, y_pred) = e_original

        Input:
             trained model f
             feature matrix X
             target vector y
             error measure L(y, y_pred)

        1) Estimate the original model error
        2) For each feature:
          - Generate a feature matrix with the p feature permutated
            N times to breaks the association between Xj and y
          - estimate the error using the permutated X feature matrix
          - calculate the feature importance FI = e_perm/e_original or
            FI = e_perm - e_original
        3) Sort variables by descending Fi

        The error estimation utilised is the mean squared error calculated
        with this formula
        mse = (np.square(A - B)).mean(axis=0)
        """
        i = 0
        if verbose == 1:
            printProgressBar(0,
                             len(self.featlst)*self.iterations,
                             prefix = 'Feature Importance Progress:',
                             suffix = 'Complete',
                             length = 50)
        FI = {}
        ypred = self.model.predict(self.X)
        e_orig_mse = MAE(list(self.y), list(ypred))
        e_orig_mae = MAE(list(self.y), list(ypred))
        # calculate the feature importance for the descriptors
        for fid_ in range(len(self.featlst)):
            FI[self.featlst[fid_]] = {'mae': [], 'mse': []}
            for it in range(self.iterations):
                x_val_perm = self.FeaturePermutation(self.X, fid=fid_)
                ypred_perm = self.model.predict(x_val_perm)
                e_perm_mse = MSE(list(self.y), list(ypred_perm))
                e_perm_mae = MAE(list(self.y), list(ypred_perm))
                FI[self.featlst[fid_]]['mae'].append(e_perm_mae/e_orig_mae)
                FI[self.featlst[fid_]]['mse'].append(e_perm_mse/e_orig_mse)
                if verbose == 1:
                    printProgressBar(i,
                                     len(self.featlst)*self.iterations,
                                     prefix = 'Progress:',
                                     suffix = 'Complete',
                                     length = 50)
                    i += 1
        return FI


def CalcFeatImpFinalResults(x):
    """
    Calculate min, first quantile, median, third quantile and max for an array
    """
    x = np.array(x)
    min_x = x.min()
    q1 = np.percentile(x, 25)
    med_x = np.percentile(x, 50)
    q3 = np.percentile(x, 75)
    max_x = x.max()
    return [min_x, q1, med_x, q3, max_x]


def WriteFeatureImportance(FI, fout):
    """
    Write output for the feature importance results
    """
    fo = open(fout, "w")
    fo.write("Feature,Min(MSE),Q1(MSE),MED(MSE),Q3(MSE),MAX(MSE),")
    fo.write("Min(MAE),Q1(MAE),MED(MAE),Q3(MAE),MAX(MAE)\n")
    for key in FI.keys():
        fo.write("%s," % (key))
        r_mse = CalcFeatImpFinalResults(FI[key]['mse'])
        r_mae = CalcFeatImpFinalResults(FI[key]['mae'])
        fo.write("%.4f,%.4f,%.4f,%.4f,%.4f," % (r_mse[0],
                                                r_mse[1],
                                                r_mse[2],
                                                r_mse[3],
                                                r_mse[4]))

        fo.write("%.4f,%.4f,%.4f,%.4f,%.4f\n" % (r_mae[0],
                                                 r_mae[1],
                                                 r_mae[2],
                                                 r_mae[3],
                                                 r_mae[4]))
    fo.close()
