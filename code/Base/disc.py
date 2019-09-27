#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details
"""
Maximum dissimilarity object selection
"""

from random import randrange
from random import seed
from datetime import datetime
import time

def _median(lst):
    """ Get mediane of list """
    slst = sorted(lst)
    if len(slst) % 2 != 0:
        return slst[len(slst)/2]
    else:
        return (slst[len(slst)/2] + slst[len(slst)/2-1])/2.0


class DISC(object):
    """Perform Dissimilarity compound selection
       through different algorithm variants:

       Max - Complete Linkage
       Min - Single Linkage
       Med - Median Method
       Sum - Group Average Method

    Parameters
    ----------
    dmx : array, shape(row,row)
        A square distance matrix.
        To build a distance matrix see scipy at:
        http://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    nobjects : int, optional, default: 0
        Number of object to select. 0 means an autostop
        criterion.

    method: string, default: Max
        Select the dissimilarity method. Available methods are:
        Max, Min, Med, Sum

    Returns
    ------
    disids: list
        Return the list of id selected from the algorithm.


    Notes
    -----
    See examples/plot_dis_example.py for an example.

    References
    ----------
    John D. Holliday and Peter Willett
    Definitions of "Dissimilarity" for Dissimilarity-Based Compound Selection
    Journal of Biomolecular Screening Vol. 1, Number 3, pag 145-151, 1996
    """

    def __init__(self, dmx, method, nobjects=0):
        try:
            self.dmx_ = dmx.tolist() #convert to list to be faster
        except AttributeError:
            self.dmx_ = dmx
        self.nobjects = nobjects
        self.disids = []
        self.method = method.lower().strip()

    def dislist(self):
        """ Return the list of dissimilar compounds """
        return self.disids

    def select(self):
        """ Run the Dissimilarity selection"""
        seed(datetime.now().microsecond)
        self.disids.append(randrange(0, len(self.dmx_)-1))

        if self.method == "min":
            while len(self.disids) < self.nobjects:
                self._appendnext_min()
        elif self.method == "med":
            while len(self.disids) < self.nobjects:
                self._appendnext_med()
        elif self.method == "sum":
            while len(self.disids) < self.nobjects:
                self._appendnext_sum()
        else:
            while len(self.disids) < self.nobjects:
                self._appendnext_max()
        return self.dislist()

    def _appendnext_med(self):
        """ Append the next object following the sum dissimilarity """
        # first column is the distance and second is the objectid
        dis = [[0, i] for i in range(len(self.dmx_))]
        for i in range(len(self.dmx_)):
            if i not in self.disids:
                dlist = []
                for j in range(len(self.disids)):
                    dlist.append(self.dmx_[i][self.disids[j]])
                dis[i][0] = _median(dlist)
            else:
                continue
        # first column is the distance and second is the objectid
        dis = sorted(dis, key=lambda item: item[0])
        # Select the object with the maximum distances
        # between all the summed distances list
        # and this is the last object in list
        self.disids.append(dis[-1][-1])

    def _appendnext_sum(self):
        """ Append the next object following the median dissimilarity """
        # first column is the distance and second is the objectid
        dis = [[0, i] for i in range(len(self.dmx_))]
        for i in range(len(self.disids)):
            for j in range(len(self.dmx_)):
                if j not in self.disids:
                    dis[j][0] += self.dmx_[self.disids[i]][j]
                else:
                    continue
        # first column is the distance and second is the objectid
        dis = sorted(dis, key=lambda item: item[0])
        # Select the object with the maximum distances
        # between all the summed distances list
        # and this is the last object in list
        self.disids.append(dis[-1][-1])

    def _appendnext_max(self):
        """ Append the next object following the maximum dissimilarity """
        dis = [[0, i] for i in range(len(self.dmx_))]
        for i in range(len(self.dmx_)):
            if i not in self.disids:
                maxdisid = 0
                maxdis = self.dmx_[i][self.disids[maxdisid]]
                for j in range(1, len(self.disids)):
                    if self.dmx_[i][self.disids[j]] > maxdis:
                        maxdis = self.dmx_[i][self.disids[j]]
                        maxdisid = i
                    else:
                        continue
                dis[i][0] = maxdis
            else:
                continue
        # first column is the distance and second is the objectid
        dis = sorted(dis, key=lambda item: item[0])
        # Select the object with the max distances
        # between all the minimum distances list
        # and this is the last object in list
        self.disids.append(dis[-1][-1])

    def _appendnext_min(self):
        """ Append the next object following the minimum dissimilarity """
        dis = [[0, i] for i in range(len(self.dmx_))]
        # t = time.time()
        for i in range(len(self.dmx_)):
            if i not in self.disids:
                mindis = self.dmx_[i][self.disids[0]]
                for j in range(1, len(self.disids)):
                    if self.dmx_[i][self.disids[j]] < mindis:
                        mindis = self.dmx_[i][self.disids[j]]
                    else:
                        continue
                dis[i][0] = mindis
            else:
                continue
        # print "Time1: %.3f" % (time.time()-t)
        # t = time.time()
        # first column is the distance and second is the objectid
        dis = sorted(dis, key=lambda item: item[0])
        # print "Time2: %.3f" % (time.time()-t)
        # Select the object with the max distances
        # between all the minimum distances list
        # and this is the last object in list
        self.disids.append(dis[-1][-1])
        # print "-"*10

    def _appendnext_min2(self):
        """ Append the next object following the minimum dissimilarity """
        # t = time.time()
        dis = []
        for i in range(len(self.disids)):
            mindis = None
            objid = None
            for j in range(len(self.dmx_[int(self.disids[i])])):
                if self.disids[i] != j and j not in self.disids:
                    if mindis == None:
                        mindis = self.dmx_[int(self.disids[i])][j]
                        objid = j
                    elif self.dmx_[int(self.disids[i])][j] > mindis:
                        mindis = self.dmx_[int(self.disids[i])][j]
                        objid = j
                    else:
                        continue
                else:
                  continue

            if mindis != None:
                dis.append([mindis, objid])
            else:
                continue

        # print "Time1: %.3f" % (time.time()-t)
        # t = time.time()
        # first column is the distance and second is the objectid
        dis = sorted(dis, key=lambda item: item[0])
        # print "Time2: %.3f" % (time.time()-t)
        # Select the object with the max distances
        # between all the minimum distances list
        # and this is the last object in list
        self.disids.append(dis[-1][-1])
        # print "-"*10
