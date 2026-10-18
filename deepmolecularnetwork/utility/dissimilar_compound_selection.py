#!/usr/bin/env python3
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details
"""
Maximum dissimilarity object selection
"""

from random import randrange, seed
from datetime import datetime


def _median(lst):
    """Get median of list"""
    slst = sorted(lst)
    mid = len(slst) // 2
    if len(slst) % 2 != 0:
        return slst[mid]
    return (slst[mid] + slst[mid - 1]) / 2.0


class DISC:
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
            self.dmx_ = dmx.tolist()  # convert to list to be faster
        except AttributeError:
            self.dmx_ = dmx
        self.nobjects = nobjects
        self.disids = []
        self.method = method.lower().strip()

    def dislist(self):
        """Return the list of dissimilar compounds"""
        return self.disids

    def select(self):
        """Run the Dissimilarity selection"""
        seed(datetime.now().microsecond)
        self.disids.append(randrange(0, len(self.dmx_) - 1))

        dispatch = {
            "min": self._appendnext_min,
            "med": self._appendnext_med,
            "sum": self._appendnext_sum,
        }
        step_fn = dispatch.get(self.method, self._appendnext_max)

        while len(self.disids) < self.nobjects:
            step_fn()
        return self.dislist()

    def _appendnext_med(self):
        """Append the next object following the sum dissimilarity"""
        dis = [[0, i] for i in range(len(self.dmx_))]
        for i in range(len(self.dmx_)):
            if i not in self.disids:
                dlist = [self.dmx_[i][d_id] for d_id in self.disids]
                dis[i][0] = _median(dlist)
        dis = sorted(dis, key=lambda item: item[0])
        self.disids.append(dis[-1][-1])

    def _appendnext_sum(self):
        """Append the next object following the median dissimilarity"""
        dis = [[0, i] for i in range(len(self.dmx_))]
        for d_id in self.disids:
            for j in range(len(self.dmx_)):
                if j not in self.disids:
                    dis[j][0] += self.dmx_[d_id][j]
        dis = sorted(dis, key=lambda item: item[0])
        self.disids.append(dis[-1][-1])

    def _appendnext_max(self):
        """Append the next object following the maximum dissimilarity"""
        dis = [[0, i] for i in range(len(self.dmx_))]
        for i in range(len(self.dmx_)):
            if i not in self.disids:
                maxdis = self.dmx_[i][self.disids[0]]
                for j in range(1, len(self.disids)):
                    if self.dmx_[i][self.disids[j]] > maxdis:
                        maxdis = self.dmx_[i][self.disids[j]]
                dis[i][0] = maxdis
        dis = sorted(dis, key=lambda item: item[0])
        self.disids.append(dis[-1][-1])

    def _appendnext_min(self):
        """Append the next object following the minimum dissimilarity"""
        dis = [[0, i] for i in range(len(self.dmx_))]
        for i in range(len(self.dmx_)):
            if i not in self.disids:
                mindis = self.dmx_[i][self.disids[0]]
                for j in range(1, len(self.disids)):
                    if self.dmx_[i][self.disids[j]] < mindis:
                        mindis = self.dmx_[i][self.disids[j]]
                dis[i][0] = mindis
        dis = sorted(dis, key=lambda item: item[0])
        self.disids.append(dis[-1][-1])

    def _appendnext_min2(self):
        """Append the next object following the minimum dissimilarity"""
        dis = []
        for d_id in self.disids:
            row_idx = int(d_id)
            mindis = None
            objid = None
            for j in range(len(self.dmx_[row_idx])):
                if row_idx != j and j not in self.disids:
                    dist = self.dmx_[row_idx][j]
                    if mindis is None or dist > mindis:
                        mindis = dist
                        objid = j
            if mindis is not None:
                dis.append([mindis, objid])
        dis = sorted(dis, key=lambda item: item[0])
        self.disids.append(dis[-1][-1])
