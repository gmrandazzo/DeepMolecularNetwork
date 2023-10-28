#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

'''
GenVoxelHOMOLUMO generate a Voxel with the same volume size and grid step size
using orca and openbabel
'''

import sys
import os
from math import ceil
from openDX import *
import numpy as np

orca_dir = "/opt/orca_3_0_3_linux_x86-64/"
obabel_dir = "/usr/bin/"


def HOMOLUMOSUM(homodx,  lumodx):
    odx = homodx.split(".")[0]+".homolumo.dx"
    hv, ho, hd, = readOpenDX(homodx)
    lv, lo, ld, = readOpenDX(lumodx)
    nv = np.zeros(hv.shape)
    for i in range(hv.shape[0]):
        for j in range(hv.shape[1]):
            for k in range(hv.shape[2]):
                nv[i][j][k] = hv[i][j][k] + lv[i][j][k]
    writeOpenDX(nv, ho, hd, odx)


def DXRescale(nx, ny, nz, dx):
    odx = dx.replace(".dx", ".rescale.dx")
    v, o, d, = readOpenDX(dx)
    h = [int(nx/2.), int(ny/2.), int(nz/2.)]
    nv = np.zeros((nx, ny, nz))
    s = v.shape
    hs = [int(s[0]/2.), int(s[1]/2.), int(s[2]/2.)]
    for i in range(s[0]):
        x = (h[0]-hs[0])+i
        for j in range(s[1]):
            y = (h[1]-hs[1])+j
            for k in range(s[2]):
                z = (h[2]-hs[2])+k
                nv[x][y][z] = v[i][j][k]
    no = [o[0]-(h[0]*d[0]), o[1]-(h[1]*d[1]), o[2]-(h[2]*d[2])]
    writeOpenDX(nv, no, d, odx)
    return odx


def calc_n_steps(r, d):
    return int(ceil(r*d))


def Cube2DX(cube, suffix=""):
    dx = None
    if len(suffix) > 0:
        dx = cube.replace(".cube", ".%s.dx" % (suffix))
    else:
        dx = cube.replace(".cube", ".dx" % (suffix))
    cmd = '%s/obabel -icube "%s" -odx -O "%s"' % (obabel_dir, cube, dx)
    os.system(cmd)
    return dx


def WriteSplotFile(splotfile, nx, ny, nz, orbital):
    f = open(splotfile, "w")
    # set the alpha orbitals
    f.write("3\n")
    f.write("0\n")
    # set the output type: gaussian cube file
    f.write("5\n")
    f.write("7\n")
    # set orbital number
    f.write("2\n")
    f.write("%d\n" % (orbital))
    # set the grid steps
    f.write("4\n")
    f.write("%d %d %d\n" % (nx, ny, nz))
    # write file
    f.write("10\n")
    # exit from the software
    f.write("11\n")
    f.close()


def GenOrbitals(nx, ny, nz, gbw, onum):
    WriteSplotFile("splotfile", nx, ny, nz, onum)
    cmd = '%s/orca_plot %s -i < %s' % (orca_dir, gbw, "splotfile")
    os.system(cmd)
    os.remove("splotfile")
    return gbw.replace(".gbw", ".mo%da.cube" % (onum))


def FitNewVoxelSize():
    return

def GetCoordInfo(mol2file):
    f = open(mol2file, "r")
    getatomcc = False
    xmin = 9999
    xmax = -9999
    ymin = 9999
    ymax = -9999
    zmin = 9999
    zmax = -9999
    for line in f:
        if "@<TRIPOS>ATOM" in line:
            getatomcc = True
        elif "@<TRIPOS>BOND" in line:
            getatomcc = False
            break
        else:
            if getatomcc is True:
                v = list(str.split(line.strip(), " "))
                v = list(filter(None, v))
                if len(v) == 9:
                    # 1 au = 0.529177249 angstrom
                    x = float(v[2])/0.529177249
                    y = float(v[3])/0.529177249
                    z = float(v[4])/0.529177249
                    if x < xmin:
                        xmin = x
                    if x > xmax:
                        xmax = x
                    if y < ymin:
                        ymin = y
                    if y > ymax:
                        ymax = y
                    if z < zmin:
                        zmin = z
                    if z > zmax:
                        zmax = z
                else:
                    continue
            else:
                continue
    f.close()
    return xmin, xmax, ymin, ymax, zmin, zmax


def GetHOMOLUMONumber(orcaout):
    f = open(orcaout, "r")
    getoeneg = False
    orbitals = []
    for line in f:
        if "ORBITAL ENERGIES" in line:
            getoeneg = True
        elif "MOLECULAR ORBITALS" in line:
            getoeneg = False
            break
        else:
            if getoeneg is True:
                if "OCC" in line:
                    continue
                else:
                    v = str.split(line.strip(), " ")
                    v = filter(None, v)
                    if len(v) == 4:
                        orbitals.append(v)
                    else:
                        continue
            else:
                continue
    f.close()
    homo = 0
    lumo = 0
    prev = int(float(orbitals[0][1]))
    for i in range(1, len(orbitals)):
        if int(float(orbitals[i][1])) != prev:
            homo = i-1
            lumo = i
            break
        else:
            prev = int(float(orbitals[i][1]))
    return homo, lumo


def main():
    if len(sys.argv) == 6:
        step_size = float(sys.argv[5])
        xmin, xmax, ymin, ymax, zmin, zmax = GetCoordInfo(sys.argv[1])
        #print(xmin, xmax, ymin, ymax, zmin, zmax)
        homo, lumo = GetHOMOLUMONumber(sys.argv[2])
        # +/- 7 because orca do this.......
        nx = calc_n_steps(step_size, (xmax+7.)-(xmin-7.))
        ny = calc_n_steps(step_size, (ymax+7.)-(ymin-7.))
        nz = calc_n_steps(step_size, (zmax+7.)-(zmin-7.))
        homo_cube = GenOrbitals(nx, ny, nz, sys.argv[3], homo)
        lumo_cube = GenOrbitals(nx, ny, nz, sys.argv[3], lumo)
        homo_dx = Cube2DX(homo_cube, "homo")
        lumo_dx = Cube2DX(lumo_cube, "lumo")
        gsize = int(sys.argv[4])
        homo_dx_rescale = DXRescale(gsize, gsize, gsize, homo_dx)
        lumo_dx_rescale = DXRescale(gsize, gsize, gsize, lumo_dx)
        #HOMOLUMOSUM(homo_dx_rescale,  lumo_dx_rescale)
        # Clean files
        os.remove(homo_cube)
        os.remove(lumo_cube)
        os.remove(homo_dx)
        os.remove(lumo_dx)
        #os.remove(homo_dx_rescale)
        #os.remove(lumo_dx_rescale)
        #print(homo_dx)
        #print(lumo_dx)
    else:
        print("\nUsage: %s [mol2 file] [orca out] [gbw file] [grid size] [resolution]\n" % (sys.argv[0]))
    return 0


if __name__ == '__main__':
    main()
