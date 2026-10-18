"""
Voxel dataset management and 3D transformation utilities.
(c) 2018-2023 gmrandazzo@gmail.com
This file is part of DeepMolecularNetwork.
You can use,modify, and distribute it under
the terms of the GNU General Public Licenze, version 3.
See the file LICENSE for details
"""

from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import random
import sys

import numpy as np

from deepmolecularnetwork.voxels.opendx import readOpenDX, writeOpenDX


def SaveVoxelDatabase(path_, name, voxels):
    vpath_ = f"{path_}/{name}.vdb"
    if not Path(vpath_).is_dir():
        os.makedirs(vpath_, exist_ok=True)
        i = 0
    else:
        i = len(list(Path(vpath_).glob("**/*.npy")))
    db_path = f"{vpath_}/db{i}"
    np.save(db_path, voxels)


def LoadVoxelDatabase(path_):
    d = {}
    for p in Path(path_).iterdir():
        if p.is_dir() and ".vdb" in str(p):
            name = p.resolve().stem.split(".")[0]
            for db in Path(str(p)).iterdir():
                if db.is_file() and ".npy" in str(db):
                    arr = np.load(str(db), mmap_mode=None)
                    if name in d:
                        d[name].append(arr)
                    else:
                        d[name] = [arr]
    return d


class VoxelObject:
    """Object to define the 3D Object dataset."""

    def __init__(self, dxnames):
        """Initialisation voxel dataset; dxnames is a list for multichannel."""
        self.name = str(Path(dxnames[0]).resolve().stem.split(".")[0])
        if "Conf" in self.name:
            a = str.split(self.name, "_")[1:]
            self.name = "_".join(a)

        self.voxels = []
        self.origins = []
        self.deltas = []
        for dxname in dxnames:
            v, o, d = self.dxRead(dxname)
            self.voxels.append(v)
            self.origins.append(o)
            self.deltas.append(d)

    def dxRead(self, dxfile):
        """Read the DX file and convert to numpy array."""
        v, o, d = readOpenDX(dxfile)
        return v, o, d

    def toFlatten(self):
        """Flatten multichannel voxels into 1D vector."""
        return np.array(self.voxels).ravel()

    def ComputeVoxelRotation(self, n_rotations, multiple_ch=False):
        """Compute 3D rotations for voxels."""
        if n_rotations == 0:
            x = list(self.voxels)
            if multiple_ch:
                return np.array([x])
            return np.array(x)

        p = Pool(processes=cpu_count())
        x_rotated = p.map(self.VoxelRotation, [self.voxels for _ in range(n_rotations)])
        p.close()
        return np.array(x_rotated)

    def ComputeChannelGeneration(self, voxels):
        """Generate multichannel representation using multiprocessing."""
        p = Pool(processes=cpu_count())
        ch_voxels = p.map(self.ChannelsGenerator, list(voxels))
        p.close()
        return np.array(ch_voxels)

    def ChannelsGenerator(
        self,
        voxel,
        isovalues=(-0.3, -0.2, -0.15, -0.1, -0.05, 0.05, 0.10, 0.15, 0.20, 0.30),
    ):
        """Generate multichannel isovalue representation."""
        ch = [np.zeros_like(voxel) for _ in range(len(isovalues) + 1)]
        ch[0] = np.where(voxel < isovalues[0], voxel, 0.0)
        for c in range(1, len(isovalues)):
            mask = (voxel >= isovalues[c - 1]) & (voxel < isovalues[c])
            ch[c] = np.where(mask, voxel, 0.0)
        ch[-1] = np.where(voxel > isovalues[-1], voxel, 0.0)
        return np.array(ch)

    @staticmethod
    def mk3dtr(x, y, z):
        """Pure 3D translation matrix."""
        ret = np.eye(4)
        ret[0:3, 3] = x, y, z
        return ret

    @staticmethod
    def mk3drotx(theta):
        """Pure rotation around x axis."""
        ret = np.eye(4)
        ret[1, [1, 2]] = [np.cos(theta), -np.sin(theta)]
        ret[2, [1, 2]] = [np.sin(theta), np.cos(theta)]
        return ret

    @staticmethod
    def mk3droty(theta):
        """Pure rotation around y axis."""
        ret = np.eye(4)
        ret[0, [0, 2]] = [np.cos(theta), -np.sin(theta)]
        ret[2, [0, 2]] = [np.sin(theta), np.cos(theta)]
        return ret

    @staticmethod
    def mk3drotz(theta):
        """Pure rotation around z axis."""
        ret = np.eye(4)
        ret[0, [0, 1]] = [np.cos(theta), -np.sin(theta)]
        ret[1, [0, 1]] = [np.sin(theta), np.cos(theta)]
        return ret

    def VoxelRotation(self, voxels):
        """Rotate a multichannel voxel representation."""
        gs = [np.array(v) for v in voxels]
        ngs = [np.zeros_like(v) for v in voxels]
        h_x = gs[0].shape[0] / 2.0
        h_y = gs[0].shape[1] / 2.0
        h_z = gs[0].shape[2] / 2.0
        yaw = random.uniform(-np.pi, np.pi)
        pitch = random.uniform(-np.pi, np.pi)
        roll = random.uniform(-np.pi, np.pi)

        f_t = self.mk3dtr(-h_x, -h_y, -h_z)
        f_r = self.mk3drotz(yaw) @ self.mk3droty(pitch) @ self.mk3drotx(roll)
        f_b = self.mk3dtr(h_x, h_y, h_z)

        dim0, dim1, dim2 = gs[0].shape
        for i in range(dim0):
            for j in range(dim1):
                for k in range(dim2):
                    p = [i, j, k, 1]
                    np_ = f_t @ p
                    np_r = f_r @ np_
                    np_r_t = f_b @ np_r
                    n_i, n_j, n_k, _ = np.floor(np_r_t).astype("int")
                    if 0 <= n_i < dim0 and 0 <= n_j < dim1 and 0 <= n_k < dim2:
                        for ch in range(len(gs)):
                            ngs[ch][n_i, n_j, n_k] = gs[ch][i, j, k]
        return ngs


# Snake_case alias
save_voxel_database = SaveVoxelDatabase
load_voxel_database = LoadVoxelDatabase


if __name__ == "__main__":
    if len(sys.argv) == 3:
        vobj = VoxelObject([sys.argv[1]])
        new_voxels = vobj.VoxelRotation(vobj.voxels)
        writeOpenDX(new_voxels[0], vobj.origins[0], vobj.deltas[0], sys.argv[2])
    else:
        print(f"\n Usage: {sys.argv[0]} [input dx] [output dx]\n")
