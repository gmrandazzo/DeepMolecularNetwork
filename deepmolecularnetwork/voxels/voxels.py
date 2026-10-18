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


def save_voxel_database(path_, name, voxels):
    vpath_ = f"{path_}/{name}.vdb"
    if not Path(vpath_).is_dir():
        os.makedirs(vpath_, exist_ok=True)
        i = 0
    else:
        i = len(list(Path(vpath_).glob("**/*.npy")))
    db_path = f"{vpath_}/db{i}"
    np.save(db_path, voxels)


def _load_single_vdb(vdb_dir, result_dict):
    name = vdb_dir.resolve().stem.split(".")[0]
    npy_files = [f for f in vdb_dir.iterdir() if f.is_file() and ".npy" in f.name]
    for db in npy_files:
        arr = np.load(str(db), mmap_mode=None)
        if name in result_dict:
            result_dict[name].append(arr)
        else:
            result_dict[name] = [arr]


def load_voxel_database(path_):
    d = {}
    base = Path(path_)
    if not base.exists():
        return d
    for p in base.iterdir():
        if p.is_dir() and ".vdb" in str(p):
            _load_single_vdb(p, d)
    return d


# Backward compatible aliases
SaveVoxelDatabase = save_voxel_database
LoadVoxelDatabase = load_voxel_database


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
            v, o, d = self.dx_read(dxname)
            self.voxels.append(v)
            self.origins.append(o)
            self.deltas.append(d)

    _LEGACY_ALIASES = {
        "dxRead": "dx_read",
        "toFlatten": "to_flatten",
        "ComputeVoxelRotation": "compute_voxel_rotation",
        "ComputeChannelGeneration": "compute_channel_generation",
        "ChannelsGenerator": "channels_generator",
        "VoxelRotation": "voxel_rotation",
    }

    def __getattr__(self, name: str):
        if name in self._LEGACY_ALIASES:
            return getattr(self, self._LEGACY_ALIASES[name])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def dx_read(self, dxfile):
        """Read the DX file and convert to numpy array."""
        v, o, d = readOpenDX(dxfile)
        return v, o, d

    def to_flatten(self):
        """Flatten multichannel voxels into 1D vector."""
        return np.array(self.voxels).ravel()

    def compute_voxel_rotation(self, n_rotations, multiple_ch=False):
        """Compute 3D rotations for voxels."""
        if n_rotations == 0:
            x = list(self.voxels)
            if multiple_ch:
                return np.array([x])
            return np.array(x)

        p = Pool(processes=cpu_count())
        x_rotated = p.map(self.voxel_rotation, [self.voxels for _ in range(n_rotations)])
        p.close()
        return np.array(x_rotated)

    def compute_channel_generation(self, voxels):
        """Generate multichannel representation using multiprocessing."""
        p = Pool(processes=cpu_count())
        ch_voxels = p.map(self.channels_generator, list(voxels))
        p.close()
        return np.array(ch_voxels)

    def channels_generator(
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

    def voxel_rotation(self, voxels):
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
        transform_matrix = f_b @ f_r @ f_t

        dim0, dim1, dim2 = gs[0].shape
        coords = np.indices((dim0, dim1, dim2)).reshape(3, -1)
        homo_coords = np.vstack([coords, np.ones((1, coords.shape[1]))])

        transformed = np.floor(transform_matrix @ homo_coords).astype(int)
        n_i, n_j, n_k = transformed[0], transformed[1], transformed[2]

        valid = (n_i >= 0) & (n_i < dim0) & (n_j >= 0) & (n_j < dim1) & (n_k >= 0) & (n_k < dim2)

        orig_i, orig_j, orig_k = coords[0][valid], coords[1][valid], coords[2][valid]
        target_i, target_j, target_k = n_i[valid], n_j[valid], n_k[valid]

        for ch in range(len(gs)):
            ngs[ch][target_i, target_j, target_k] = gs[ch][orig_i, orig_j, orig_k]

        return ngs


if __name__ == "__main__":
    if len(sys.argv) == 3:
        vobj = VoxelObject([sys.argv[1]])
        new_voxels = vobj.voxel_rotation(vobj.voxels)
        writeOpenDX(new_voxels[0], vobj.origins[0], vobj.deltas[0], sys.argv[2])
    else:
        print(f"\n Usage: {sys.argv[0]} [input dx] [output dx]\n")
