"""
opendx file read/write
(c) 2018-2023 gmrandazzo@gmail.com
This file is part of DeepMolecularNetwork.
You can use,modify, and distribute it under
the terms of the GNU General Public Licenze, version 3.
See the file LICENSE for details
"""

import numpy as np


def nsplit(line, sep):
    v = str.split(line.strip(), sep)
    return list(filter(None, v))


def _extract_numbers(stripped, values):
    if "attribute" in stripped:
        parts = nsplit(stripped, "attribute")[:-1]
        if not parts:
            return
        stripped = str(parts[0])

    v = nsplit(stripped, " ")
    if 1 <= len(v) <= 3:
        values.extend(float(item) for item in v)


def readOpenDX(ifile):
    voxel = None
    origin = None
    delta = []
    values = []
    grid_size = None
    c = 1

    with open(ifile) as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if "gridpositions counts" in stripped:
                v = nsplit(stripped, " ")
                grid_size = [int(v[5]), int(v[6]), int(v[7])]
                voxel = np.zeros(grid_size)
            elif "origin" in stripped:
                v = nsplit(stripped, " ")
                origin = [float(v[1]), float(v[2]), float(v[3])]
            elif "delta" in stripped:
                v = nsplit(stripped, " ")
                delta.append(float(v[c]))
                c += 1
            else:
                _extract_numbers(stripped, values)

    if grid_size is not None:
        total = grid_size[0] * grid_size[1] * grid_size[2]
        voxel = np.array(values[:total], dtype=np.float64).reshape(grid_size)

    return voxel, origin, delta


def writeOpenDX(voxel, origin, delta, outfile):
    s0, s1, s2 = voxel.shape[0], voxel.shape[1], voxel.shape[2]
    with open(outfile, "w") as fo:
        fo.write(f"object 1 class gridpositions counts {s0} {s1} {s2}\n")
        fo.write(f"origin {origin[0]:f} {origin[1]:f} {origin[2]:f}\n")
        fo.write(f"delta {delta[0]:f} 0.0 0.0\n")
        fo.write(f"delta 0.0 {delta[1]:f} 0.0\n")
        fo.write(f"delta 0.0 0.0 {delta[2]:f}\n")
        fo.write(f"object 2 class gridconnections counts {s0} {s1} {s2}\n")
        total_items = voxel.shape[0] * voxel.shape[1] * voxel.shape[2]
        fo.write(f"object 3 class array type float rank 0 items {total_items} data follows\n")
        flat = voxel.ravel()
        for i in range(0, len(flat), 3):
            chunk = flat[i : i + 3]
            fo.write(" ".join(f"{val:.6e}" for val in chunk) + "\n")
        fo.write('object  "regular positions regular connections" class field\n')
        fo.write('component "positions" value 1\n')
        fo.write('component "connections" value 2\n')
        fo.write('component "data" value 3\n')


# Snake_case aliases for modern standard conventions
read_open_dx = readOpenDX
write_open_dx = writeOpenDX
