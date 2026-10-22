#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""CLI executable for make3DCNNModel."""

import sys
from deepmolecularnetwork.voxels.model import main

if __name__ == "__main__":
    sys.exit(main())
