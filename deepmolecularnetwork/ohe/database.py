#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""Database abstractions for one-hot encoded molecular datasets."""

from pathlib import Path
import random
from typing import Any
import numpy as np


def read_one_hot_encoding_csv(ohe_csv: str | Path) -> np.ndarray:
    """Read a CSV file containing one or more one-hot encoded matrices separated by 'END'.

    Args:
        ohe_csv: Path to the CSV file.

    Returns:
        3D numpy array of shape (n_variants, n_features, padding).
    """
    matrices: list[list[list[float]]] = []
    current_matrix: list[list[float]] = []

    with open(ohe_csv, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if "END" in stripped:
                if current_matrix:
                    matrices.append(current_matrix)
                    current_matrix = []
            else:
                current_matrix.append([float(val) for val in stripped.split(",") if val])

    if current_matrix:
        matrices.append(current_matrix)

    return np.asarray(matrices, dtype=np.float32)


# Backward-compatible alias
ReadOneHotEncodingCSV = read_one_hot_encoding_csv


def load_one_hot_encoding_csv_dir(
    ohe_dir: str | Path,
) -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
    """Load all one-hot encoded CSV files from a directory into memory.

    Args:
        ohe_dir: Path to directory containing CSV files.

    Returns:
        Tuple of (dictionary mapping stem to 3D array, input_shape).
    """
    p = Path(ohe_dir).glob("**/*")
    dataset: dict[str, np.ndarray] = {}
    for entry in p:
        if entry.is_file() and entry.suffix.lower() == ".csv":
            key = entry.stem.split(".")[0]
            dataset[key] = read_one_hot_encoding_csv(entry)

    if not dataset:
        raise ValueError(f"No valid .csv files found in directory '{ohe_dir}'")

    first_shape = next(iter(dataset.values())).shape
    input_shape = first_shape[1:] if len(first_shape) == 3 else first_shape
    return dataset, input_shape


# Backward-compatible alias
LoadOneHotEncodingCSVDir = load_one_hot_encoding_csv_dir


class DBFile:
    """Lazy loader for persisted .npy dataset files."""

    def __init__(self, fname: str | Path) -> None:
        """Initialize DBFile.

        Args:
            fname: Path to the .npy file.
        """
        self.fname = Path(str(fname)).resolve()
        self._dbf: np.ndarray | None = None

    def get_data(self) -> np.ndarray:
        """Load and return the memory-mapped numpy array."""
        if self._dbf is None:
            self._dbf = np.load(self.fname, mmap_mode="r")
        return self._dbf

    _LEGACY_ALIASES = {
        "getData": "get_data",
        "getShape": "get_shape",
    }

    def __getattr__(self, name: str) -> Any:
        if name in self._LEGACY_ALIASES:
            return getattr(self, self._LEGACY_ALIASES[name])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_shape(self) -> tuple[int, ...]:
        """Return array shape without loading entire contents into memory."""
        return self.get_data().shape


class OHEDatabase:
    """Manager for saving, loading, and sampling One-Hot Encoded molecular databases."""

    _LEGACY_ALIASES = {
        "saveOHEdb": "save_ohe_db",
        "loadOHEdb": "load_ohe_db",
        "loadOHEdbFromCSVDir": "load_ohe_db_from_csv_dir",
        "getRandomSample": "get_random_sample",
    }

    def __getattr__(self, name: str) -> Any:
        if name in self._LEGACY_ALIASES:
            return getattr(self, self._LEGACY_ALIASES[name])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __init__(self) -> None:
        """Initialize empty database."""
        self.X: dict[str, Any] = {}
        self.input_shape: tuple[int, ...] | None = None
        self.ninstances: int = 0

    def save_ohe_db(self, dbpath: str | Path) -> None:
        """Save database entries into a directory as individual .npy files.

        Args:
            dbpath: Directory path where files will be written.
        """
        target_path = Path(dbpath)
        target_path.mkdir(parents=True, exist_ok=True)

        for key, arr in self.X.items():
            fdb = target_path / f"{key}.npy"
            data = arr.get_data() if isinstance(arr, DBFile) else arr
            np.save(fdb, data)

    def load_ohe_db(self, dbpath: str | Path) -> None:
        """Load database entries from a directory containing .npy files.

        Args:
            dbpath: Directory path containing .npy files.
        """
        self.X = {}
        path_obj = Path(dbpath)
        if not path_obj.is_dir():
            raise FileNotFoundError(f"Directory not found: '{dbpath}'")

        for p in path_obj.iterdir():
            if p.is_file() and p.suffix.lower() == ".npy":
                name = p.stem.split(".")[0]
                self.X[name] = DBFile(p)

        if not self.X:
            raise ValueError(f"No .npy files found in database path '{dbpath}'")

        first_shape = next(iter(self.X.values())).get_shape()
        self.input_shape = first_shape[1:] if len(first_shape) == 3 else first_shape
        self.ninstances = len(self.X)

    def load_ohe_db_from_csv_dir(self, ohe_dir: str | Path) -> None:
        """Populate database from a directory of CSV files.

        Args:
            ohe_dir: Path to directory containing CSV files.
        """
        self.X, self.input_shape = load_one_hot_encoding_csv_dir(ohe_dir)
        self.ninstances = len(self.X)

    def get_random_sample(
        self,
        a: list[str] | int | float | None = None,
        random_seed: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve a subsample of the dataset by key list, count, or fraction.

        Args:
            a: Subset specification: list of compound names, integer count,
                or float fraction (0.0 to 1.0).
            random_seed: Optional seed for reproducible sampling.

        Returns:
            Dictionary containing the selected subset.
        """
        if random_seed is not None:
            random.seed(random_seed)

        sub_x: dict[str, Any] = {}
        if isinstance(a, list):
            for name in a:
                if name in self.X:
                    sub_x[name] = self.X[name]
        elif isinstance(a, (int, np.integer)):
            keys = list(self.X.keys())
            sample_count = min(int(a), len(keys))
            sub_keys = random.sample(keys, sample_count)
            for key in sub_keys:
                sub_x[key] = self.X[key]
        elif isinstance(a, (float, np.floating)):
            keys = list(self.X.keys())
            n = max(1, int(round(self.ninstances * float(a))))
            n = min(n, len(keys))
            sub_keys = random.sample(keys, n)
            for key in sub_keys:
                sub_x[key] = self.X[key]
        elif a is None:
            return self.X.copy()
        else:
            raise TypeError(f"Type {type(a)} not supported for getRandomSample")

        return sub_x
