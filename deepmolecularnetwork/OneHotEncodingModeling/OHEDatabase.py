"""Backward compatibility shim for OHEDatabase."""

from deepmolecularnetwork.ohe.database import (
    DBFile,
    LoadOneHotEncodingCSVDir,
    OHEDatabase,
    ReadOneHotEncodingCSV,
)

__all__ = [
    "DBFile",
    "LoadOneHotEncodingCSVDir",
    "OHEDatabase",
    "ReadOneHotEncodingCSV",
]
