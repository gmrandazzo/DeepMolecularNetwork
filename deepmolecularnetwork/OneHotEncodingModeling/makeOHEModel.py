"""Backward compatibility shim for makeOHEModel."""

from deepmolecularnetwork.ohe.model import (
    NNTrain,
    build_gridsearch_model,
    example_build_2DData_model,
    example_build_2d_data_model,
    example_build_model,
)

__all__ = [
    "NNTrain",
    "example_build_model",
    "example_build_2DData_model",
    "example_build_2d_data_model",
    "build_gridsearch_model",
]
