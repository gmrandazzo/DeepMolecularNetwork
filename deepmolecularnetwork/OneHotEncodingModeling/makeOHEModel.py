"""Backward compatibility shim for makeOHEModel."""

import sys
from deepmolecularnetwork.ohe.model import (
    NNTrain,
    build_gridsearch_model,
    example_build_2DData_model,
    example_build_2d_data_model,
    example_build_model,
    main,
)

__all__ = [
    "NNTrain",
    "example_build_model",
    "example_build_2DData_model",
    "example_build_2d_data_model",
    "build_gridsearch_model",
    "main",
]

if __name__ == "__main__":
    sys.exit(main())
