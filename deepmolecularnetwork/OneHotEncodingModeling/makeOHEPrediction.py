"""Backward compatibility shim for makeOHEPrediction."""

from deepmolecularnetwork.ohe.predict import ModelPredictor, main

__all__ = ["ModelPredictor", "main"]

if __name__ == "__main__":
    main()
