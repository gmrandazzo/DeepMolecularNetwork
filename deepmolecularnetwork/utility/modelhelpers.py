#!/usr/bin/env python3
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

from pathlib import Path
import sys

from deepmolecularnetwork.framework import load_model

CREATE_MODEL_PY_MSG = "Please create a model.py file in your directory"


def get_validation_fnc():
    """Get custom validation function from model.py in the current directory."""
    p = Path(".")
    sys.path.append("%s" % (str(p.absolute())))
    try:
        from model import validation

        return validation
    except ImportError:
        print(CREATE_MODEL_PY_MSG)
        print("with a validation function which take as input a list of names.")
        return None


def get_train_test_fnc():
    """Get custom train/test split function from model.py in current directory."""
    p = Path(".")
    sys.path.append("%s" % (str(p.absolute())))
    try:
        from model import traintest

        return traintest
    except ImportError:
        print(CREATE_MODEL_PY_MSG)
        print(
            "with a traintest function which take as input a list of names, "
            "the %test size and the random number initializator"
        )
        return None


def get_keras_model():
    """Get custom model builder function from model.py in current directory."""
    p = Path(".")
    sys.path.append("%s" % (str(p.absolute())))
    try:
        from model import build_model

        return build_model
    except ImportError:
        print(CREATE_MODEL_PY_MSG)
        print("with a build_model function.")
        return None


def _safe_load_model(target_path, **kwargs):
    """Load model with fallback to compile=False if deserialization fails in Keras 3."""
    try:
        return load_model(str(target_path), **kwargs)
    except Exception:
        if kwargs.get("compile", True):
            kwargs_no_compile = dict(kwargs)
            kwargs_no_compile["compile"] = False
            try:
                return load_model(str(target_path), **kwargs_no_compile)
            except Exception:
                pass
        raise


def load_model_compat(model_path, **kwargs):
    """Load Keras model supporting both .h5 and .keras extensions and Keras 2/3 serialization."""
    p = Path(model_path)
    if p.exists():
        return _safe_load_model(p, **kwargs)
    keras_path = p.parent / f"{p.name}.keras"
    if keras_path.exists():
        return _safe_load_model(keras_path, **kwargs)
    h5_path = p.parent / f"{p.name}.h5"
    if h5_path.exists():
        return _safe_load_model(h5_path, **kwargs)
    return _safe_load_model(model_path, **kwargs)


def get_load_model_fnc():
    """Get custom load model function from model.py or default compatibility loader."""
    p = Path(".")
    sys.path.append("%s" % (str(p.absolute())))
    try:
        from model import load_personal_model

        return load_personal_model
    except ImportError:
        return load_model_compat


def read_descriptor_order(mpath):
    """Read the feature ordering saved during model training."""
    odesc = []
    odesc_file = "%s/odesc_header.csv" % (str(Path(mpath).absolute()))
    if Path(odesc_file).exists():
        with open(odesc_file) as f:
            for line in f:
                odesc.append(line.strip())
    return odesc


def load_keras_models(mpath):
    """Generator to load models produced using cross-validation or single model file."""
    loadfnc = get_load_model_fnc()
    p_path = Path(mpath)
    if p_path.is_file():
        yield loadfnc(str(p_path))
        return
    p = list(p_path.glob("**/*.h5")) + list(p_path.glob("**/*.keras"))
    files = [x for x in p if x.is_file()]
    for file_ in sorted(files):
        yield loadfnc(str(file_))


def get_cuda_devices():
    """Get Free Cuda devices."""
    return None


# Backward compatible CamelCase aliases
GetValidationFnc = get_validation_fnc
GetTrainTestFnc = get_train_test_fnc
GetKerasModel = get_keras_model
GetLoadModelFnc = get_load_model_fnc
ReadDescriptorOrder = read_descriptor_order
LoadKerasModels = load_keras_models
GetCudaDevices = get_cuda_devices
