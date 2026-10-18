#!/usr/bin/env python3
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import sys
from pathlib import Path
from deepmolecularnetwork.framework import load_model


def GetValidationFnc():
    """
    Get custom model from model.py file

    """
    p = Path(".")
    sys.path.append("%s" % (str(p.absolute())))
    try:
        from model import validation

        return validation
    except ImportError:
        print("Please create a model.py file in your directory")
        print("with a validation function which take as input a list of names.")
        return None


def GetTrainTestFnc():
    """
    Get custom model from model.py file

    """
    p = Path(".")
    sys.path.append("%s" % (str(p.absolute())))
    try:
        from model import traintest

        return traintest
    except ImportError:
        print("Please create a model.py file in your directory")
        print(
            "with a traintest function which take as input a list of names, "
            "the %test size and the random number initializator"
        )
        return None


def GetKerasModel():
    """
    Get custom model from model.py file

    """
    p = Path(".")
    sys.path.append("%s" % (str(p.absolute())))
    try:
        from model import build_model

        return build_model
    except ImportError:
        print("Please create a model.py file in your directory")
        print("with a build_model function.")
        return None


def load_model_compat(model_path, **kwargs):
    p = Path(model_path)
    if p.exists():
        return load_model(str(p), **kwargs)
    keras_path = p.parent / f"{p.name}.keras"
    if keras_path.exists():
        return load_model(str(keras_path), **kwargs)
    h5_path = p.parent / f"{p.name}.h5"
    if h5_path.exists():
        return load_model(str(h5_path), **kwargs)
    return load_model(str(model_path), **kwargs)


def GetLoadModelFnc():
    """
    Get custom load model function in model.py file.
    N.B.: the load function should have this fixed name with the
          model path argument as input.

        def load_personal_model(model_path):
            ...
    """
    p = Path(".")
    sys.path.append("%s" % (str(p.absolute())))
    try:
        from model import load_personal_model

        return load_personal_model
    except ImportError:
        return load_model_compat


def ReadDescriptorOrder(mpath):
    odesc = []
    odesc_file = "%s/odesc_header.csv" % (str(Path(mpath).absolute()))
    if Path(odesc_file).exists():
        with open(odesc_file) as f:
            for line in f:
                odesc.append(line.strip())
    return odesc


def LoadKerasModels(mpath):
    """
    function to load models produced using cross validation by
    make1Dmodel.py, make3DCNNmodel.py, makeSMILESmodel.py
    """
    loadfnc = GetLoadModelFnc()
    p = list(Path(mpath).glob("**/*.h5")) + list(Path(mpath).glob("**/*.keras"))
    files = [x for x in p if x.is_file()]
    for file_ in sorted(files):
        yield loadfnc(str(file_))


def GetCudaDevices():
    """
    Get Free Cuda devices...
    """
    return None
