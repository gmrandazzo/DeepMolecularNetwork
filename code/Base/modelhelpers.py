#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import sys
from pathlib import Path
from keras.models import load_model

def GetKerasModel():
    """
    Get custom model from model.py file
    
    """
    p = Path('.')
    sys.path.append("%s" % (str(p.absolute())))
    try:
        from model import build_model
        return build_model
    except ImportError:
        print("Please create a model.py file in your directory")
        print("with a build_model function.")
        return None

def GetLoadModelFnc():
    """
    Get custom load model function in model.py file.
    N.B.: the load function should have this fixed name with the 
          model path argument as input.
          
        def load_personal_model(model_path):
            ...
    
    Example:
    
    def myscore(ytrue, ypred):
        mask = K.cast(K.not_equal(ytrue,-9999), K.floatx())
        return K.log(K.mean(K.abs(ytrue*mask - ypred*mask), axis=-1)+)
    
    def mymse(ytrue, ypred):
        mask = K.cast(K.not_equal(ytrue,-9999), K.floatx())
        return K.mean(K.square(ytrue*mask - ypred*mask), axis=-1)
        
    def load_personal_model(model_path):
        return load_model(model_path, custom_object={"myscore": myscore,
                                                     "mymse": mymse})
    """
    p = Path('.')
    sys.path.append("%s" % (str(p.absolute())))
    try:
        from model import load_personal_model
        return load_personal_model
    except ImportError:
        return load_model

def LoadKerasModels(mpath, custom_objects_=None):
    """
    function to load models produced using cross validation by
    make1Dmodel.py, make3DCNNmodel.py, makeSMILESmodel.py
    """
    models = []
    p = Path(mpath).glob('**/*.h5')
    # file order based on data creation time
    files = [x for x in p if x.is_file()]
    # Load models
    if custom_objects_ is None:
        for file_ in files:
            models.append(load_model(str(file_)))
    else:
        for file_ in files:
            models.append(load_model(str(file_),  
custom_objects=custom_objects_))
    # Load order descriptorss
    odesc = []
    odesc_file = "%s/odesc_header.csv" % (str(Path(mpath).absolute()))
    if Path(odesc_file).exists():
        f = open(odesc_file, "r")
        for line in f:
            odesc.append(line.strip())
        f.close()
    return models, odesc



def GetCudaDevices():
    """
    Get Free Cuda devices...
    """
    return None


