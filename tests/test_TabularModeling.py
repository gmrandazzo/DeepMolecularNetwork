import numpy as np
import random
from deepmolecularnetwork.Base.dmnnio import(
    ReadDescriptors,
    ReadTarget
)
from deepmolecularnetwork.TableModeling.makeTABModel import NNTrain

def test_makeTABModel():
    np.random.seed(12345)
    random.seed(12345)
    X_raw, nfeatures_, xheader = ReadDescriptors("data/example_01/dataset.desc.csv")
    target = ReadTarget("data/example_01/target.csv")
    nn = NNTrain(X_raw, target, xheader)
    nn.verbose = 1
    nn.simplerun(256,
                0,
                10,
                3,
                256,
                12345,
                "testmodel")
