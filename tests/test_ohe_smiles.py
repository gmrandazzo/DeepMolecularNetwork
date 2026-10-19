"""Unit tests for SMILES parsing, One-Hot Encoding database, and OHE CNN modeling."""

from pathlib import Path
import tempfile
import numpy as np

from deepmolecularnetwork.ohe.database import (
    DBFile,
    OHEDatabase,
    ReadOneHotEncodingCSV,
)
from deepmolecularnetwork.ohe.model import (
    NNTrain,
    build_gridsearch_model,
    example_build_2DData_model,
    example_build_model,
)
from deepmolecularnetwork.ohe.smiles2matrix import DEFAULT_SMILE_CHARS, SMILES2MX


def test_smiles2mx_parsing_and_shape():
    """Test SMILES tokenization and 2D matrix conversion."""
    s2m = SMILES2MX(row_padding=32)
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    tokens = s2m.parse(smi)
    assert len(tokens) > 0
    assert "C" in tokens
    assert "c" in tokens
    assert "=" in tokens

    shape = s2m.getshape()
    assert shape == (len(DEFAULT_SMILE_CHARS), 32)

    matrix = s2m.smi2mx(smi)
    assert len(matrix) == len(DEFAULT_SMILE_CHARS)
    assert len(matrix[0]) == 32

    arr = s2m.smi2array(smi)
    assert arr.shape == (len(DEFAULT_SMILE_CHARS), 32)
    assert np.all(np.isin(arr, [0, 1]))
    assert np.sum(arr) == len(tokens)


def test_ohe_database_io_and_sampling():
    """Test OHEDatabase persistence, DBFile lazy loading, and getRandomSample."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dbpath = Path(tmpdir) / "ohedb"
        db = OHEDatabase()
        s2m = SMILES2MX(row_padding=16)

        mol_data = {
            "MOL1": s2m.smi2array("CCO"),
            "MOL2": s2m.smi2array("c1ccccc1"),
            "MOL3": s2m.smi2array("CC(=O)O"),
        }
        db.X = mol_data
        db.ninstances = len(mol_data)
        db.saveOHEdb(dbpath)

        loaded_db = OHEDatabase()
        loaded_db.loadOHEdb(dbpath)

        assert loaded_db.ninstances == 3
        assert "MOL1" in loaded_db.X
        assert isinstance(loaded_db.X["MOL1"], DBFile)
        assert loaded_db.X["MOL1"].getData().shape == (len(DEFAULT_SMILE_CHARS), 16)

        # Sampling tests
        sample_by_key = loaded_db.getRandomSample(["MOL1", "MOL3"])
        assert len(sample_by_key) == 2
        assert "MOL2" not in sample_by_key

        sample_by_int = loaded_db.getRandomSample(2, random_seed=42)
        assert len(sample_by_int) == 2

        sample_by_frac = loaded_db.getRandomSample(0.5, random_seed=42)
        assert len(sample_by_frac) >= 1


def test_read_onehotencoding_csv():
    """Test parsing CSV matrices separated by END."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("1,0,0\n0,1,0\nEND\n0,0,1\n1,0,0\nEND\n")

        arr = ReadOneHotEncodingCSV(csv_file)
        assert arr.shape == (2, 2, 3)


def test_ohe_models_and_nntrain_forward_pass():
    """Test OHE model architectures forward pass and NNTrain pipeline."""
    dshape = (len(DEFAULT_SMILE_CHARS), 16)

    # 1-branch model
    model1 = example_build_model(dshape, nfilters=8, nunits=16)
    dummy_input1 = np.zeros((2, dshape[0], dshape[1], 1), dtype=np.float32)
    out1 = model1(dummy_input1)
    assert out1.shape == (2, 1)

    # 2-branch model with tabular descriptors
    model2 = example_build_2DData_model(dshape, input_shape2=5, nfilters=8, nunits=16)
    dummy_desc = np.zeros((2, 5), dtype=np.float32)
    out2 = model2([dummy_input1, dummy_desc])
    assert out2.shape == (2, 1)

    # Grid search model
    model3 = build_gridsearch_model(dshape, n_conv=1, nfilters=8, ndense_layers=1, nunits=16)
    out3 = model3(dummy_input1)
    assert out3.shape == (2, 1)

    # NNTrain end-to-end simplerun
    with tempfile.TemporaryDirectory() as tmpdir:
        dbpath = Path(tmpdir) / "ohedb"
        db = OHEDatabase()
        s2m = SMILES2MX(row_padding=16)
        db.X = {f"MOL{i}": s2m.smi2array("CCO") for i in range(10)}
        db.ninstances = len(db.X)
        db.saveOHEdb(dbpath)

        targets = {f"MOL{i}": float(i * 1.5) for i in range(10)}
        trainer = NNTrain(dbpath, targets)
        trainer.verbose = 0
        trained_model, metrics = trainer.simplerun(
            batch_size=4,
            num_epochs=1,
            nfilters=4,
            nunits=8,
            random_state=42,
        )
        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
