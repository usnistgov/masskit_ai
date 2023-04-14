import pytest
from hydra import compose, initialize
from masskit_ai.apps.ml.peptide import predict
import os
from pathlib import Path

"""
pytest fixtures

Placed in the package so that they can be used as plugins for pytest unit tests in
other packages.  To use in other packages, put
pytest_plugins = ("masskit_ai.test_fixtures",)
in the conftest.py file at the root of the package unit tests

"""

@pytest.fixture(scope="session")
def predicted_human_uniprot_trunc_parquet(tmpdir_factory):
    return tmpdir_factory.mktemp('predict_peptides') / 'human_uniprot_trunc'

@pytest.fixture(scope="session")
def config_predict_peptide(predicted_human_uniprot_trunc_parquet, create_peptide_parquet_file):
    with initialize(version_base=None, config_path="../apps/ml/peptide/conf"):
        cfg = compose(config_name="config_predict",
                      overrides=[f"input.test.spectral_library={create_peptide_parquet_file}",
                                 f"output_prefix={predicted_human_uniprot_trunc_parquet}",
                                 "output_suffixes=[parquet,msp]",
                                 "num=6"])
        return cfg

@pytest.fixture(scope="session")
def create_predicted_peptide_parquet_file(config_predict_peptide):
    predict.main(config_predict_peptide)
    return config_predict_peptide.output_prefix+"."+config_predict_peptide.output_suffixes[0], config_predict_peptide.output_prefix+"."+config_predict_peptide.output_suffixes[1]
