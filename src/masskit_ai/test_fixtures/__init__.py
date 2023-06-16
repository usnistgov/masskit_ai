import pytest
from hydra import compose, initialize
from masskit_ai.apps.ml.peptide import predict

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
                                 f"predict.output_prefix={predicted_human_uniprot_trunc_parquet}",
                                 "predict.output_suffixes=[arrow,msp]",   # [arrow,msp]
                                 "predict.num=6",  # 6
                                 "predict.row_group_size=2",  # 2
                                 "predict.annotate=True",  # True
                                 "predict.upres=True",  # True
                                 ])
        return cfg

@pytest.fixture(scope="session")
def predicted_airi_parquet(tmpdir_factory):
    return tmpdir_factory.mktemp('predicted_airi') / 'predicted_airi'

@pytest.fixture(scope="session")
def config_predict_airi(predicted_airi_parquet, batch_converted_smiles_path_file):
    with initialize(version_base=None, config_path="../apps/ml/peptide/conf"):
        cfg = compose(config_name="config_predict_ei_ri_2023",
                      overrides=[f"input.test.spectral_library={batch_converted_smiles_path_file}.parquet",
                                 f"predict.output_prefix={predicted_airi_parquet}",
                                 "predict.output_suffixes=[csv]",   # [arrow,csv]
                                 "predict.num=2",  # 6
                                 ])
        return cfg

