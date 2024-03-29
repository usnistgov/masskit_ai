import pytest
from pathlib import Path
import subprocess
from hydra import compose, initialize
import os

"""
pytest fixtures

Placed in the package so that they can be used as plugins for pytest unit tests in
other packages.  To use in other packages, put
pytest_plugins = ("masskit_ai.test_fixtures",)
in the conftest.py file at the root of the package unit tests

"""

@pytest.fixture(scope="session")
def data_dir():
    """
    the directory containing the test data files

    Note that trying to get data from the masskit/tests/data directory is not straightforward
    as there is no generic way to find the path to this directory as it is not installed.
    The workaround is to copy the needed datafiles from masskit/tests/data to 
    masskit_ai/tests/data
    """
    if Path("tests/data").exists():
        return Path("tests/data")
    elif Path("data").exists():
        return Path("data")
    else:
        raise FileNotFoundError(f'Unable to find test data directory, cwd={os.getcwd()}')

@pytest.fixture(scope="session")
def human_uniprot_trunc_fasta(data_dir):
    return data_dir / "human_uniprot_trunc.fasta"

@pytest.fixture(scope="session")
def human_uniprot_trunc_parquet(human_uniprot_trunc_fasta, tmpdir_factory):
    human_uniprot_trunc_parquet = tmpdir_factory.mktemp('fasta2peptides') / 'human_uniprot_trunc.parquet'
    # we run a subprocess as there isn't any easy way to find configuration in masskit
    # that is needed to run fasta2peptides as a function
    subprocess.run(['fasta2peptides',
                    f"input.file={human_uniprot_trunc_fasta}",
                    f"output.file={human_uniprot_trunc_parquet}"],
                    check=True)
    return human_uniprot_trunc_parquet

@pytest.fixture(scope="session")
def predicted_human_uniprot_trunc_parquet(tmpdir_factory):
    return tmpdir_factory.mktemp('predict_peptides') / 'human_uniprot_trunc'

@pytest.fixture(scope="session")
def predicted_airi_csv(tmpdir_factory):
    return tmpdir_factory.mktemp('predicted_airi') / 'predicted_airi_csv'

@pytest.fixture(scope="session")
def test_new_sdf_parquet(data_dir, tmpdir_factory):
    test_new_sdf_parquet_prefix = tmpdir_factory.mktemp('batch_converter') / 'batch_converted_sdf'
    subprocess.run(['batch_converter', 
                    f"input.file.names={data_dir / 'test.new.sdf'}",
                    f"output.file.name={test_new_sdf_parquet_prefix}",
                    f"output.file.types=[parquet]",
                    f"conversion.row_batch_size=100",
                    f"conversion/sdf=sdf_nist_mol"
                   ], check=True)
    test_new_sdf_path_parquet = f'{test_new_sdf_parquet_prefix}_path.parquet'
    subprocess.run(["shortest_path",
                f'input.file.name={test_new_sdf_parquet_prefix}.parquet',
                f"output.file.name={test_new_sdf_path_parquet}"], check=True)

    return test_new_sdf_path_parquet

@pytest.fixture(scope="session")
def SRM1950_lumos_short_parquet_ai(data_dir, tmpdir_factory):
    SRM1950_lumos_short_parquet_prefix = tmpdir_factory.mktemp('batch_converter') / 'SRM1950_lumos_short_ai'
    subprocess.run(['batch_converter', 
                    f"input.file.names={data_dir / 'SRM1950_lumos_short.sdf'}",
                    f"output.file.name={SRM1950_lumos_short_parquet_prefix}",
                    f"output.file.types=[parquet]",
                    f"conversion/sdf=sdf_nist_mol"
                   ], check=True)
    return SRM1950_lumos_short_parquet_prefix + ".parquet"


# configurations are kept here so that the config_path can resolve correctly

@pytest.fixture(scope="session")
def config_predict_peptide(predicted_human_uniprot_trunc_parquet, human_uniprot_trunc_parquet):
    with initialize(version_base=None, config_path="../apps/ml/peptide/conf"):
        cfg = compose(config_name="config_predict",
                      overrides=[f"input.test.spectral_library={human_uniprot_trunc_parquet}",
                                 f"predict.output_prefix={predicted_human_uniprot_trunc_parquet}",
                                 "predict.output_suffixes=[arrow,msp]",   # [arrow,msp]
                                 "predict.num=6",  # 6
                                 "predict.row_group_size=2",  # 2
                                 "predict.annotate=True",  # True
                                 "predict.upres=True",  # True
                                 ])
        return cfg
    assert False
    


@pytest.fixture(scope="session")
def config_predict_airi_csv(tmpdir_factory, data_dir, predicted_airi_csv):
    batch_converted_csv_files = tmpdir_factory.mktemp('batch_converter') / 'batch_converted_csv'
    test_csv =  data_dir / "test.csv"
    subprocess.run(['batch_converter', 
     f"input.file.names={test_csv}",
     f"output.file.name={batch_converted_csv_files}",
     f"output.file.types=[parquet]",
     f"conversion.row_batch_size=100",
    ], check=True)
    batch_converted_csv_path_file = tmpdir_factory.mktemp('batch_converter') / 'batch_converted_csv_path_file.parquet'
    subprocess.run(["shortest_path",
                    f'input.file.name={batch_converted_csv_files}.parquet',
                    f"output.file.name={batch_converted_csv_path_file}"], check=True)

    with initialize(version_base=None, config_path="../apps/ml/peptide/conf"):
        cfg = compose(config_name="config_predict_ei_ri_2023",
                      overrides=[f"input.test.spectral_library={batch_converted_csv_path_file}",
                                 f"input.test.where=null",
                                 f"predict.output_prefix={predicted_airi_csv}",
                                 "predict.output_suffixes=[csv,arrow]",
                                 "predict.num=3",
                                 "predict.set_to_load=test"
                                 ])
        return cfg
    assert False