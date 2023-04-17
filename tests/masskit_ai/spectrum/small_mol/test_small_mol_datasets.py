import pytest
from pytest import approx
from masskit_ai.spectrum.small_mol.small_mol_datasets import TandemArrowSearchDataset
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

@pytest.fixture()
def config():
    GlobalHydra.instance().clear()
    initialize(config_path='.', version_base=None)
    cfg = compose(config_name="config_search", 
                  overrides=['input=2022_tandem_search_test',
                             'setup.num_workers=0'
                             ])
    return cfg

@pytest.mark.skip(reason="needs data/nist/tandem/SRM1950/SRM1950_lumos.ecfp4.pynndescent")
def test_TandemArrowSearchDataset(config):
    ds = TandemArrowSearchDataset('libraries/tests/data/SRM1950_lumos.parquet', config, 'test',
                                  store_search='libraries/tests/data/SRM1950_lumos.parquet')
    row_with_hits = ds.get_data_row(0)
    assert row_with_hits['hit_spectrum'][0].name == 'N-Acetyl-L-alanine'
    assert ds.get_x(ds.get_data_row(0)).shape == (30, 2, 20000)
    pass
