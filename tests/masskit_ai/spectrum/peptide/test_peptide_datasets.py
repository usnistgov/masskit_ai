import pytest
from pytest import approx
from masskit_ai.spectrum.spectrum_datasets import TandemArrowDataset
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

@pytest.fixture()
def config():
    GlobalHydra.instance().clear()
    initialize(config_path='.', version_base=None)
    cfg = compose(config_name="config")
    return cfg

def test_TandemArrowDataset(config):
    ds = TandemArrowDataset('data/cho_uniq_short.parquet', config, 'train')
    pass
