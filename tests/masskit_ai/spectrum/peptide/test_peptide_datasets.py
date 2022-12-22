import pytest
from pytest import approx
from pyarrow import plasma
from masskit_ai.spectrum.spectrum_datasets import TandemArrowDataset
import builtins
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(scope="session")
def start_plasma():
    with plasma.start_plasma_store(1000000000) as ps:
        builtins.instance_settings = {'plasma': {'socket': ps[0], 'pid': ps[1].pid}}
        yield ps

@pytest.fixture()
def config():
    GlobalHydra.instance().clear()
    initialize(config_path='.', version_base=None)
    cfg = compose(config_name="config")
    return cfg

def test_TandemArrowDataset(config, start_plasma):
    ds = TandemArrowDataset('data/cho_uniq_short.parquet', config, 'train')
    pass
