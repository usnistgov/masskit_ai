import pytest
from pytest import approx
from masskit_ai.spectrum.small_mol.small_mol_datasets import TandemArrowSearchDataset
from masskit_ai.spectrum.small_mol.models.small_mol_models import *
import torch
from masskit_ai.base_objects import ModelInput
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

@pytest.fixture()
def config():
    GlobalHydra.instance().clear()
    initialize(config_path='..', version_base=None)
    cfg = compose(config_name="config_search", overrides=['input=2022_tandem_search_test', "ml/model=ResNetBaseline"])
    return cfg

@pytest.fixture()
def ds(config):
    ds = TandemArrowSearchDataset('data/nist/tandem/srm/1950/SRM1950_lumos.parquet', config, 'test',
                                store_search='libraries/tests/data/SRM1950_lumos.parquet')
    return ds

@pytest.mark.skip(reason="uses gpu, missing libraries/tests/data/SRM1950_lumos.parquet")
def test_SimpleModel(config, ds):
    y = torch.unsqueeze(ds.get_y(ds.get_data_row(0))[1], 0)
    x = ModelInput(x=ds.get_x(ds.get_data_row(0))[1], y=None, index=None)
    model = SimpleNet(config)
    y_prime = model(x).y_prime
    assert tuple(y.shape) == (1,)
    assert tuple(x.x.shape) == (2, model.bins)
    assert tuple(y_prime.shape) == (2, config.ml.model.SimpleModel.fp_size)

@pytest.mark.skip(reason="uses gpu, missing libraries/tests/data/SRM1950_lumos.parquet")
def test_AIMSNet(config, ds):
    y = torch.unsqueeze(ds.get_y(ds.get_data_row(0))[1], 0)
    x = ModelInput(x=ds.get_x(ds.get_data_row(0))[1], y=None, index=None)
    model = AIMSNet(config)
    y_prime = model(x).y_prime
    assert tuple(y.shape) == (1,)
    assert tuple(x.x.shape) == (2, model.bins)
    assert tuple(y_prime.shape) == (2, config.ml.model.AIMSNet.fp_size)

@pytest.mark.skip(reason="needs data/nist/tandem/SRM1950/SRM1950_lumos.ecfp4.pynndescent")
def test_ResNetBaseline(config, ds):
    y = torch.unsqueeze(ds.get_y(ds.get_data_row(0))[1], 0)
    x = ModelInput(x=ds.get_x(ds.get_data_row(0))[1], y=None, index=None)
    model = ResNetBaseline(config)
    y_prime = model(x).y_prime
    assert tuple(y.shape) == (1,)
    assert tuple(x.x.shape) == (2, model.bins)
    assert tuple(y_prime.shape) == (2, config.ml.model.ResNetBaseline.fp_size)

