import pytest
from pytest import approx
from pyarrow import plasma
import pytorch_lightning as pl
from masskit_ai.spectrum.small_mol.small_mol_lightning import SearchLightningModule, SmallMolSearchDataModule
from masskit_ai.base_objects import ModelInput
import builtins
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(scope="session")
def start_plasma():
    with plasma.start_plasma_store(8000000000) as ps:
        builtins.instance_settings = {'plasma': {'socket': ps[0], 'pid': ps[1].pid}}
        yield ps

@pytest.fixture()
def config():
    GlobalHydra.instance().clear()
    initialize(config_path='.', version_base=None)
    cfg = compose(config_name="config_search", overrides=['input=2022_tandem_search_test'])
    return cfg

@pytest.mark.skip(reason="need to set up to use test data and missing data/nist/tandem/SRM1950/SRM1950_lumos.ecfp4.pynndescent. also, uses gpu")
def test_SearchLightningModule(config, start_plasma):
    model = SearchLightningModule(config)
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1,
        max_epochs=1,
        limit_train_batches=2,
    )
    dm = SmallMolSearchDataModule(config)
    trainer.fit(model, datamodule=dm)

