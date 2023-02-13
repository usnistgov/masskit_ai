import pytest
from pytest import approx
from masskit_ai.lightning import setup_datamodule
from masskit_ai.mol.mol_datasets import MolPropDataset
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import pytorch_lightning as pl
from masskit_ai.spectrum.spectrum_lightning import SpectrumDataModule, SpectrumLightningModule

@pytest.fixture()
def config_ri():
    GlobalHydra.instance().clear()
    initialize(config_path='.', version_base=None)
    cfg = compose(config_name="config_ri", overrides=['input.train.spectral_library=data/mainlib_2017_path_trunc.parquet',
        'input.train.where=null',
        'input.valid.spectral_library=data/mainlib_2017_path_trunc.parquet',
        'input.valid.where=null',
    ])
    return cfg

def test_MolPropDataset(config_ri):
    ds = MolPropDataset('data/mainlib_2017_path_trunc.parquet', config_ri, 'train')
    assert ds.get_data_row(0)['name'] == "Valeramide, 5-chloro-N-propyl-N-octyl-"

def test_SearchLightningModule(config_ri):
    model = SpectrumLightningModule(config_ri)
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1,
        max_epochs=1,
        limit_train_batches=9,
        limit_val_batches=9,
        log_every_n_steps=3,
    )
    dm = setup_datamodule(config_ri) # MolDataModule
    trainer.fit(model, datamodule=dm)