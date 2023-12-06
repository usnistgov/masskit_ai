import pathlib
from abc import ABC, abstractmethod

import numpy as np
import torch
from masskit.utils.general import get_file

from masskit_ai import _device
from masskit_ai.lightning import MasskitDataModule, setup_datamodule
from masskit_ai.spectrum.spectrum_lightning import SpectrumLightningModule


class Predictor(ABC):
    def __init__(self, config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.row_group_size = self.config.predict.get('row_group_size', 5000)
        self.original_start = self.config.predict.get('start', 0)
        # list of dataloaders
        self.dataloaders = None

    def apply_dropout(self, model):
        """
        for use by torch apply to turn on dropout in a model in eval mode.

        :param model: the model
        """
        if type(model) == torch.nn.Dropout:
            model.train()

    def prep_model_for_prediction(self, model):
        """
        prepare the model for inference

        :param model: the model
        :param dropout: should dropout be turned on?
        """
        model.eval()  # eval mode turns off training flag in all layers of model
        if self.config.predict.dropout:  # optionally turn dropout back on
            model.model.apply(self.apply_dropout)

    def load_model(self, model_name):
        filename = get_file(model_name, search_path=self.config.paths.search_path, tgz_extension='.ckpt')
        if filename is None:
            raise ValueError(f'model {model_name} is not found')
        try:
            model = SpectrumLightningModule.load_from_checkpoint(filename, map_location=_device)
        except NotImplementedError:
            posix_backup = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            model = SpectrumLightningModule.load_from_checkpoint(filename, map_location=_device)
            pathlib.PosixPath = posix_backup

        # for some reason, pytorch lightning insists on loading model to cpu even though state dict is on gpu
        # so force it onto the current device
        model.to(device=_device)
        # replace parts of the model configuration to use the configuration for this program
        model.config.input = self.config.input
        model.config.setup = self.config.setup
        model.config.paths = self.config.paths
        # create singleton batches in order
        model.config.ml.shuffle = False
        model.config.ml.batch_size = 1
        self.prep_model_for_prediction(model)
        return model
    
    @abstractmethod
    def create_items(self, dataloader_idx, start):
        """
        create items for holding predictions
        
        :param dataloader_idx: the index of the dataloader in self.dataloaders
        :param start: the start row of the batch
        """
        pass

    @abstractmethod
    def create_dataloaders(self, model):
        model.config.ms.dataloader = self.config.predict.dataloader
        model.config.ms.columns = None

        loaders = setup_datamodule(model.config).create_loader(self.config.predict.set_to_load)

        if isinstance(loaders, list):
            self.dataloaders = loaders
        else:
            self.dataloaders = [loaders]
    
    @abstractmethod
    def single_prediction(self, model, item_idx, dataloader_idx):
        pass
    
    @abstractmethod
    def add_item(self, item_idx, item):
        pass

    @abstractmethod
    def finalize_items(self, dataloader_idx, start):
        pass

    @abstractmethod
    def write_items(self, dataloader_idx, start):
        pass

