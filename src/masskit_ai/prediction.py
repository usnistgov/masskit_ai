import numpy as np
import torch
from abc import ABC, abstractmethod
from masskit.utils.general import get_file
from masskit_ai.spectrum.spectrum_lightning import SpectrumLightningModule
class Predictor(ABC):
    def __init__(self, config=None, row_batch_size=25000, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.row_batch_size = row_batch_size
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

    def apply_dropout(self, model):
        """
        for use by torch apply to turn on dropout in a model in eval mode.

        :param model: the model
        """
        if type(model) == torch.nn.Dropout:
            model.train()

    def prep_model_for_prediction(self, model, cpu=False):
        """
        prepare the model for inference

        :param model: the model
        :param dropout: should dropout be turned on?
        :param model: place model on cpu?
        """
        if cpu:
            model.cpu()
        model.eval()  # eval mode turns off training flag in all layers of model
        if self.config.predict.dropout:  # optionally turn dropout back on
            model.model.apply(self.apply_dropout)

    def load_model(self, model_name):
        filename = get_file(model_name, search_path=self.config.paths.search_path, tgz_extension='.ckpt')
        if filename is None:
            raise ValueError(f'model {model_name} is not found')
        model = SpectrumLightningModule.load_from_checkpoint(filename)
        # model.to(device=self.device)
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
    def get_items(self, loader, start):
        pass

    @abstractmethod
    def create_dataloaders(self, model):
        pass

    @abstractmethod
    def single_prediction(self, model, dataset_element):
        pass

    @abstractmethod
    def finalize_items(self, items, dataset, start):
        pass

    @abstractmethod
    def write_items(self, items):
        pass

