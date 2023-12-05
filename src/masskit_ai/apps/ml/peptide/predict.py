#!/usr/bin/env python
import logging
import random

import hydra
import numpy as np
import pyarrow as pa
import pytorch_lightning as pl
from hydra.core.plugins import Plugins
from masskit.utils.general import MassKitSearchPathPlugin, class_for_name
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from masskit_ai.loggers import filter_pytorch_lightning_warnings

Plugins.instance().register(MassKitSearchPathPlugin)


# set up matplotlib to use a non-interactive back end
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    matplotlib = None


@hydra.main(config_path="conf", config_name="config_predict", version_base=None)
def predict_app(config):

    # Quick fix for the following error message:
    #
    #   RuntimeWarning: pickle-based deserialization of pyarrow.PyExtensionType subclasses is disabled by default; 
    #   if you only ingest trusted data files, you may re-enable this using `pyarrow.PyExtensionType.set_auto_load(True)`.
    #   In the future, Python-defined extension subclasses should derive from pyarrow.ExtensionType 
    #   (not pyarrow.PyExtensionType) and implement their own serialization mechanism.
    # TODO: needs to be fixed right
    pa.PyExtensionType.set_auto_load(True)

    with logging_redirect_tqdm():

        if config.setup.reproducable_seed is None:
            pl.seed_everything(random.randint(
                np.iinfo(np.uint32).min, np.iinfo(np.uint32).max))
        else:
            pl.seed_everything(config.setup.reproducable_seed)

        predictor = class_for_name(config.paths.modules.prediction,
                                config.predict.get("predictor", "SinglePeptideSpectrumPredictor"))(config)

        # get the first model in order to load the datasets
        loaded_model = config.predict.model_ensemble[0]
        model = predictor.load_model(loaded_model)
        #  get the dataloaders
        predictor.create_dataloaders(model)

        # iterate through datasets
        for dataloader_idx in range(len(predictor.dataloaders)):
            start = predictor.original_start
            # iterate through the batches
            while True:
                logging.info(
                    f'starting batch at {start} for dataset of length {len(predictor.dataloaders[dataloader_idx])}')
                predictor.create_items(dataloader_idx, start)
                # iterate through the models
                for model_idx in tqdm(range(len(config.predict.model_ensemble)), desc="model"):
                    if loaded_model != config.predict.model_ensemble[model_idx]:
                        loaded_model = config.predict.model_ensemble[model_idx]
                        model = predictor.load_model(loaded_model)

                    # iterate through the singletons

                    for idx in tqdm(range(len(predictor.items)), desc='items', leave=False):
                        # predict spectra with multiple draws
                        for _ in range(config.predict.model_draws):
                            # do the prediction
                            new_item = predictor.single_prediction(
                                model, start + idx, dataloader_idx)
                            predictor.add_item(idx, new_item)

                # finalize the batch TODO: how to subset to the predictions?
                predictor.finalize_items(dataloader_idx, start)
                # write the batch out
                predictor.write_items(dataloader_idx, start)
                # increment the batch if not at end
                start += predictor.row_group_size
                if start >= len(predictor.dataloaders[dataloader_idx]) or \
                        (config.predict.num != 0 and start - predictor.original_start >= config.predict.num):
                    break


if __name__ == "__main__":
    filter_pytorch_lightning_warnings()
    predict_app()
