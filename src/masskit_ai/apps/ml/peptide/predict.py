import logging
from masskit.utils.general import class_for_name, get_file
from masskit_ai.loggers import filter_pytorch_lightning_warnings
from masskit_ai.spectrum.spectrum_lightning import SpectrumLightningModule
import hydra
import pytorch_lightning as pl
from masskit.utils.files import spectra_to_array, spectra_to_msp, spectra_to_mgf
import torch
from tqdm import tqdm

from masskit_ai.spectrum.spectrum_prediction import PeptideSpectrumPredictor

# set up matplotlib to use a non-interactive back end
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    matplotlib = None


@hydra.main(config_path="conf", config_name="config_predict", version_base=None)
def main(config):

    pl.seed_everything(config.setup.reproducable_seed)

    # find prediction apis
    # there are separate function instead of one class to simplify use from a jupyter notebook
    # prediction_type = config.get("prediction_type", "spectrum")
    # create_prediction_dataset = class_for_name(config.paths.modules.prediction,
    #     config.get("create_prediction_dataset", "create_prediction_dataset"))
    # finalize_prediction_dataset = class_for_name(config.paths.modules.prediction,
    #     config.get("finalize_prediction_dataset", "finalize_prediction_dataset"))
    # single_prediction = class_for_name(config.paths.modules.prediction,
    #     config.get("single_prediction", "single_spectrum_prediction"))
    
    predictor = PeptideSpectrumPredictor(config, batch_size=2)

    # get the first model in order to load the datasets
    loaded_model = config.model_ensemble[0]
    model = predictor.load_model(loaded_model)
    #  get the dataloaders
    dataloaders = predictor.create_dataloaders(model)

    # iterate through datasets
    for dataloader in dataloaders:
        start = 0
        # iterate through the batches
        while True:
            spectra = predictor.get_items(dataloader, start)
            # iterate through the models
            for i in range(len(config.model_ensemble)):
                if loaded_model != config.model_ensemble[i]:
                    loaded_model = config.model_ensemble[i]
                    model = predictor.load_model(loaded_model)

                # iterate through the singletons
                for idx in range(len(spectra)):
                    # predict spectra with multiple draws
                    for _ in range(config.model_draws):
                        # do the prediction
                        new_item = predictor.single_prediction(model, dataloader.dataset[start + idx])
                        spectra[idx].add(new_item)

            # finalize the batch TODO: how to subset to the predictions?
            predictor.finalize_items(spectra, dataloader, start)
            # write the batch out
            predictor.write_items(spectra)
            # increment the batch if not at end
            start += predictor.batch_size
            if start >= len(dataloader):
                break

        # if prediction_type == 'spectrum':
        #     if "cosine_score" in dfs[j].get_props():
        #         logging.info(f'mean cosine score for set {j} is {dfs[j]["cosine_score"].mean()}')


if __name__ == "__main__":
    filter_pytorch_lightning_warnings()
    main()
