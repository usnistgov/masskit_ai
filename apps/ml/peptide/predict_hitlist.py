import logging
from hitlist import Hitlist, PeptideIdentityScore

from index import ArrowLibraryMap

from masskit.utils.general import parse_filename, search_for_file
from masskit_ai.loggers import filter_pytorch_lightning_warnings
from masskit_ai.spectrum.spectrum_lightning import SpectrumLightningModule
import hydra
from omegaconf import DictConfig, open_dict
from masskit_ai.spectrum.spectrum_prediction import (
    create_prediction_dataset,
    create_prediction_dataset_from_hitlist,
    single_spectrum_prediction, prep_model_for_prediction, finalize_prediction_dataset,
    upres_peptide_spectra, )
from tqdm import tqdm
import pytorch_lightning as pl
from masskit.utils.files import load_mzTab
import pyarrow.parquet as pq
import pandas as pd

"""
Program to add experimental and predicted spectra to a hitlist
todo:
= copy energy to column

"""


@hydra.main(config_path="conf", config_name="config_predict_hitlist")
def main(config):

    pl.seed_everything(config.setup.reproducable_seed)

    first_model = True  # is this the first model in the ensemble?
    max_mz = 0  # maximum mz predicted by first model

    for i in range(len(config.model_ensemble)):
        filename = search_for_file(config.model_ensemble[i], config.paths.search_path)
        if filename is None:
            raise ValueError(f'model {config.model_ensemble[i]} is not found')
        model = SpectrumLightningModule.load_from_checkpoint(filename)
        # replace parts of the model configuration to use the configuration for this program
        model.config.input = config.input
        model.config.setup = config.setup
        model.config.paths = config.paths
        if first_model:
            in_file_root, in_file_extension = parse_filename(config.hitlist_in)
            if in_file_extension.lower() == 'mztab':
                hitlist = load_mzTab(config.hitlist_in)
            else:
                hitlist = pd.read_pickle(config.hitlist_in)
                hitlist = Hitlist(hitlist)
            experimental_tablemap = ArrowLibraryMap.from_parquet(config.input[config.set_to_load].spectral_library)

            df, dataset = create_prediction_dataset_from_hitlist(model, 
                                                                   hitlist, 
                                                                   experimental_tablemap, 
                                                                   set_to_load=config.set_to_load,
                                                                   num=config.num,
                                                                   copy_annotations=False,
                                                                   predicted_spectrum_column=config.predicted_spectrum_column, 
                                                                   return_singleton=True
                                                                   )
            first_model = False
            max_mz = model.config.ms.max_mz
        prep_model_for_prediction(model, config.dropout)

        for k in tqdm(range(len(df.index))):
            # predict spectra with multiple draws
            for _ in range(config.model_draws):
                new_spectrum = single_spectrum_prediction(model, dataset[k], take_sqrt=config.ms.take_sqrt, l2norm=config.get('l2norm', False))
                df[config.predicted_spectrum_column].iat[k].add(new_spectrum)

        # create the consensus, including stddev
    finalize_prediction_dataset(df, 
                                predicted_spectrum_column=config.predicted_spectrum_column,
                                min_intensity=config.min_intensity, 
                                mz_window=config.mz_window,
                                max_mz=max_mz, 
                                min_mz=config.min_mz
                                )
    if config.get("upres", False):
        upres_peptide_spectra(df, predicted_spectrum_column=config.predicted_spectrum_column, max_mz=max_mz, min_mz=config.min_mz)

    hitlist = Hitlist(df) 
    PeptideIdentityScore().score(hitlist)
    logging.info(f'mean cosine score is {df["cosine_score"].mean()}')

    # write out the predictions
    hitlist.hitlist.to_pickle(f"{config.output_prefix}.pkl")


if __name__ == "__main__":
    filter_pytorch_lightning_warnings()
    main()
