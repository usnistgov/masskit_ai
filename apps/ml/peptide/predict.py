import logging

from masskit.utils.general import search_for_file
from masskit_ai.loggers import filter_pytorch_lightning_warnings
from masskit_ai.spectrum.spectrum_lightning import SpectrumLightningModule
import hydra
from pyarrow import plasma
from omegaconf import DictConfig, open_dict
from masskit_ai.spectrum.spectrum_prediction import (
    create_prediction_dataset,
    single_spectrum_prediction, prep_model_for_prediction, finalize_prediction_dataset,
    upres_peptide_spectra, )
from tqdm import tqdm
import pytorch_lightning as pl
import builtins
from masskit.utils.files import spectra_to_array
import pyarrow.parquet as pq


# set up matplotlib to use a non-interactive back end
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    matplotlib = None


@hydra.main(config_path="conf", config_name="config_predict")
def main(config):

    pl.seed_everything(config.setup.reproducable_seed)

    first_model = True  # is this the first model in the ensemble?
    datasets = []  # datasets used to get the records to be predicted
    dfs = []  # dataframes containing the records to be predicted
    max_mz = 0  # maximum mz predicted by first model

    with plasma.start_plasma_store(config.setup.plasma.size) as ps:
        builtins.instance_settings = {'plasma': {'socket': ps[0], 'pid': ps[1].pid}}

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
                dfs, datasets = create_prediction_dataset(
                    model, config.set_to_load, config.dataloader, num=config.num,
                    predicted_spectrum_column=config.predicted_spectrum_column,
                    return_singleton=False,
                )
                first_model = False
                max_mz = model.config.ms.max_mz
            prep_model_for_prediction(model, config.dropout)

            for j in range(len(dfs)):
                for k in tqdm(range(len(dfs[j].index))):
                    # predict spectra with multiple draws
                    for _ in range(config.model_draws):
                        new_spectrum = single_spectrum_prediction(model, datasets[j][k], take_sqrt=config.ms.take_sqrt, l2norm=config.get('l2norm', False))
                        dfs[j][config.predicted_spectrum_column].iat[k].add(new_spectrum)

        for j in range(len(dfs)):
            # create the consensus, including stddev
            finalize_prediction_dataset(dfs[j], predicted_spectrum_column=config.predicted_spectrum_column,
                                        min_intensity=config.min_intensity, mz_window=config.mz_window,
                                        max_mz=max_mz, min_mz=config.min_mz)

            if config.get("upres", False):
                upres_peptide_spectra(dfs[j], predicted_spectrum_column=config.predicted_spectrum_column, max_mz=max_mz, min_mz=config.min_mz)

            logging.info(f'mean cosine score for set {j} is {dfs[j]["cosine_score"].mean()}')

            # write out the predictions
            ending = "" if len(dfs) == 1 else f"_{j}"
            if "csv" in config.output_suffixes:
                logging.info(f'saving {config.output_prefix}{ending}.csv')
                dfs[j].to_csv(f"{config.output_prefix}{ending}.csv")
            if "pkl" in config.output_suffixes:
                logging.info(f'saving {config.output_prefix}{ending}.pkl')
                dfs[j].to_pickle(f"{config.output_prefix}{ending}.pkl")
            if "parquet" in config.output_suffixes:
                logging.info(f'saving {config.output_prefix}{ending}.parquet')
                table = spectra_to_array(dfs[j][config.predicted_spectrum_column], write_starts_stops=config.get("upres", False))
                pq.write_table(table, f"{config.output_prefix}{ending}.parquet", row_group_size=5000)
            if "msp" in config.output_suffixes:
                logging.info(f'saving {config.output_prefix}{ending}.msp')
                dfs[j].lib.to_msp(f"{config.output_prefix}{ending}.msp", spectrum_column='predicted_spectrum', annotate=True)


if __name__ == "__main__":
    filter_pytorch_lightning_warnings()
    main()
