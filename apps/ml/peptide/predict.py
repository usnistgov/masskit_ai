import logging
from masskit.utils.general import class_for_name, search_for_file
from masskit_ai.loggers import filter_pytorch_lightning_warnings
from masskit_ai.spectrum.spectrum_lightning import SpectrumLightningModule
import hydra
from masskit_ai.prediction import prep_model_for_prediction
from masskit_ai.spectrum.peptide.peptide_prediction import upres_peptide_spectra
import pytorch_lightning as pl
from masskit.utils.files import spectra_to_array
import pyarrow.parquet as pq
import torch

# set up matplotlib to use a non-interactive back end
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    matplotlib = None


@hydra.main(config_path="conf", config_name="config_predict", version_base=None)
def main(config):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pl.seed_everything(config.setup.reproducable_seed)

    first_model = True  # is this the first model in the ensemble?
    datasets = []  # datasets used to get the records to be predicted
    dfs = []  # dataframes containing the records to be predicted
    max_mz = 0  # maximum mz predicted by first model

    # find prediction apis
    # there are separate function instead of one class to simplify use from a jupyter notebook
    prediction_type = config.get("prediction_type", "spectrum")
    create_prediction_dataset = class_for_name(config.paths.modules.prediction,
        config.get("create_prediction_dataset", "create_prediction_dataset"))
    finalize_prediction_dataset = class_for_name(config.paths.modules.prediction,
        config.get("finalize_prediction_dataset", "finalize_prediction_dataset"))
    single_prediction = class_for_name(config.paths.modules.prediction,
        config.get("single_prediction", "single_spectrum_prediction"))

    for i in range(len(config.model_ensemble)):
        filename = search_for_file(config.model_ensemble[i], config.paths.search_path)
        if filename is None:
            raise ValueError(f'model {config.model_ensemble[i]} is not found')
        model = SpectrumLightningModule.load_from_checkpoint(filename)
        model.to(device=device)
        # replace parts of the model configuration to use the configuration for this program
        model.config.input = config.input
        model.config.setup = config.setup
        model.config.paths = config.paths
        # create singleton batches in order
        model.config.ml.shuffle = False
        model.config.ml.batch_size = 1
        if first_model:
            dfs, datasets = create_prediction_dataset(
                model, config.set_to_load, config.dataloader, num=config.num,
                predicted_column=config.predicted_column,
                return_singleton=False,
            )
            first_model = False
            max_mz = model.config.ms.get("max_mz", 2000)
        prep_model_for_prediction(model, config.dropout)

        for j in range(len(dfs)):
            for idx, singleton_batch in enumerate(datasets[j]):
                if config.num is not None and config.num > 0 and idx >= config.num:
                    break
                # predict spectra with multiple draws
                for _ in range(config.model_draws):
                    new_item = single_prediction(model, 
                        singleton_batch,
                        take_sqrt=config.ms.get('take_sqrt', False), 
                        l2norm=config.get('l2norm', False),
                        device=device)
                    dfs[j][config.predicted_column].iat[idx].add(new_item)

    for j in range(len(dfs)):
        # create the consensus, including stddev
        finalize_prediction_dataset(dfs[j], predicted_column=config.predicted_column,
                                    min_intensity=config.get('min_intensity', 0), mz_window=config.get('mz_window',7),
                                    max_mz=max_mz, min_mz=config.get('min_mz', 0))

        if prediction_type == 'spectrum':
            if config.get("upres", False):
                upres_peptide_spectra(dfs[j], predicted_column=config.predicted_column, max_mz=max_mz, min_mz=config.min_mz)
            logging.info(f'mean cosine score for set {j} is {dfs[j]["cosine_score"].mean()}')

        # write out the predictions
        ending = "" if len(dfs) == 1 else f"_{j}"
        if "csv" in config.output_suffixes:
            logging.info(f'saving {config.output_prefix}{ending}.csv')
            dfs[j].to_csv(f"{config.output_prefix}{ending}.csv")
        if "pkl" in config.output_suffixes:
            logging.info(f'saving {config.output_prefix}{ending}.pkl')
            dfs[j].to_pickle(f"{config.output_prefix}{ending}.pkl")
        if prediction_type == 'spectrum':
            if "parquet" in config.output_suffixes:
                logging.info(f'saving {config.output_prefix}{ending}.parquet')
                table = spectra_to_array(dfs[j][config.predicted_column], write_starts_stops=config.get("upres", False))
                pq.write_table(table, f"{config.output_prefix}{ending}.parquet", row_group_size=5000)
            if "msp" in config.output_suffixes:
                logging.info(f'saving {config.output_prefix}{ending}.msp')
                dfs[j].lib.to_msp(f"{config.output_prefix}{ending}.msp", spectrum_column='predicted_spectrum', annotate=True)


if __name__ == "__main__":
    filter_pytorch_lightning_warnings()
    main()
