import logging
from masskit.utils.general import class_for_name
from masskit_ai.loggers import filter_pytorch_lightning_warnings
import hydra
import pytorch_lightning as pl
from tqdm import tqdm

# set up matplotlib to use a non-interactive back end
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    matplotlib = None


@hydra.main(config_path="conf", config_name="config_predict", version_base=None)
def predict_app(config):

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
            logging.info(f'starting batch at {start} for dataset of length {len(predictor.dataloaders[dataloader_idx])}')
            predictor.create_items(dataloader_idx, start)
            # iterate through the models
            for model_idx in range(len(config.predict.model_ensemble)):
                if loaded_model != config.predict.model_ensemble[model_idx]:
                    loaded_model = config.predict.model_ensemble[model_idx]
                    model = predictor.load_model(loaded_model)

                # iterate through the singletons
                for idx in tqdm(range(len(predictor.items))):
                    # predict spectra with multiple draws
                    for _ in range(config.predict.model_draws):
                        # do the prediction
                        new_item = predictor.single_prediction(model, start + idx, dataloader_idx)
                        predictor.add_item(idx, new_item)

            # finalize the batch TODO: how to subset to the predictions?
            predictor.finalize_items(dataloader_idx, start)
            # write the batch out
            predictor.write_items(dataloader_idx, start)
            # increment the batch if not at end
            start += predictor.row_group_size
            if start >= len(predictor.dataloaders[dataloader_idx]) or \
                start - predictor.original_start >= config.predict.num:
                break

        # if prediction_type == 'spectrum':
        #     if "cosine_score" in dfs[j].get_props():
        #         logging.info(f'mean cosine score for set {j} is {dfs[j]["cosine_score"].mean()}')


if __name__ == "__main__":
    filter_pytorch_lightning_warnings()
    predict_app()
