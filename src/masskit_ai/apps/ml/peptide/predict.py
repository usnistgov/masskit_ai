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
def main(config):

    pl.seed_everything(config.setup.reproducable_seed)
    
    predictor = class_for_name(config.paths.modules.prediction,
                               config.predict.get("predictor", "SinglePeptideSpectrumPredictor"))(config)

    # get the first model in order to load the datasets
    loaded_model = config.predict.model_ensemble[0]
    model = predictor.load_model(loaded_model)
    #  get the dataloaders
    dataloaders = predictor.create_dataloaders(model)

    # iterate through datasets
    for dataloader in dataloaders:
        start = predictor.original_start
        # iterate through the batches
        while True:
            logging.info(f'starting batch at {start} for dataset of length {len(dataloader)}')
            predictor.create_items(dataloader, start)
            # iterate through the models
            for i in range(len(config.predict.model_ensemble)):
                if loaded_model != config.predict.model_ensemble[i]:
                    loaded_model = config.predict.model_ensemble[i]
                    model = predictor.load_model(loaded_model)

                # iterate through the singletons
                for idx in tqdm(range(len(predictor.items))):
                    # predict spectra with multiple draws
                    for _ in range(config.predict.model_draws):
                        # do the prediction
                        # some implementation notes: the dataloader, since it iterates
                        # over batches, doesn't have __getitem__, so we use the dataset instead
                        # to get a single record. We use the collate_fn, perhaps incorrectly
                        # to convert the input data to data for the model.  However, since we are
                        # using the dataset to iterate, we have to explicitly call the collate_fn
                        # putting the argument into a list to fake a batch of size 1, since collate_fn
                        # is intended to work on batches.  In the future, we may wish to move the
                        # collate_fn functionality into the dataset and also predict on batches
                        # of size greater than one (may require a special purpose sampler to 
                        # use start to set the start of the batch).
                        new_item = predictor.single_prediction(model, dataloader.collate_fn([dataloader.dataset[start + idx]]))
                        predictor.add_item(idx, new_item)

            # finalize the batch TODO: how to subset to the predictions?
            predictor.finalize_items(dataloader, start)
            # write the batch out
            predictor.write_items()
            # increment the batch if not at end
            start += predictor.row_group_size
            if start >= len(dataloader) or start - predictor.original_start >= config.predict.num:
                break

        # if prediction_type == 'spectrum':
        #     if "cosine_score" in dfs[j].get_props():
        #         logging.info(f'mean cosine score for set {j} is {dfs[j]["cosine_score"].mean()}')


if __name__ == "__main__":
    filter_pytorch_lightning_warnings()
    main()
