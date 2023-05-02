import torch
from masskit.accumulator import AccumulatorProperty
import logging
from pathlib import Path
from masskit_ai.prediction import Predictor
from masskit_ai import _device
import pyarrow as pa


# def create_molprop_prediction_dataset(model, set_to_load='test', dataloader='MolPropDataset', num=0,
#                               predicted_column='predicted_property', return_singleton=True, **kwargs):
#     """
#     Create pandas dataframe(s) that contains mols and can predict a property.

#     :param dataloader: name of the dataloader class, e.g. TandemArrowDataset
#     :param set_to_load: name of the set to use, e.g. "valid", "test", "train"
#     :param model: the model to use to predict spectrum
#     :param num: the number of spectra to predict (0 = all)
#     :param predicted_column: name of the column containing the predicted spectrum
#     :param return_singleton: if there is only one dataframe, don't return lists
#     :return: list of dataframes for doing predictions, list of dataset objects
#     """
#     if dataloader is not None:
#         model.config.ms.dataloader = dataloader
#     # load in all columns
#     model.config.ms.columns = None

#     loaders = setup_datamodule(model.config).create_loader(set_to_load)

#     if isinstance(loaders, list):
#         dfs = [x.dataset.to_pandas() for x in loaders]
#         datasets = loaders
#     else:
#         dfs = [loaders.dataset.to_pandas()]
#         datasets = [loaders]
#     # truncate list if requested
#     if num > 0:
#         for i in range(len(dfs)):
#             dfs[i] = dfs[i].drop(dfs[i].index[num:])
#     # the final predicted spectra.  Will be a consensus of predicted_spectrum_list
#     for df in dfs:
#         df[predicted_column] = [
#             AccumulatorProperty()
#             for _ in range(len(df.index))
#         ]
#         # standard deviation column
#         df[predicted_column + "_stddev"] = None

#     if return_singleton and len(dfs) == 1:
#         return dfs[0], datasets[0]
#     else:
#         return dfs, datasets


# def single_molprop_prediction(model, dataset_element, device=None, **kwargs):
#     """
#     predict a single spectrum

#     :param model: the prediction model
#     :param dataset_element: dataset element
#     :return: the predicted spectrum
#     """
    
#     with torch.no_grad():
#         output = model([dataset_element.x]) # no .to(device) as this is a python molGraph handled in collate_fn
#         property = output.y_prime[0].item() * 10000.0
#     return property


# def finalize_molprop_prediction_dataset(df, predicted_column='predicted_property', **kwargs):
#     """
#     do final processing on the predicted spectra

#     :param df: dataframe containing spectra
#     :param predicted_column:  name of the predicted spectrum column
#     """
#     for j in range(len(df.index)):
#         accumulator = df[predicted_column].iat[j]
#         accumulator.finalize()
#         df[predicted_column].iat[j] = accumulator.predicted_mean
#         df[predicted_column + '_stddev'].iat[j] = accumulator.predicted_stddev
#     df[predicted_column] = df[predicted_column].astype(float)


class MolPropPredictor(Predictor):
    """
    class used to predict multiple spectra per record, which are averaged into
    a single spectrum with a standard deviation
    """

    def __init__(self, config=None, *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
        self.items = []  # the items to predict
        self.output_prefix = str(Path(config.predict.output_prefix).expanduser())
        if "csv" in self.config.predict.output_suffixes:
            self.csv = open(f"{self.output_prefix}.csv", "w")
            if self.csv is None:
                logging.error(f'Unable to open {self.output_prefix}.csv')
        if "arrow" in self.config.predict.output_suffixes:
            self.arrow = None

    def create_dataloaders(self, model):
        """
        Create dataloaders.

        :param model: the model to use to predict spectrum
        :return: list of dataloader objects
        """
        return super().create_dataloaders(model)
    

    def create_items(self, loader, start):
        """
        for a given loader, return back a batch of accumulators

        :param loader: the input loader, assume to contain a TableMap
        :param start: the start row of the batch
        :param length: the length of the batch
        """
        self.items = []

        if self.config.predict.num is not None and self.config.predict.num > 0:
            end = min(start+self.row_group_size, self.config.predict.num + self.original_start, len(loader.dataset.data))
        else:
            end = min(start+self.row_group_size, len(loader.dataset.data))

        for i in range(start, end):
            self.items.append(AccumulatorProperty()) 

        return self.items
        
    def single_prediction(self, model, dataset_element):
        """
        predict a single spectrum

        :param model: the prediction model
        :param dataset_element: dataset element
        :return: the predicted spectrum
        """
        # send input to model, adding a batch dimension
        with torch.no_grad():
            output = model([dataset_element.x]) # no .to(device) as this is a python molGraph handled in collate_fn
            property = output.y_prime[0].item() * 10000.0
            return property

    def add_item(self, idx, item):
        """
        add newly predicted item at index idx
        
        :param idx: index into items
        :param item: item to add
        """
        self.items[idx].add(item)

    def finalize_items(self, dataset, start):
        """
        do final processing on a batch of predicted spectra

        :param items: ListLike of spectra
        :param dataset: dataset containing experimental spectra
        :param start: position of the start of the batch
        """       
        for j in range(len(self.items)):
            self.items[j].finalize()

    def write_items(self):
        """
        write the spectra to files
        
        :param items: the spectra to write
        """
        if "arrow" in self.config.predict.output_suffixes:
            # here we want to combine the dataset.data with two new columns
            # containing the property and std dev
            # need to pass in dataset to write_items, along with start and end
            # end has to come from code in create items
            # also need the names of the new columns
            table = None
            if self.arrow is None:
                self.arrow = pa.RecordBatchFileWriter(pa.OSFile(f'{self.output_prefix}.arrow', 'wb'), table.schema)
                if self.arrow is None:
                    logging.error(f'Unable to open {self.output_prefix}.arrow')
            # self.arrow.write_table(table)
        if "csv" in self.config.predict.output_suffixes and self.csv is not None:
            for item in self.items():
                print(f'{item.predicted_mean},{item.predicted_stddev}',fp=self.csv)
            self.csv.flush()

    def __del__(self):
        # if "arrow" in self.config.predict.output_suffixes:
        #     if self.arrow is not None:
        #         self.arrow.close()
        if "csv" in self.config.predict.output_suffixes:
            if self.csv is not None:
                self.csv.close()


# for command line, need to take original compounds and tautomerize and derivatize them
# would be nice to display a graph.