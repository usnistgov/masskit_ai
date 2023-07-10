import torch
from masskit.accumulator import AccumulatorProperty
import logging
from pathlib import Path
from masskit_ai.prediction import Predictor
from masskit_ai import _device
import pyarrow as pa
from pyarrow import csv
import numpy as np
from masskit.data_specs.file_schemas import csv_drop_fields


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
            self.csv = None
        if "arrow" in self.config.predict.output_suffixes:
            self.arrow = None

    def create_dataloaders(self, model):
        """
        Create dataloaders.

        :param model: the model to use to predict spectrum
        :return: list of dataloader objects
        """
        return super().create_dataloaders(model)
    
    def create_items(self, dataloader_idx, start):
        """
        for a given loader, return back a batch of accumulators

        :param dataloader_idx: the index of the dataloader in self.dataloaders
        :param start: the start row of the batch
        """
        self.items = []

        if self.config.predict.num is not None and self.config.predict.num > 0:
            end = min(start+self.row_group_size, self.config.predict.num + self.original_start,
                      len(self.dataloaders[dataloader_idx].dataset.data))
        else:
            end = min(start+self.row_group_size, len(self.dataloaders[dataloader_idx].dataset.data))

        for i in range(start, end):
            self.items.append(AccumulatorProperty()) 

        return self.items
        
    def single_prediction(self, model, item_idx, dataloader_idx):
        """
        predict a single spectrum

        :param model: the prediction model
        :param item_idx: the index of item in the current dataset
        :param dataloader_idx: the index of the dataloader in self.dataloaders
        :return: the predicted spectrum
        """
        dataset_element = self.dataloaders[dataloader_idx].collate_fn([self.dataloaders[dataloader_idx].dataset[item_idx]])
        # send input to model, adding a batch dimension
        with torch.no_grad():
            output = model([dataset_element.x]) # no .to(device) as this is a python molGraph handled in collate_fn
            property = output.y_prime[0].item() * 10000.0
            return property

    def add_item(self, item_idx, item):
        """
        add newly predicted item at index idx
        
        :param item_idx: index into items
        :param item: item to add
        """
        self.items[item_idx].add(item)

    def finalize_items(self, dataloader_idx, start):
        """
        do final processing on a batch of predicted spectra

        :param dataloader_idx: the index of the dataloader in self.dataloaders
        :param start: position of the start of the batch
        """       
        for j in range(len(self.items)):
            self.items[j].finalize()

    def write_items(self, dataloader_idx, start):
        """
        write the spectra to files
        
        :param dataloader_idx: the index of the dataloader in self.dataloaders
        :param start: position of the start of the batch
        """
        if "arrow" in self.config.predict.output_suffixes:
            # here we want to combine the dataset.data with two new columns
            # containing the property and std dev
            # need to pass in dataset to write_items, along with start and end
            # end has to come from code in create items
            # also need the names of the new columns
            raise NotImplementedError()
            table = None
            if self.arrow is None:
                self.arrow = pa.RecordBatchFileWriter(pa.OSFile(f'{self.output_prefix}.arrow', 'wb'), table.schema)
                if self.arrow is None:
                    logging.error(f'Unable to open {self.output_prefix}.arrow')
            # self.arrow.write_table(table)
        if "csv" in self.config.predict.output_suffixes:
            table = self.dataloaders[dataloader_idx].dataset.data.to_arrow()
            null_means = pa.nulls(start, type=pa.float64())
            means_out = np.empty((len(self.items,)), dtype=np.float64)
            null_stddev = pa.nulls(start, type=pa.float64())
            stddev_out = np.empty((len(self.items,)), dtype=np.float64)
            for item_idx in range(len(self.items)):
                means_out[item_idx] = self.items[item_idx].predicted_mean
                stddev_out[item_idx] = self.items[item_idx].predicted_stddev
            table = table.append_column('predicted_ri', 
                                        pa.concat_arrays([null_means, pa.array(means_out)]))
            table = table.append_column('predicted_ri_stddev', 
                                        pa.concat_arrays([null_stddev, pa.array(stddev_out)]))
            # cut out large list fields
            table = table.drop([name for name in csv_drop_fields if name in table.column_names])
            if self.csv is None:
                self.csv = csv.CSVWriter(f"{self.output_prefix}.csv", table.schema)
                if self.csv is None:
                    logging.error(f'Unable to open {self.output_prefix}.csv')
            self.csv.write_table(table)
            # self.csv.flush()

    def __del__(self):
        # if "arrow" in self.config.predict.output_suffixes:
        #     if self.arrow is not None:
        #         self.arrow.close()
        if "csv" in self.config.predict.output_suffixes:
            if self.csv is not None:
                self.csv.close()


# for command line, need to take original compounds and tautomerize and derivatize them
# would be nice to display a graph.