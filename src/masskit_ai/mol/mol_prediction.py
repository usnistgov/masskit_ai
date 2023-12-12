import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from masskit.utils.accumulator import AccumulatorProperty
from masskit.data_specs.file_schemas import csv_drop_fields
from masskit.utils.general import class_for_name
from pyarrow import csv as pacsv

from masskit_ai import _device
from masskit_ai.base_objects import ModelInput
from masskit_ai.prediction import Predictor


class MolPropPredictor(Predictor):
    """
    class used to predict multiple spectra per record, which are averaged into
    a single spectrum with a standard deviation
    """

    def __init__(self, config=None, *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
        self.items = []  # the items to predict
        # the accumulator class
        self.accumulator = class_for_name(config.paths.modules.prediction,
                                          config.predict.get("accumulator", 
                                                             "AccumulatorProperty"))

        self.output_prefix = str(Path(config.predict.output_prefix).expanduser())
        if "csv" in self.config.predict.output_suffixes:
            self.csv = None
        if "arrow" in self.config.predict.output_suffixes:
            self.arrow = None
        if "parquet" in self.config.predict.output_suffixes:
            self.parquet = None

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
            self.items.append(self.accumulator()) 

        return self.items
        
    def single_prediction(self, model, item_idx, dataloader_idx):
        """
        predict a single spectrum

        :param model: the prediction model
        :param item_idx: the index of item in the current dataset
        :param dataloader_idx: the index of the dataloader in self.dataloaders
        :return: the predicted spectrum
        """
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

        # don't use __item__ to avoid trying to load y target values, which may not be available
        data_row = self.dataloaders[dataloader_idx].dataset.get_data_row(item_idx)
        input = ModelInput(x=self.dataloaders[dataloader_idx].dataset.get_x(data_row), y=None, index=item_idx)
        dataset_element = self.dataloaders[dataloader_idx].collate_fn([input])
        # send input to model, adding a batch dimension
        with torch.no_grad():
            output = model([dataset_element.x]) # no .to(device) as this is a python molGraph handled in collate_fn
            property = output.y_prime[0].item() * self.config.ms.normalization
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

        # combine the dataset.data with two new columns
        # containing the property and std dev
        table = self.dataloaders[dataloader_idx].dataset.data.to_arrow().slice(start, self.row_group_size)
        means_out = np.empty((len(self.items,)), dtype=np.float64)
        stddev_out = np.empty((len(self.items,)), dtype=np.float64)
        for item_idx in range(len(self.items)):
            means_out[item_idx] = self.items[item_idx].predicted_mean
            stddev_out[item_idx] = self.items[item_idx].predicted_stddev
        table = table.append_column('predicted_ri', pa.array(means_out))
        table = table.append_column('predicted_ri_stddev', pa.array(stddev_out))
        if isinstance(self.items[item_idx], AccumulatorAIRI):
            confidence_interval = np.empty((len(self.items,)), dtype=np.float64)
            for item_idx in range(len(self.items)):
                confidence_interval[item_idx] = self.items[item_idx].confidence_interval
            table = table.append_column('confidence_interval', pa.array(confidence_interval))
        if "arrow" in self.config.predict.output_suffixes:
            if self.arrow is None:
                self.arrow = pa.RecordBatchFileWriter(pa.OSFile(f'{self.output_prefix}.arrow', 'wb'),
                                                      table.schema)
                if self.arrow is None:
                    logging.error(f'Unable to open {self.output_prefix}.arrow')
            self.arrow.write_table(table)
        if "parquet" in self.config.predict.output_suffixes:
            if self.parquet is None:
                self.parquet = pa.parquet.ParquetWriter(pa.OSFile(f'{self.output_prefix}.parquet', 'wb'),
                                                      table.schema)
                if self.parquet is None:
                    logging.error(f'Unable to open {self.output_prefix}.parquet')
            self.parquet.write_table(table)
        if "csv" in self.config.predict.output_suffixes:
            # delete completely null columns
            table = table.drop([table.column_names[i] for i in range(table.num_columns)
                                if table.column(i).null_count == len(table)])
            # cut out large list fields
            table = table.drop([name for name in csv_drop_fields if name in table.column_names])
            if self.csv is None:
                self.csv = pacsv.CSVWriter(f"{self.output_prefix}.csv", table.schema)
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


class AccumulatorAIRI(AccumulatorProperty):
    """
    accumulator with the AIRI predicted standard deviation correction from Geer et al. 2024
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corrections = [
            3.3025433724785307,
            2.106851096543382,
            1.6125795977955588,
            1.3811891225110602,
            1.257337617482262,
            1.198053135796777,
            1.1555841253757473,
            1.0978786961593128,
            1.083984534298343,
            1.0643445858640375,
            1.0537509166216747,
            1.0109826383137996,
            0.9930032511483983,
            0.991049752547334,
            0.9796592787796974, 
            0.9475810609426271, 
            0.9209559852985938, 
            1.0298742018601368, 
            0.8491898961202649, 
            0.919136405411796, 
            1.028022270964323, 
            0.9525348786923363, 
            0.9655304623288548, 
            0.961960139659409, 
            1.0352858813567667, 
            0.9197595302255215, 
            1.0286739978644306, 
            1.1100359022427024, 
            1.0283136145494205, 
            1.0760922588908612, 
            1.0992211363911804, 
            0.9113198257028539, 
            1.1407931723731588, 
            1.2338976143312628, 
            1.2295029908697706, 
            1.3243341631437622, 
            1.3660884050112316
            ]
        self.correction_step = 2
        # 95th percentile confidence interval
        self.confidence_interval = None


    def finalize(self):
        """
        finalize the std deviation after all the the spectra have been added, 
        then apply a correction. Corrections generated from analysis in analyze_new_model.ipynb
        """
        super().finalize()
        i = int(max(0, min(len(self.corrections)-1, self.predicted_stddev // self.correction_step)))
        self.predicted_stddev *= self.corrections[i]
        self.confidence_interval = self.predicted_stddev * 2.04635  # from analyze_new_model.ipynb

