import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from masskit.peptide.encoding import calc_precursor_mz
from masskit.peptide.spectrum_generator import create_peptide_name
from masskit.spectra import AccumulatorSpectrum, Spectrum
from masskit.spectra.ions import MassInfo
from masskit.utils.files import spectra_to_array
from masskit.utils.spectrum_writers import spectra_to_mgf, spectra_to_msp

from masskit_ai import _device
from masskit_ai.base_objects import ModelInput
from masskit_ai.prediction import Predictor
from masskit_ai.spectrum.peptide.peptide_prediction import \
    upres_peptide_spectrum


def finalize_spectrum(spectrum, min_intensity, mz_window, upres=False):
    """
    function to finalize a predicted spectrum.  Separated from the class
    so can be used in multiprocessing

    :param spectrum: spectrum to be finalized
    :param min_intensity: minimum peak intensity for filtering
    :param mz_window: size of window for filtering
    :param upres: upres the resulting spectrum
    :return: the finalized spectrum
    """
    spectrum.finalize()
    spectrum.filter(min_intensity=min_intensity, inplace=True)
    spectrum.products.windowed_filter(inplace=True, mz_window=mz_window)
    if upres:
        upres_peptide_spectrum(spectrum)
    return spectrum

class PeptideSpectrumPredictor(Predictor):
    """
    class used to predict multiple spectra per record, which are averaged into
    a single spectrum with a standard deviation
    """

    def __init__(self, config=None, *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
        self.max_mz=None
        self.mz = None
        self.tolerance = None
        self.items = []  # the items to predict
        self.output_prefix = str(Path(config.predict.output_prefix).expanduser())
        self.max_intensity = self.config.predict.get("max_intensity", 999.0)
        if "arrow" in self.config.predict.output_suffixes:
            self.arrow = None
        if "msp" in self.config.predict.output_suffixes:
            self.msp = open(f"{self.output_prefix}.msp", "w")
            if self.msp is None:
                logging.error(f'Unable to open {self.output_prefix}.msp')
        if "mgf" in self.config.predict.output_suffixes:
            self.mgf = open(f"{self.output_prefix}.mgf", "w")
            if self.mgf is None:
                logging.error(f'Unable to open {self.output_prefix}.mgf')

    def create_dataloaders(self, model):
        """
        Create dataloaders that contains experimental spectra.

        :param model: the model to use to predict spectrum
        :return: list of dataloader objects
        """
        self.mz, self.tolerance = self.create_mz_tolerance(model)
        self.max_mz = model.config.ms.get("max_mz", 2000)

        return super().create_dataloaders(model)

    def make_spectrum(self, precursor_mz):
        return AccumulatorSpectrum(mz=self.mz, tolerance=self.tolerance, precursor_mz=precursor_mz)

    def create_items(self, dataloader_idx, start):
        """
        for a given loader, return back a batch of accumulators

        :param dataloader_idx: the index of the dataloader in self.dataloaders
        :param start: the start row of the batch
        """
        self.items = []
        table_map = self.dataloaders[dataloader_idx].dataset.data

        if self.config.predict.num is not None and self.config.predict.num > 0:
            end = min(start+self.row_group_size, self.config.predict.num + self.original_start, len(table_map))
        else:
            end = min(start+self.row_group_size, len(table_map))

        for j in range(start, end):
            row = table_map.getitem_by_row(j)
            # we are assuming a peptide spectrum here.  To generalize, this needs to be put into the spectrum class.
            precursor_mz = calc_precursor_mz(row['peptide'], row['charge'], mod_names=row["mod_names"],
                                             mod_positions=row["mod_positions"])
            new_spectrum = self.make_spectrum(precursor_mz)
            new_spectrum.copy_props_from_dict(row)
            new_spectrum.name = create_peptide_name(row['peptide'], row['charge'], row['mod_names'],
                                                    row['mod_positions'], row.get('nce', None))
            self.items.append(new_spectrum)

        return self.items
        
    def single_prediction(self, model, item_idx, dataloader_idx):
        """
        predict a single spectrum

        :param model: the prediction model
        :param item_idx: the index of item in the current dataset
        :param dataloader_idx: the index of the dataloader in self.dataloaders
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

        take_sqrt=self.config.ms.get('take_sqrt', False)
        l2norm=self.config.predict.get('l2norm', False)
        # don't use __item__ to avoid trying to load y target values, which may not be available
        data_row = self.dataloaders[dataloader_idx].dataset.get_data_row(item_idx)
        input = ModelInput(x=self.dataloaders[dataloader_idx].dataset.get_x(data_row), y={}, index=item_idx)
        dataset_element = self.dataloaders[dataloader_idx].collate_fn([input])

        with torch.no_grad():
            # send input to model, adding a batch dimension
            output = model(torch.unsqueeze(dataset_element.x, 0).to(device=_device))
            intensity = output.y_prime[0, 0, :].detach().cpu().numpy()
            if take_sqrt:
                intensity = np.square(intensity)
            if self.max_intensity != 0:
                intensity *= self.max_intensity / np.max(intensity)
            spectrum = Spectrum().from_arrays(
                self.mz,
                intensity,
                product_mass_info=MassInfo(self.tolerance, "daltons", "monoisotopic", evenly_spaced=True),
            )
            if l2norm:
                spectrum = spectrum.norm(max_intensity_in=1.0, ord=2)
        return spectrum

    def create_mz_tolerance(self, model):
        """
        generate mz array and mass tolerance for model

        :param model: the model to use
        :return: mz, tolerance
        """
        tolerance = model.config.ms.bin_size / 2.0
        shift = model.config.ms.down_shift - tolerance
        mz = np.linspace(model.config.ms.bin_size + shift, model.config.ms.max_mz + shift, model.model.bins, endpoint=True)
        return mz, tolerance

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
        min_intensity=self.config.predict.get('min_intensity', 0), 
        mz_window=self.config.predict.get('mz_window',7),
        min_mz=self.config.predict.get('min_mz', 0),
        upres=self.config.predict.get("upres", False)
        
        with Pool(self.config.predict.get('num_workers', 2)) as p:
            self.items = p.map(partial(finalize_spectrum, min_intensity=min_intensity, mz_window=mz_window,
                                       upres=upres), self.items)
        for j in range(len(self.items)):
            row = self.dataloaders[dataloader_idx].dataset.data.getitem_by_row(start + j)
            if 'spectrum' in row and row['spectrum'] is not None and row['spectrum'].products.mz is not None:
                self.items[j].cosine_score = self.items[j].cosine_score(
                    row['spectrum'].filter(max_mz=self.max_mz, min_mz=min_mz), tiebreaker='mz')
    
    def write_items(self, dataloader_idx, start):
        """
        write the spectra to files
        
        :param dataloader_idx: the index of the dataloader in self.dataloaders
        :param start: position of the start of the batch
        """
        if "arrow" in self.config.predict.output_suffixes:
            table = spectra_to_array(self.items, write_tolerance=self.config.predict.get("upres", False))
            if self.arrow is None:
                self.arrow = pa.RecordBatchFileWriter(pa.OSFile(f'{self.output_prefix}.arrow', 'wb'), table.schema)
                if self.arrow is None:
                    logging.error(f'Unable to open {self.output_prefix}.arrow')
            self.arrow.write_table(table)
        if "msp" in self.config.predict.output_suffixes and self.msp is not None:
            spectra_to_msp(self.msp, self.items, annotate_peptide=self.config.predict.get('annotate', False))
            self.msp.flush()
        if "mgf" in self.config.predict.output_suffixes and self.mgf is not None:
            spectra_to_mgf(self.mgf, self.items)
            self.mgf.flush()

    def __del__(self):
        if "arrow" in self.config.predict.output_suffixes:
            if self.arrow is not None:
                self.arrow.close()
        if "msp" in self.config.predict.output_suffixes:
            if self.msp is not None:
                self.msp.close()
        if "mgf" in self.config.predict.output_suffixes:
            if self.mgf is not None:
                self.mgf.close()


class SinglePeptideSpectrumPredictor(PeptideSpectrumPredictor):
    """
    class used to predict a single spectrum per records
    """
    def __init__(self, config=None, *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
    
    def make_spectrum(self, precursor_mz):
        spectrum = Spectrum()
        spectrum.from_arrays(self.mz, np.zeros_like(self.mz), 
                        product_mass_info=MassInfo(self.tolerance, "daltons", "monoisotopic", evenly_spaced=True), 
                        precursor_mz=precursor_mz, precursor_intensity=999.0,
                        precursor_mass_info=MassInfo(0.0, "ppm", "monoisotopic"))
        return spectrum

    def add_item(self, idx, item):
        self.items[idx].products.mz = item.products.mz
        self.items[idx].products.intensity = item.products.intensity     


# def create_prediction_dataset_from_hitlist(model, hitlist, experimental_tablemap, set_to_load='test', num=0, copy_annotations=False,
#                               predicted_column='predicted_spectrum', return_singleton=True, **kwargs
#                               ):
#     """
#     Create pandas dataframe(s) that contains experimental spectra and can be used for predicting spectra
#     each dataframe corresponds to a single validation/test/train set.

#     :param model: the model to use to predict spectrum
#     :param set_to_load: name of the set to use, e.g. "valid", "test", "train"
#     :param hitlist: the Hitlist object
#     :param experimental_spectra: TableMap containing the experimental spectra, used to get eV
#     :param num: the number of spectra to predict (0 = all)
#     :param copy_annotations: copy annotations and precursor from experimental spectra to predicted spectra
#     :param predicted_column: name of the column containing the predicted spectrum
#     :param return_singleton: if there is only one dataframe, don't return lists
#     :return: list of dataframes for doing predictions, list of dataset objects
#     """
#     mz, tolerance = create_mz_tolerance(model)
    
#     df = hitlist.hitlist

#     # truncate list of spectra if requested
#     if num > 0:
#         df = df.drop(df.index[num:])
#     # the final predicted spectra.  Will be a consensus of predicted_spectrum_list
#     df[predicted_column] = [
#         AccumulatorSpectrum(mz=mz, tolerance=tolerance)
#         for _ in range(len(df.index))
#         ]
#     # the cosine score
#     df["cosine_score"] = None
#     df['ev'] = [experimental_tablemap.getitem_by_id(id)['ev'] for id in df.index.get_level_values(0)]
#     df['nce'] = [experimental_tablemap.getitem_by_id(id)['nce'] for id in df.index.get_level_values(0)]
#     df['spectrum'] = [experimental_tablemap.getitem_by_id(id)['spectrum'] for id in df.index.get_level_values(0)]

#         # copy annotations and precursor
#         # change to use tablemap and insert experimental spectrum
#     if copy_annotations:
#         for row in df.itertuples():
#             getattr(row, predicted_column).precursor = copy.deepcopy(row.spectrum.precursor)
#             getattr(row, predicted_column).props = copy.deepcopy(row.spectrum.props)
#     else:
#         for row in df.itertuples():
#             # copy the precursor but set the props from columns
#             getattr(row, predicted_column).precursor = copy.deepcopy(row.spectrum.precursor)
#             getattr(row, predicted_column).charge = row.charge
#             getattr(row, predicted_column).mod_names = copy.deepcopy(row.mod_names)
#             getattr(row, predicted_column).mod_positions = copy.deepcopy(row.mod_positions)
#             getattr(row, predicted_column).peptide = copy.deepcopy(row.peptide)
#             getattr(row, predicted_column).peptide_len = len(row.peptide)
#             getattr(row, predicted_column).ev = row.ev
#             getattr(row, predicted_column).nce = row.nce

    
#     dataset = TandemDataframeDataset(df, model.config, set_to_load)

#     if return_singleton:
#         return df, dataset
#     else:
#         return [df], [dataset]
