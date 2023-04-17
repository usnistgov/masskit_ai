import copy
import numpy as np
import torch
from masskit.spectrum.spectrum import HiResSpectrum, MassInfo, AccumulatorSpectrum
from masskit.peptide.spectrum_generator import create_peptide_name
from masskit_ai.prediction import Predictor
from masskit_ai.spectrum.peptide.peptide_prediction import upres_peptide_spectrum
from masskit_ai.spectrum.spectrum_datasets import TandemDataframeDataset
from masskit_ai.lightning import MasskitDataModule
from masskit.peptide.encoding import calc_precursor_mz
from masskit_ai import _device
import pyarrow as pa
from masskit.utils.files import spectra_to_array, spectra_to_msp, spectra_to_mgf

class PeptideSpectrumPredictor(Predictor):

    def __init__(self, config=None, row_batch_size=50000, device=None, *args, **kwargs):
        super().__init__(config=config, row_batch_size=row_batch_size, device=device, *args, **kwargs)
        self.max_mz=None
        self.mz = None
        self.tolerance = None
        self.max_intensity = self.config.predict.get("max_intensity", 999.0)
        if "arrow" in self.config.predict.output_suffixes:
            self.arrow = pa.OSFile(f'{config.predict.output_prefix}.arrow', 'wb')
        if "msp" in self.config.predict.output_suffixes:
            self.msp = open(f"{config.predict.output_prefix}.msp", "w")
        if "mgf" in self.config.predict.output_suffixes:
            self.mgf = open(f"{config.predict.output_prefix}.mgf", "w")

    def create_dataloaders(self, model):
        """
        Create dataloaders that contains experimental spectra.

        :param model: the model to use to predict spectrum
        :return: list of dataloader objects
        """
        model.config.ms.dataloader = self.config.predict.dataloader
        model.config.ms.columns = None
        self.mz, self.tolerance = self.create_mz_tolerance(model)
        self.max_mz = model.config.ms.get("max_mz", 2000)

        loaders = MasskitDataModule(model.config).create_loader(self.config.predict.set_to_load)

        if isinstance(loaders, list):
            datasets = loaders
        else:
            datasets = [loaders]

        return datasets
    
    def get_items(self, loader, start):
        """
        for a given loader, return back a batch of accumulators

        :param loader: the input loader, assume to contain a TableMap
        :param start: the start row of the batch
        :param length: the length of the batch
        """
        spectra = []
        table_map = loader.dataset.data

        if self.config.predict.num is not None and self.config.predict.num > 0:
            end = min(start+self.row_batch_size, self.config.predict.num, len(table_map))
        else:
            end = min(start+self.row_batch_size, len(table_map))

        for i in range(start, end):
            row = table_map.getitem_by_row(i)
            # we are assuming a peptide spectrum here.  To generalize, this needs to be put into the spectrum class.
            precursor_mz = calc_precursor_mz(row['peptide'], row['charge'], mod_names=row["mod_names"], mod_positions=row["mod_positions"])
            new_spectrum = AccumulatorSpectrum(mz=self.mz, tolerance=self.tolerance, precursor_mz=precursor_mz)
            new_spectrum.copy_props_from_dict(row)
            new_spectrum.name = create_peptide_name(row['peptide'], row['charge'], row['mod_names'], row['mod_positions'], row.get('nce', None))
            spectra.append(new_spectrum)

        return spectra
        
    def single_prediction(self, model, dataset_element):
        """
        predict a single spectrum

        :param model: the prediction model
        :param dataset_element: dataset element
        :return: the predicted spectrum
        """
        # send input to model, adding a batch dimension
        take_sqrt=self.config.ms.get('take_sqrt', False)
        l2norm=self.config.predict.get('l2norm', False)
        
        with torch.no_grad():
            output = model([torch.unsqueeze(dataset_element.x, 0).to(device=_device)])
            intensity = output.y_prime[0, 0, :].detach().cpu().numpy()
            if take_sqrt:
                intensity = np.square(intensity)
            if self.max_intensity != 0:
                intensity *= self.max_intensity / np.max(intensity)
            spectrum = HiResSpectrum().from_arrays(
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

    def finalize_items(self, items, dataset, start):
        """
        do final processing on a batch of predicted spectra

        :param items: ListLike of spectra
        :param dataset: dataset containing experimental spectra
        :param start: position of the start of the batch
        """
        min_intensity=self.config.predict.get('min_intensity', 0), 
        mz_window=self.config.predict.get('mz_window',7),
        min_mz=self.config.predict.get('min_mz', 0),
        upres=self.config.predict.get("upres", False)
        
        for j in range(len(items)):
            items[j].finalize()
            items[j].filter(min_intensity=min_intensity, inplace=True)
            items[j].products.windowed_filter(inplace=True, mz_window=mz_window)
            if upres:
                upres_peptide_spectrum(items[j])

            row = dataset.dataset.data.getitem_by_row(start + j)
            if 'spectrum' in row and row['spectrum'] is not None and row['spectrum'].products.mz is not None:
                items[j].cosine_score = items[j].cosine_score(
                    row['spectrum'].filter(max_mz=self.max_mz, min_mz=min_mz), tiebreaker='mz')
    
    def write_items(self, items):
        """
        write the spectra to files
        
        :param items: the spectra to write
        """
        if "arrow" in self.config.predict.output_suffixes:
            table = spectra_to_array(items, write_starts_stops=self.config.predict.get("upres", False))
            writer = pa.RecordBatchFileWriter(self.arrow, table.schema)
            writer.write_table(table)
        if "msp" in self.config.predict.output_suffixes:
            spectra_to_msp(self.msp, items, annotate=True)
            self.msp.flush()
        if "mgf" in self.config.predict.output_suffixes:
            spectra_to_mgf(self.mgf, items)
            self.mgf.flush()


def create_prediction_dataset_from_hitlist(model, hitlist, experimental_tablemap, set_to_load='test', num=0, copy_annotations=False,
                              predicted_column='predicted_spectrum', return_singleton=True, **kwargs
                              ):
    """
    Create pandas dataframe(s) that contains experimental spectra and can be used for predicting spectra
    each dataframe corresponds to a single validation/test/train set.

    :param model: the model to use to predict spectrum
    :param set_to_load: name of the set to use, e.g. "valid", "test", "train"
    :param hitlist: the Hitlist object
    :param experimental_spectra: TableMap containing the experimental spectra, used to get eV
    :param num: the number of spectra to predict (0 = all)
    :param copy_annotations: copy annotations and precursor from experimental spectra to predicted spectra
    :param predicted_column: name of the column containing the predicted spectrum
    :param return_singleton: if there is only one dataframe, don't return lists
    :return: list of dataframes for doing predictions, list of dataset objects
    """
    mz, tolerance = create_mz_tolerance(model)
    
    df = hitlist.hitlist

    # truncate list of spectra if requested
    if num > 0:
        df = df.drop(df.index[num:])
    # the final predicted spectra.  Will be a consensus of predicted_spectrum_list
    df[predicted_column] = [
        AccumulatorSpectrum(mz=mz, tolerance=tolerance)
        for _ in range(len(df.index))
        ]
    # the cosine score
    df["cosine_score"] = None
    df['ev'] = [experimental_tablemap.getitem_by_id(id)['ev'] for id in df.index.get_level_values(0)]
    df['nce'] = [experimental_tablemap.getitem_by_id(id)['nce'] for id in df.index.get_level_values(0)]
    df['spectrum'] = [experimental_tablemap.getspectrum_by_id(id) for id in df.index.get_level_values(0)]

        # copy annotations and precursor
        # change to use tablemap and insert experimental spectrum
    if copy_annotations:
        for row in df.itertuples():
            getattr(row, predicted_column).precursor = copy.deepcopy(row.spectrum.precursor)
            getattr(row, predicted_column).props = copy.deepcopy(row.spectrum.props)
    else:
        for row in df.itertuples():
            # copy the precursor but set the props from columns
            getattr(row, predicted_column).precursor = copy.deepcopy(row.spectrum.precursor)
            getattr(row, predicted_column).charge = row.charge
            getattr(row, predicted_column).mod_names = copy.deepcopy(row.mod_names)
            getattr(row, predicted_column).mod_positions = copy.deepcopy(row.mod_positions)
            getattr(row, predicted_column).peptide = copy.deepcopy(row.peptide)
            getattr(row, predicted_column).peptide_len = len(row.peptide)
            getattr(row, predicted_column).ev = row.ev
            getattr(row, predicted_column).nce = row.nce

    
    dataset = TandemDataframeDataset(df, model.config, set_to_load)

    if return_singleton:
        return df, dataset
    else:
        return [df], [dataset]


