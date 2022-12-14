import copy

import numpy as np
import torch
from masskit.peptide.spectrum_generator import add_theoretical_spectra
from masskit.spectrum.join import Join
from masskit.spectrum.spectrum import HiResSpectrum, MassInfo, AccumulatorSpectrum
from masskit_ai.spectrum.spectrum_datasets import TandemDataframeDataset
from masskit_ai.spectrum.spectrum_lightning import SpectrumDataModule


# todo: this class was added for backward compatibility when loading jsonpickle.  remove in future version.
class PredictedSpectrum(AccumulatorSpectrum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def apply_dropout(model):
    """
    for use by torch apply to turn on dropout in a model in eval mode.

    :param model: the model
    """
    if type(model) == torch.nn.Dropout:
        model.train()


def prep_model_for_prediction(model, dropout=False, cpu=False):
    """
    prepare the model for inference

    :param model: the model
    :param dropout: should dropout be turned on?
    :param model: place model on cpu?
    """
    if cpu:
        model.cpu()
    model.eval()  # eval mode turns off training flag in all layers of model
    if dropout:  # optionally turn dropout back on
        model.model.apply(apply_dropout)


def create_prediction_dataset(model, set_to_load='test', dataloader='TandemArrowDataset', num=0, copy_annotations=True,
                              predicted_spectrum_column='predicted_spectrum', return_singleton=True):
    """
    Create pandas dataframe(s) that contains experimental spectra and can be used for predicting spectra
    each dataframe corresponds to a single validation/test/train set.

    :param dataloader: name of the dataloader class, e.g. TandemArrowDataset
    :param set_to_load: name of the set to use, e.g. "valid", "test", "train"
    :param model: the model to use to predict spectrum
    :param num: the number of spectra to predict (0 = all)
    :param copy_annotations: copy annotations and precursor from experimental spectra to predicted spectra
    :param predicted_spectrum_column: name of the column containing the predicted spectrum
    :param return_singleton: if there is only one dataframe, don't return lists
    :return: list of dataframes for doing predictions, list of dataset objects
    """
    if dataloader is not None:
        model.config.ms.dataloader = dataloader
    model.config.ms.columns = None
    mz, tolerance = create_mz_tolerance(model)

    loaders = SpectrumDataModule(model.config).create_loader(set_to_load)
    if isinstance(loaders, list):
        dfs = [x.dataset.to_pandas() for x in loaders]
        datasets = [x.dataset for x in loaders]
    else:
        dfs = [loaders.dataset.to_pandas()]
        datasets = [loaders.dataset]
    # truncate list of spectra if requested
    if num > 0:
        for i in range(len(dfs)):
            dfs[i] = dfs[i].drop(dfs[i].index[num:])
    # the final predicted spectra.  Will be a consensus of predicted_spectrum_list
    for df in dfs:
        df[predicted_spectrum_column] = [
            AccumulatorSpectrum(mz=mz, tolerance=tolerance)
            for _ in range(len(df.index))
        ]
        # the cosine score
        df["cosine_score"] = None

        # copy annotations and precursor
        if copy_annotations:
            for row in df.itertuples():
                getattr(row, predicted_spectrum_column).precursor = copy.deepcopy(row.spectrum.precursor)
                getattr(row, predicted_spectrum_column).props = copy.deepcopy(row.spectrum.props)

    if return_singleton and len(dfs) == 1:
        return dfs[0], datasets[0]
    else:
        return dfs, datasets

def create_prediction_dataset_from_hitlist(model, hitlist, experimental_tablemap, set_to_load='test', num=0, copy_annotations=False,
                              predicted_spectrum_column='predicted_spectrum', return_singleton=True,
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
    :param predicted_spectrum_column: name of the column containing the predicted spectrum
    :param return_singleton: if there is only one dataframe, don't return lists
    :return: list of dataframes for doing predictions, list of dataset objects
    """
    mz, tolerance = create_mz_tolerance(model)
    
    df = hitlist.hitlist

    # truncate list of spectra if requested
    if num > 0:
        df = df.drop(df.index[num:])
    # the final predicted spectra.  Will be a consensus of predicted_spectrum_list
    df[predicted_spectrum_column] = [
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
            getattr(row, predicted_spectrum_column).precursor = copy.deepcopy(row.spectrum.precursor)
            getattr(row, predicted_spectrum_column).props = copy.deepcopy(row.spectrum.props)
    else:
        for row in df.itertuples():
            # copy the precursor but set the props from columns
            getattr(row, predicted_spectrum_column).precursor = copy.deepcopy(row.spectrum.precursor)
            getattr(row, predicted_spectrum_column).charge = row.charge
            getattr(row, predicted_spectrum_column).mod_names = copy.deepcopy(row.mod_names)
            getattr(row, predicted_spectrum_column).mod_positions = copy.deepcopy(row.mod_positions)
            getattr(row, predicted_spectrum_column).peptide = copy.deepcopy(row.peptide)
            getattr(row, predicted_spectrum_column).peptide_len = len(row.peptide)
            getattr(row, predicted_spectrum_column).ev = row.ev
            getattr(row, predicted_spectrum_column).nce = row.nce

    
    dataset = TandemDataframeDataset(df, model.config, set_to_load)

    if return_singleton:
        return df, dataset
    else:
        return [df], [dataset]


def single_spectrum_prediction(model, dataset_element, max_intensity_in=999, cpu=False,
                               take_sqrt=False, l2norm=False):
    """
    predict a single spectrum

    :param model: the prediction model
    :param dataset_element: dataset element
    :param max_intensity_in: the intensity of the most intense peak (0=don't normalize)
    :param cpu: do computations on cpu.  Note, model should be prepared on cpu
    :param take_sqrt: square the output of the model
    :param l2norm: l2 normalize the predicted specta
    :return: the predicted spectrum
    """
    # send input to model, adding a batch dimension

    mz, tolerance = create_mz_tolerance(model)
    
    with torch.no_grad():
        if cpu:
            value = torch.unsqueeze(dataset_element.x, dim=0).cpu()
        else:
            value = torch.unsqueeze(dataset_element.x, dim=0)
        output = model(dataset_element._replace(x=value))
        intensity = output.y_prime[0, 0, :].detach().numpy()
        if take_sqrt:
            intensity = np.square(intensity)
        if max_intensity_in != 0:
            intensity *= max_intensity_in / np.max(intensity)
        spectrum = HiResSpectrum().from_arrays(
            mz,
            intensity,
            product_mass_info=MassInfo(tolerance, "daltons", "monoisotopic", evenly_spaced=True),
        )
        if l2norm:
            spectrum = spectrum.norm(max_intensity_in=1.0, ord=2)
    return spectrum


def create_mz_tolerance(model):
    """
    generate mz array and mass tolerance for model

    :param model: the model to use
    :return: mz, tolerance
    """
    tolerance = model.config.ms.bin_size / 2.0
    shift = model.config.ms.down_shift - tolerance
    mz = np.linspace(model.config.ms.bin_size + shift, model.config.ms.max_mz + shift, model.model.bins, endpoint=True)
    return mz, tolerance


def finalize_prediction_dataset(df, predicted_spectrum_column='predicted_spectrum', min_intensity=0.1, mz_window=7,
                                max_mz=0, min_mz=0):
    """
    do final processing on the predicted spectra

    :param df: dataframe containing spectra
    :param predicted_spectrum_column:  name of the predicted spectrum column
    :param min_intensity: the minimum intensity of the predicted spectra
    :param mz_window: half size of mz_window for filtering.  0 = no filtering
    :param max_mz: maximum mz value for calculating cosine score.  0 means don't filter
    :param min_mz: the minimum mz value for calculation the cosine score
    """
    for j in range(len(df.index)):
        df[predicted_spectrum_column].iat[j].finalize()
        df[predicted_spectrum_column].iat[j].props = copy.deepcopy(df["spectrum"].iat[j].props)
        df[predicted_spectrum_column].iat[j].precursor = copy.deepcopy(df["spectrum"].iat[j].precursor)
        df[predicted_spectrum_column].iat[j].filter(min_intensity=min_intensity, inplace=True)
        df[predicted_spectrum_column].iat[j].products.windowed_filter(inplace=True, mz_window=mz_window)
        df["cosine_score"].iat[j] = df["spectrum"].iat[j].cosine_score(
            df[predicted_spectrum_column].iat[j].filter(max_mz=max_mz, min_mz=min_mz), tiebreaker='mz')


def upres_peptide_spectrum(predicted_spectrum, theoretical_spectrum):
    """
    match a theoretical peptide spectrum to a predicted spectrum and copy over the theoretical mz values to
    the predicted spectrum. If there are more than one matches to a predicted spectrum, don't match

    :param predicted_spectrum: the predicted spectrum
    :param theoretical_spectrum: the theoretical spectrum
    """
    
    result = Join.join_2_spectra(predicted_spectrum, theoretical_spectrum, tiebreaker='delete')
    starts = predicted_spectrum.products.starts
    stops = predicted_spectrum.products.stops
    mz = predicted_spectrum.products.mz
    for i in range(len(result[0])):
        if result[0][i] is not None and result[1][i] is not None:
            theo_mz = theoretical_spectrum.products.mz[result[1][i]]
            mz[result[0][i]] = theo_mz
            starts[result[0][i]] =  theo_mz
            stops[result[0][i]] =  theo_mz
    predicted_spectrum.precursor.mz = copy.deepcopy(theoretical_spectrum.precursor.mz)
    predicted_spectrum.precursor_mass_info = copy.deepcopy(theoretical_spectrum.precursor.mass_info)
    
    
def upres_peptide_spectra(df, predicted_spectrum_column=None, theoretical_spectrum_column=None, ion_types=None,
                          max_mz=0, min_mz=0):
    """
    take a dataframe with predicted spectra, generate matching theoretical spectra, and upres
    matching peaks 

    :param df: the dataframe
    :param predicted_spectrum_column: name of the column containing predicted spectra, defaults to None
    :param theoretical_spectrum_column: name of the column to contain theoretical spectra, defaults to None
    :param ion_types: ion types to use when generating theoretical spectra, defaults to None
    :param max_mz: maximum mz value for calculating cosine score.  0 means don't filter
    :param min_mz: the minimum mz value for calculation the cosine score
    """
    if theoretical_spectrum_column is None:
        theoretical_spectrum_column = "theoretical_spectrum"
    if predicted_spectrum_column is None:
        predicted_spectrum_column = "predicted_spectrum"
        
    add_theoretical_spectra(df, theoretical_spectrum_column=theoretical_spectrum_column, ion_types=ion_types)
    
    for j in range(len(df.index)):
        upres_peptide_spectrum(df[predicted_spectrum_column].iat[j], df[theoretical_spectrum_column].iat[j])
        df["cosine_score"].iat[j] = df["spectrum"].iat[j].cosine_score(
            df[predicted_spectrum_column].iat[j].filter(max_mz=max_mz, min_mz=min_mz), tiebreaker='mz')


