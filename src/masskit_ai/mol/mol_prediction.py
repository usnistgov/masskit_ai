import torch
from masskit.accumulator import AccumulatorProperty
from masskit_ai.spectrum.spectrum_lightning import SpectrumDataModule
from masskit_ai.lightning import setup_datamodule


def create_molprop_prediction_dataset(model, set_to_load='test', dataloader='MolPropDataset', num=0,
                              predicted_column='predicted_property', return_singleton=True, **kwargs):
    """
    Create pandas dataframe(s) that contains mols and can predict a property.

    :param dataloader: name of the dataloader class, e.g. TandemArrowDataset
    :param set_to_load: name of the set to use, e.g. "valid", "test", "train"
    :param model: the model to use to predict spectrum
    :param num: the number of spectra to predict (0 = all)
    :param predicted_column: name of the column containing the predicted spectrum
    :param return_singleton: if there is only one dataframe, don't return lists
    :return: list of dataframes for doing predictions, list of dataset objects
    """
    if dataloader is not None:
        model.config.ms.dataloader = dataloader
    # load in all columns
    model.config.ms.columns = None

    loaders = setup_datamodule(model.config).create_loader(set_to_load)

    if isinstance(loaders, list):
        dfs = [x.dataset.to_pandas() for x in loaders]
        datasets = loaders
    else:
        dfs = [loaders.dataset.to_pandas()]
        datasets = [loaders]
    # truncate list if requested
    if num > 0:
        for i in range(len(dfs)):
            dfs[i] = dfs[i].drop(dfs[i].index[num:])
    # the final predicted spectra.  Will be a consensus of predicted_spectrum_list
    for df in dfs:
        df[predicted_column] = [
            AccumulatorProperty()
            for _ in range(len(df.index))
        ]
        # standard deviation column
        df[predicted_column + "_stddev"] = None

    if return_singleton and len(dfs) == 1:
        return dfs[0], datasets[0]
    else:
        return dfs, datasets


def single_molprop_prediction(model, dataset_element, **kwargs):
    """
    predict a single spectrum

    :param model: the prediction model
    :param dataset_element: dataset element
    :return: the predicted spectrum
    """
    
    with torch.no_grad():
        output = model([dataset_element.x])
        property = output.y_prime[0].item() * 10000.0
    return property


def finalize_molprop_prediction_dataset(df, predicted_column='predicted_property', **kwargs):
    """
    do final processing on the predicted spectra

    :param df: dataframe containing spectra
    :param predicted_column:  name of the predicted spectrum column
    """
    for j in range(len(df.index)):
        accumulator = df[predicted_column].iat[j]
        accumulator.finalize()
        df[predicted_column].iat[j] = accumulator.mean
        df[predicted_column + '_stddev'].iat[j] = accumulator.stddev
    df[predicted_column] = df[predicted_column].astype(float)
