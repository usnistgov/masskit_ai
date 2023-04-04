from masskit.peptide.spectrum_generator import add_theoretical_spectra
from masskit.spectrum.join import Join
import copy

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
    
    
def upres_peptide_spectra(df, predicted_column=None, theoretical_spectrum_column=None, ion_types=None,
                          max_mz=0, min_mz=0):
    """
    take a dataframe with predicted spectra, generate matching theoretical spectra, and upres
    matching peaks 

    :param df: the dataframe
    :param predicted_column: name of the column containing predicted spectra, defaults to None
    :param theoretical_spectrum_column: name of the column to contain theoretical spectra, defaults to None
    :param ion_types: ion types to use when generating theoretical spectra, defaults to None
    :param max_mz: maximum mz value for calculating cosine score.  0 means don't filter
    :param min_mz: the minimum mz value for calculation the cosine score
    """
    if theoretical_spectrum_column is None:
        theoretical_spectrum_column = "theoretical_spectrum"
    if predicted_column is None:
        predicted_column = "predicted_spectrum"
        
    add_theoretical_spectra(df, theoretical_spectrum_column=theoretical_spectrum_column, ion_types=ion_types)
    
    for j in range(len(df.index)):
        upres_peptide_spectrum(df[predicted_column].iat[j], df[theoretical_spectrum_column].iat[j])
        if 'spectrum' in df.columns:
            df["cosine_score"].iat[j] = df["spectrum"].iat[j].cosine_score(
                df[predicted_column].iat[j].filter(max_mz=max_mz, min_mz=min_mz), tiebreaker='mz')


