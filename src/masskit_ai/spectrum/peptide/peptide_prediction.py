import copy

import numpy as np
from masskit.peptide.spectrum_generator import add_theoretical_spectra
from masskit.spectra.join import Join
from masskit.spectra.theoretical_spectrum import TheoreticalPeptideSpectrum


def upres_peptide_spectrum(predicted_spectrum, ion_types=None):
    """
    match a theoretical peptide spectrum to a predicted spectrum and copy over the theoretical mz values to
    the predicted spectrum. If there are more than one matches to a predicted spectrum, don't match

    :param predicted_spectrum: the predicted spectrum
    :param ion_types: ion types to use when generating theoretical spectra, defaults to None
    """
    theoretical_spectrum = TheoreticalPeptideSpectrum(predicted_spectrum.peptide,
                                                      ion_types=ion_types,
                                                      charge=predicted_spectrum.charge,
                                                      mod_names=predicted_spectrum.mod_names,
                                                      mod_positions=predicted_spectrum.mod_positions,
                                                      analysis_annotations=True,
                                                      )
    result = Join.join_2_spectra(predicted_spectrum, theoretical_spectrum, tiebreaker='delete')
    tolerance = predicted_spectrum.products.tolerance
    mz = predicted_spectrum.products.mz
    for i in range(len(result[0])):
        if result[0][i] is not None and result[1][i] is not None:
            theo_mz = theoretical_spectrum.products.mz[result[1][i]]
            mz[result[0][i]] = theo_mz
            tolerance[result[0][i]] =  np.full_like(theo_mz, 0.0, dtype=np.float64)
    predicted_spectrum.precursor.mz = copy.deepcopy(theoretical_spectrum.precursor.mz)
    predicted_spectrum.precursor_mass_info = copy.deepcopy(theoretical_spectrum.precursor.mass_info)
    
    
def upres_peptide_spectra(df, ion_types=None, max_mz=0, min_mz=0):
    """
    take a dataframe with predicted spectra, generate matching theoretical spectra, and upres
    matching peaks 

    :param df: list of spectra
    :param ion_types: ion types to use when generating theoretical spectra, defaults to None
    :param max_mz: maximum mz value for calculating cosine score.  0 means don't filter
    :param min_mz: the minimum mz value for calculation the cosine score
    """

    raise NotImplementedError



