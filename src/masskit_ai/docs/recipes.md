# Recipes

## Spectral library generation

### Peptide library to spectral library

To generate a library of peptides, which is typically the first step in generating a peptide
spectral library, use the program `predict.py`. This program takes a peptide library in parquet
format and generates a spectral library using an AI network.  The peptide library can be generated using
[fasta2peptide.py](https://pages.nist.gov/masskit/recipes.html#library-generation).
The configuration for this program is contained in the `conf/config_predict_peptide_digest.yaml`
file in the same directory as `predict.py`.

* to change the name of the input file, specify `input.test.spectral_library=myfilename.parquet` on the
command line.
* the prefix of the output file(s) is specified using `output_prefix=myfilename` on
the command line.
  * the program outputs the following formats [msp](https://chemdata.nist.gov/dokuwiki/lib/exe/fetch.php?media=chemdata:nist17:nistms_ver23man.pdf) (NIST Text Format of Individual Spectra), [mgf](http://www.matrixscience.com/help/data_file_help.html#GEN), csv, and pkl (pickled pandas dataframe) by setting `output_suffixes=[mgf,csv]`
* the program supports the following options:
  * `min_intensity=0.1` is the minimum intensity to predict (out of a max of 999)
  * `min_mz=28` is the minimum mz value for predicted ions
  * `num=0` is the number of spectra to predict, 0 = all
  * `model_ensemble=[lyg_rebuttal_166c872328264874a818bd1f4ad178ac_val_loss=-0.8471_epoch=043.ckpt]` is a list of AI networks to use for prediction
  * `upres: True` perform upresolution on the spectra
