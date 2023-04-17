# Recipes

## Spectral library generation

### Peptide library to spectral library

To generate a library of peptides, which is typically the first step in generating a peptide
spectral library, use the program `predict`. This program takes a peptide library in parquet
format and generates a spectral library using an AI network.  The peptide library can be generated using
[fasta2peptide](https://pages.nist.gov/masskit/recipes.html#library-generation).
The default configuration for this program is contained in
`masskit_ai/src/masskit_ai/apps/ml/peptide/conf/config_predict.yaml`.

* to change the name of the input file, specify `input.test.spectral_library=myfilename.parquet` on the
command line.
* the prefix of the output file(s) is specified using `predict.output_prefix=myfilename` on
the command line.
  * the program outputs the following formats [msp](https://chemdata.nist.gov/dokuwiki/lib/exe/fetch.php?media=chemdata:nist17:nistms_ver23man.pdf) (NIST Text Format of Individual Spectra), [mgf](http://www.matrixscience.com/help/data_file_help.html#GEN), csv, and pkl (pickled pandas dataframe) by setting `predict.output_suffixes=[mgf,csv]`
* the program supports the following options:
  * `predict.min_intensity=0.1` is the minimum intensity to predict (out of a max of 999)
  * `predict.min_mz=28` is the minimum mz value for predicted ions
  * `predict.num=0` is the number of spectra to predict, 0 = all
  * `predict.model_ensemble=[https://github.com/usnistgov/masskit_ai/releases/download/v1.0.0/aiomics_model.tgz]` is a list of AI networks to use for prediction
  * `predict.upres=True` perform upresolution on the spectra

An example command line: `predict input.test.spectral_library=uniprot_peptides.parquet predict.output_prefix=uniprot_peptides predict.output_suffixes=[mgf,msp]`
