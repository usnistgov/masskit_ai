# Recipes

## Spectral library generation

### Peptide library to spectral library

To generate a library of peptides, which is typically the first step in generating a peptide
spectral library, use the program `predict`. This program takes a peptide library in parquet
format and generates a spectral library using an AI network.  The peptide library can be generated using
[fasta2peptide](https://pages.nist.gov/masskit/recipes.html#protein-sequences-to-peptide-library).
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

## Predicting RI values using AIRI

The first step in prediction is to use [`batch_converter`](https://pages.nist.gov/masskit/recipes.html#library-import) to convert SDF molfiles or CSV files containing SMILES to parquet format,
which is the standard format Masskit uses for processing.

Once parquet files are generated, molecular bond path information, which is a feature used by the
AIRI model, should be calculated and added to the parquet file:
`shortest_path input.file.name=my_csv.parquet output.file.name=my_csv_path.parquet`

Finally, the AIRI predictions can be performed:
`predict --config-name config_predict_ri input.test.spectral_library=my_csv_path.parquet predict.output_prefix=my_csv_predicted predict.output_suffixes=[csv]`
The output from this command is a CSV file, which has columns that correspond to either the columns
in the original csv file or the fields in the SDF file plus some computed molecular descriptors. 
Each row corresponds to one molecular structure and has two added columns, `predicted_ri` and `predicted_ri_stddev`, which correspond to the predicted RI value as well as the standard deviation of the predicted RI.


