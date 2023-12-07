# Recipes

## Spectral library generation

### Peptide library to spectral library

To generate a library of peptides, which is typically the first step in generating a peptide
spectral library, use the program `predict`. This program takes a peptide library in parquet
format and generates a spectral library using an AI network.  The peptide library can be generated using
[fasta2peptide](https://pages.nist.gov/masskit/recipes.html#protein-sequences-to-peptide-library).
The default configuration for the `predict` program is contained in
`masskit_ai/src/masskit_ai/apps/ml/peptide/conf/config_predict.yaml`.

* to change the name of the input file, specify `input.test.spectral_library=myfilename.parquet` on the
command line.
* the prefix of the output file(s) is specified using `predict.output_prefix=myfilename` on
the command line.
  * the program outputs the following formats [msp](https://chemdata.nist.gov/dokuwiki/lib/exe/fetch.php?media=chemdata:nist17:nistms_ver23man.pdf) (NIST Text Format of Individual Spectra), [mgf](http://www.matrixscience.com/help/data_file_help.html#GEN), and [arrow](https://arrow.apache.org/docs/python/feather.html) by setting `predict.output_suffixes=[mgf,csv]`
* the program supports the following options:
  * `predict.min_intensity=0.1` is the minimum intensity to predict (out of a max of 999)
  * `predict.min_mz=28` is the minimum mz value for predicted ions
  * `predict.num=0` is the number of spectra to predict, 0 = all
  * `predict.model_ensemble=[https://github.com/usnistgov/masskit_ai/releases/download/v1.2.0/aiomics_model.tgz]` is a list of AI networks to use for prediction
  * `predict.upres=True` perform upresolution on the spectra

To get additional help on options for these programs, run the program using the `-h` option.

#### Example set of commands to predict spectra from a fasta file `uniprot.fasta`

```bash
fasta2peptides input.file=uniprot.fasta output.file=uniprot_peptides.parquet
predict input.test.spectral_library=uniprot_peptides.parquet predict.output_prefix=uniprot_peptides predict.output_suffixes=[mgf,msp]
```

The predicted spectra are found in the files `uniprot_peptides.msp` and `uniprot_peptides.mgf`.

## Predicting RI values using AIRI

The first step in prediction is to use [`batch_converter`](https://pages.nist.gov/masskit/recipes.html#library-import) to convert SDF molfiles or CSV files containing SMILES to parquet format,
which is the standard format Masskit uses for processing.
When reading from an sdf file, you can specify the field used as the ID by setting
conversion.sdf.id.field to the field name, e.g. `conversion.sdf.id.field=NISTNO`.

The program `reactor` can be optionally used to derivatize and generate tautomers of the original structures.

Once parquet files are generated, molecular bond path information, which is a feature used by the
AIRI model, should be calculated and added to the parquet file using the program `shortest_path`.

Finally, the AIRI predictions can be performed using the `predict` command line.
The output from this command is a CSV file, which has columns that correspond to either the columns
in the original csv file or the fields in the SDF file plus some computed molecular descriptors.
Each row corresponds to one molecular structure and has three added columns, `predicted_ri`, `predicted_ri_stddev` and `predicted_ri_stddev_clip`, which correspond to the predicted RI value as well as the standard deviation of the predicted RI and the standard deviation clipped at a lower bound
to generate a more normal distribution of RI values.

### Example set of commands to calculate AIRI values from a CSV file `my_csv.csv` with SMILES in the `molecules` column

```bash
batch_converter input.file.names=my_csv.csv output.file.name=my_csv output.file.types=[parquet] conversion.csv.smiles_column_name=molecules
reactor input.file.name=my_csv.parquet output.file.name=my_csv_derivatized.parquet conversion.num_tautomers=5 conversion.mass_range=[0,5000] conversion.reactant_names=[trimethylsilylation] 
shortest_path input.file.name=my_csv_derivatized.parquet output.file.name=my_csv_path.parquet
predict --config-name config_predict_ri input.test.spectral_library=my_csv_path.parquet predict.output_prefix=my_csv_predicted predict.output_suffixes=[csv]
```

The AIRI values are found in the file `my_csv_predicted.csv`. The use of `reactor` to derivatize molecules is optional.
If `reactor` is removed from the list of commands, use the output from `batch_converter`
as the input file to `shortest_path`, e.g.`shortest_path input.file.name=my_csv.parquet output.file.name=my_csv_path.parquet`.
To get additional help on options for these programs, run the program using the `-h` option.

### Example set of commands to calculate AIRI values from an SDF molfile `my_sdf.sdf`

```bash
batch_converter input.file.names=my_sdf.sdf output.file.name=my_sdf output.file.types=[parquet]
reactor input.file.name=my_sdf.parquet output.file.name=my_csv_derivatized.parquet conversion.num_tautomers=5 conversion.mass_range=[0,5000] conversion.reactant_names=[trimethylsilylation]
shortest_path input.file.name=my_sdf.parquet output.file.name=my_sdf_path.parquet
predict --config-name config_predict_ri input.test.spectral_library=my_sdf_path.parquet predict.output_prefix=my_sdf_predicted predict.output_suffixes=[csv]
```

The AIRI values are found in the file `my_sdf_predicted.csv`

If the SDF file includes latin-1 encoded characters or is a pre v2000 version SDF file,
use the program `rewrite_sdf` to create a corrected SDF file:

```bash
rewrite_sdf input.file.name=my_orginal_sdf.sdf output.file.name=my_sdf.sdf
```



