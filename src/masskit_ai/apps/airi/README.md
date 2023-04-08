# Command line for calculating AIRI retention index values
## Installation
1. install the python programming language if it is not already installed on your computer, preferably via downloading and installing [anaconda](https://www.anaconda.com/distribution/#download-section)
or the smaller [miniconda](https://docs.conda.io/en/latest/miniconda.html).  The version of python should be 3.7 or later, 
which can be checked via the command line `python --version`
1. install the command line program [git](https://git-scm.com/downloads) if it is not already installed on your computer.  
1. make sure you are connected to the NIST network
1. in a command window (e.g. the Anaconda Prompt if using anaconda, `cmd`, or Mac Terminal),
 run `pip install git+https://gitlab.nist.gov/gitlab/msdc/cheminformatics/airi.git` to run the installation.
## Upgrading the package
To upgrade to a newer version of airi_calc, run `pip install git+https://gitlab.nist.gov/gitlab/msdc/cheminformatics/airi.git --upgrade`
while connected to the NIST network.
## Running the script from a command window
You must be connected to the NIST network. The input to the program can be SMILES on the command line or tsv/csv/excel files.
 The output can be tsv/csv/excel format.  The format for the files is deduced from their extension, for example `csv` in `test.csv`.
### SMILES on the command line
`airi_calc --smiles CCCC c1ccccc1`
### SMILES in an excel spreadsheet
`airi_calc --in_file test_in.xlsx --out_file test_out.xlsx --sheet_name Sheet3 --smiles_column_name "Structure(SMILES)"`
### SMILES in a tsv file
`airi_calc --in_file test_in.tsv --out_file test_out.tsv --smiles_column_index 0`
### SMILES in a csv file, output to a tsv file
`airi_calc --in_file test_in.csv --out_file test_out.tsv --smiles_column_index 0`
## Usage instructions
```text
usage: airi_calc.py [-h] [--server SERVER] [--port PORT]
                    (--smiles SMILES [SMILES ...] | --in_file IN_FILE)
                    [--out_file OUT_FILE] [--no_header]
                    [--smiles_column_name SMILES_COLUMN_NAME | --smiles_column_index SMILES_COLUMN_INDEX]
                    [--sheet_name SHEET_NAME]

Given SMILES strings on the command line or in a file, calculate AIRI RI
predictions. Note that the predictions are done one at a time, so many
predictions may take time

optional arguments:
  -h, --help            show this help message and exit
  --server SERVER       database server
  --port PORT           database server
  --smiles SMILES [SMILES ...]
                        space delimited list of smiles strings
  --in_file IN_FILE     name of file to read in (cannot be used with
                        --smiles). File extension (.xlsx, .tsv, .csv)
                        determines type.
  --out_file OUT_FILE   name of file to write out. If not specified, write to
                        stdout
  --no_header           infile and outfile have no header
  --smiles_column_name SMILES_COLUMN_NAME
                        the name of the column containing the SMILES strings
  --smiles_column_index SMILES_COLUMN_INDEX
                        the integer index of the column containing the SMILES
                        strings. The first column is 0
  --sheet_name SHEET_NAME
                        if loading an excel xlsx file, the sheet containing
                        the SMILES. If not specified, the first sheet
```
