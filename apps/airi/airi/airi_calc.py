import urllib
import requests
import argparse
import pandas as pd
import sys
import logging


def main():

    parser = argparse.ArgumentParser(description='Given SMILES strings on the command line or in a file,'
                                                 ' calculate AIRI RI predictions.  Note that the predictions are'
                                                 ' done one at a time, so many predictions may take time')

    parser.add_argument('--server', help="database server", default="10.208.85.87")
    parser.add_argument('--port', help="database server", default="80")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smiles', nargs='+',
                       help='space delimited list of smiles strings',
                       required=False)
    group.add_argument('--in_file', help="name of file to read in (cannot be used with --smiles).  "
                                         "File extension (.xlsx, .tsv, .csv) determines type.")
    parser.add_argument('--out_file', help="name of file to write out.  If not specified, write to stdout")
    parser.add_argument('--no_header', action='store_true', help='infile and outfile have no header')
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--smiles_column_name', help='the name of the column containing the SMILES strings')
    group2.add_argument('--smiles_column_index', type=int,
                        help='the integer index of the column containing the SMILES strings. The first column is 0')
    parser.add_argument('--sheet_name',
                        help='if loading an excel xlsx file, the sheet containing the SMILES.'
                             ' If not specified, the first sheet')
    args = parser.parse_args()

    def call_service(smile_in, server, port):
        """
        call the airi service
        :param smile_in: the smiles string used to compute the ri value
        :param server: the http server containing the airi service
        :param port: the port on the server
        :return: the json response from the server
        """
        smile_in = urllib.parse.quote(smile_in, safe='')
        url = f'http://{server}:{port}/molecule/smiles/{smile_in}/ri/json'
        with requests.get(url) as response_in:
            return response_in.json()

    def get_file_extension(filename_in):
        """
        get the extension of the filename
        :param filename_in: the filename
        :return: the extension
        """
        extension_in = filename_in.split('.')[-1]
        if extension_in not in ['xlsx', 'tsv', 'csv']:
            raise ValueError(f'{filename_in} has an unrecognized extension')
        return extension_in

    header = None if args.no_header else 0  # input header
    output_header = False if args.no_header else True  # output header
    sheet_name = 0 if args.sheet_name is None else args.sheet_name  # name of the sheet to read from
    column_name = args.smiles_column_name if args.smiles_column_name else 'smiles'  # name of the smiles column

    if args.smiles:
        out_file_type = 'tsv'  # set this to get a tsv output
        df = pd.DataFrame(args.smiles, columns=[column_name])
    else:
        in_file_type = get_file_extension(args.in_file)
        out_file_type = in_file_type
        if in_file_type == 'xlsx':
            # note that this only loads in one sheet
            df = pd.read_excel(args.in_file, header=header, sheet_name=sheet_name)
        elif in_file_type == 'tsv':
            df = pd.read_csv(args.in_file, header=header, sep='\t')
        elif in_file_type == 'csv':
            df = pd.read_csv(args.in_file, header=header)

    # extract the SMILES column from the input
    if args.smiles_column_name:
        column = df[args.smiles_column_name].values
    elif args.smiles_column_index:
        column = df.iloc[:, args.smiles_column_index].values
    else:
        column = df.iloc[:, 0].values

    # set up the output file.  previous version used streams, but that causes pandas to output double space
    out_file = sys.stdout
    if args.out_file:
        out_file_type = get_file_extension(args.out_file)
        out_file = args.out_file

    # add in the airi values to the dataframe
    df['airi_values'] = 0
    for smiles in column:
        json_out = call_service(smiles, args.server, args.port)
        if 'error' not in json_out.keys():
            for key, value in json_out.items():
                df.loc[column == key, 'airi_values'] = value

    # output the data
    if out_file_type == 'xlsx':
        # to_excel doesn't use the integer sheet index, so convert it to a sheet name
        if isinstance(sheet_name, int):
            sheet_name = f'Sheet{sheet_name+1}'
        df.to_excel(out_file, header=output_header, sheet_name=sheet_name, index=False)
    elif out_file_type == 'tsv':
        df.to_csv(out_file, header=output_header, sep='\t', index=False)
    elif out_file_type == 'csv':
        df.to_csv(out_file, header=output_header, index=False)


if __name__ == "__main__":
    main()




