"""
Provides a function to convert older feature JSON files to
newer feature files in tabular formats (csv/tsv/xls/xlsx).

:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)

:organization: ETS
"""
import argparse
import json
import os
import pandas as pd

from os.path import splitext


def convert_feature_json_file(json_file, output_file, delete=False):
    """
    Convert the given feature JSON file into a tabular
    format inferred by the extension of the output file.

    Parameters
    ----------
    json_file : str
        Path to feature JSON file that is to be converted.
    output_file : str
        Path to CSV/TSV/XLS/XLSX output file.
    delete : bool, optional
        Whether to delete the original file after conversion.
        Defaults to False.

    Raises
    ------
    RuntimeError
        If the given input file is not a valid feature JSON file
        or if the output file has an unsupported extension.
    """

    # make sure the input file is a valid feature JSON file
    json_dict = json.load(open(json_file, 'r'))
    if not list(json_dict.keys()) == ['features']:
        raise RuntimeError("{} is not a valid feature JSON "
                           "file".format(json_file))

    # convert to tabular format
    df_feature = pd.DataFrame(json_dict['features'])

    # make sure the output file is in a supported format
    output_extension = splitext(output_file)[1].lower()
    if output_extension not in ['.csv', '.tsv', '.xls', '.xlsx']:
        raise RuntimeError("The output file {} has an unsupported "
                           "extension. It must be a CSV/TSV/XLS/XLSX "
                           "file.".format(output_file))

    if output_extension == '.csv':
        df_feature.to_csv(output_file, index=False)
    elif output_extension == '.tsv':
        df_feature.to_csv(output_file, sep='\t', index=False)
    elif output_extension in ['.xls', '.xlsx']:
        df_feature.to_excel(output_file, sheet_name='features', index=False)

    if delete:
        os.unlink(json_file)


def main():
    parser = argparse.ArgumentParser(prog='convert_feature_json')
    parser.add_argument('json_file',
                        help="The feature JSON file to convert "
                             "to tabular format.")
    parser.add_argument('output_file',
                        help="The output file containing the features "
                             "in tabular format.")
    parser.add_argument('--delete',
                        help="Delete original JSON file after conversion.",
                        default=False,
                        required=False,
                        action="store_true")

    args = parser.parse_args()
    convert_feature_json_file(args.json_file,
                              args.output_file,
                              delete=args.delete)


if __name__ == '__main__':

    main()
