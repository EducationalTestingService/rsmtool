"""
Utility to convert older feature JSON files to
newer feature files in tabular formats (csv/tsv/xls/xlsx).

:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import argparse
import glob
import json
import logging
import os
import sys

import pandas as pd

from os.path import dirname, exists, join, splitext
from rsmtool.utils import LogFormatter


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

    Raises
    ------
    OSError
        If the given input file is not a valid feature JSON file
        or if the output file has an unsupported extension.
    """

    # make sure the input file is a valid feature JSON file
    json_dict = json.load(open(json_file, 'r'))
    if not list(json_dict.keys()) == ['features']:
        raise OSError("{} is not a valid feature JSON file".format(json_file))

    # convert to tabular format
    df_feature = pd.DataFrame(json_dict['features'])

    # make sure the output file is in a supported format
    output_extension = splitext(output_file)[1].lower()
    if output_extension not in ['csv', 'tsv', 'xls', 'xlsx']:
        raise OSError("The output file {} has an unsupported extension. "
                      "It must be a CSV/TSV/XLS/XLSX file.".format(output_file))

    if output_extension == '.csv':
        df_feature.to_csv(output_file, index=False)
    elif output_extension == '.tsv':
        df_feature.to_csv(output_file, sep='\t', index=False)
    elif output_extension in ['.xls', '.xlsx']:
        df_feature.to_excel(output_file, sheet_name='features', index=False)

    if delete:
        os.unlink(json_file)


def convert_tests(test_dir):

    # set up the basic logging config
    fmt = LogFormatter()
    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setFormatter(fmt)
    logging.root.addHandler(hdlr)
    logging.root.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)

    # get the path to th rsmtool experiment directory
    experiments_dir = join(test_dir, 'data', 'experiments')

    # iterate over all the test experiments
    for experiment in os.listdir(experiments_dir):

        # skip over irrelevant experiments
        if ('eval' in experiment or
            'compare' in experiment or
            'summary' in experiment or
            'json' in experiment or
            'old-config' in experiment):
            continue
        if 'predict' in experiment and not 'rsmtool' in experiment:
                continue
        if experiment.startswith('.'):
            continue

        # Now deal with the relevant experiments
        logger.info(experiment)

        feature_json = join(experiments_dir, experiment, 'features.json')
        feature_csv = join(experiments_dir, experiment, 'features.csv')
        if exists(feature_json):
            logger.info("  Converting JSON to CSV.")
            convert_json_to_csv(feature_json, feature_csv, delete=True)
        else:
            logger.info("  No feature JSON file found.")
            continue

        jsons = glob.glob(join(experiments_dir, experiment, '*.json'))
        experiment_json = [json for json in jsons if not json == feature_json]
        if len(experiment_json) > 1:
            logger.info("  Found more than one experiment json in {}.".format(experiment))
        config_dict = json.load(open(experiment_json[0], 'r'))
        if 'features' in config_dict and config_dict['features'] == 'features.json':
            config_dict['features'] = 'features.csv'
            with open(experiment_json[0], 'w') as outfile:
                json.dump(config_dict, outfile, indent=4, separators=(',', ': '))

def main():
    parser = argparse.ArgumentParser(prog='convert_feature_json')
    parser.add_argument('--json',
                        dest='json_file',
                        required=False,
                        help="The feature JSON file to convert to tabular format.")
    parser.add_argument('--output',
                        dest='output_file',
                        required=False,
                        help="The output file containing the features in tabular format.")
    parser.add_argument('--delete',
                        help="Delete original JSON file after conversion.",
                        default=False,
                        required=False,
                        action="store_true")
    parser.add_argument('--tests',
                        dest='test_dir',
                        default=None,
                        required=False,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.test_dir:
        convert_tests(args.test_dir)
    else:
        if args.json_file and args.output_file:
            convert_feature_json_file(args.json_file,
                                      args.output_file,
                                      delete=args.delete)

if __name__ == '__main__':
    main()
