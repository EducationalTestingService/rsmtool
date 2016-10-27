#!/usr/bin/env python3

"""
Utility to convert older feature JSON files in tests
newer feature files in tabular formats (csv/tsv/xls/xlsx).

:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import glob
import json
import logging
import os
import sys

from os.path import exists, join

from rsmtool.convert_feature_json import convert_feature_json_file
from rsmtool.utils import LogFormatter


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
            convert_feature_json_file(feature_json, feature_csv, delete=True)
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
