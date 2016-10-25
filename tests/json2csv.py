# python

'''
A utility script to conver feature json to a tabular format
'''

import argparse
import glob
import json

import os
from os.path import dirname, exists, join, splitext


import pandas as pd

TEST_DIR = dirname(__file__)


def convert_json_to_csv(json_file, output_file, delete):
    json_dict = json.load(open(json_file, 'r'))
    if not len(json_dict.keys()) == 1 and json_dict.keys()[0] == 'features':
        print("File {} is not a valid feature json file".format(json_file))
    else:
        df_feature = pd.DataFrame(json_dict['features'])
        output_extension = splitext(output_file)[1].lower()
        if output_extension == '.csv':
            df_feature.to_csv(output_file, index=False)
            if delete:
                os.remove(json_file)
        elif output_extension == '.tsv':
            df_feature.to_csv(output_file, sep='\t', index=False)
            if delete:
                os.remove(json_file)
        elif output_extension in ['.xlsx', '.xls']:
            df_feature.to_excel(output_file, sheet_name='features', index=False)
            if delete:
                os.remove(json_file)
        else:
            print("File {} is in unsupported format".format(output_file))


def convert_all():
    experiments_dir = os.path.join(TEST_DIR, 'data', 'experiments')
    for experiment in os.listdir(experiments_dir):
        if 'predict' in experiment:
            if not 'rsmtool' in experiment:
                continue
        if 'eval' in experiment or 'compare' in experiment or 'summary' in experiment:
            continue
        if 'json' in experiment or 'old-config' in experiment:
            continue
        else:
            print(experiment)
            feature_json = join(experiments_dir, experiment, 'features.json')
            feature_csv = join(experiments_dir, experiment, 'features.csv')
            if exists(feature_json):
                convert_json_to_csv(feature_json,
                                    feature_csv, delete=True)
            else:
                print("No features.json file")
            jsons = glob.glob(join(experiments_dir, experiment, '*.json'))
            experiment_json = [json for json in jsons if not json == feature_json]
            if len(experiment_json) > 1:
                print("Found more than 1 json in {}".format(experiment))
            config_dict = json.load(open(experiment_json[0], 'r'))
            if 'features' in config_dict and config_dict['features'] == 'features.json':
                config_dict['features'] = 'features.csv'
                with open(experiment_json[0], 'w') as outfile:
                    json.dump(config_dict, outfile, indent=4, separators=(',', ': '))






if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='json2csv.py')
    parser.add_argument('--all', 
                        dest='convert_all',
                        action='store_true')
    parser.add_argument('--json', help="The original feature json file",
                        dest = 'json_file',
                        required=False)
    parser.add_argument('--output', 
                        dest='output_file',
                        help="The original feature json file",
                        required=False)
    parser.add_argument('--delete', 
                        dest='delete',
                        action="store_true")
    args = parser.parse_args()
    if convert_all:
        convert_all()
    else:
        convert_json_to_csv(args.json_file, args.output_file, args.delete)



