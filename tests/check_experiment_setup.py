#!/usr/bin/env python

"""
This script is designed to check that the tests in test_experiment.py
do not have any errors based on mismatching names between the experiment
ID in the test function, the experiment ID in the config file and the
prefixes of the expected output data.

If there are any problems, it will raise an AssertionError.

:author: Nitin Madnani
:date: March 2016
"""

import ast
import importlib
import inspect
import json
import os
import re

from os.path import exists

# import the test_experiment.py test module
test_experiment = importlib.import_module('test_experiment')

# iterate over all the members and focus on only the experiment functions
for member in inspect.getmembers(test_experiment):
    if inspect.isfunction(member[1]) and member[0].startswith('test_run_experiment'):
        function = member[1]
        # get the experiment id and the source for each test function
        function_code_lines = inspect.getsourcelines(member[1])
        experiment_id_line = [line for line in function_code_lines[0]
                              if re.search(r'experiment_id = ', line)]

        # if there was no experiment ID specified, it was either a
        # a decorated test function (which means it wouldn't have an 'output'
        # directory anyway) or it was a compare or prediction test function
        # which are not a problem for various reasons.
        if experiment_id_line:
            experiment_id_in_test = ast.literal_eval(experiment_id_line[0].strip().split(' = ')[1])

            # get the name of the source directory
            source_line = [line for line in function_code_lines[0]
                                  if re.search(r'source = ', line)]
            source = ast.literal_eval(source_line[0].strip().split(' = ')[1])
            print(source)

            # make sure the config file starts with the same name as the
            # experiment ID that was specified in the test function
            config_file = 'data/experiments/{}/{}.json'.format(source, experiment_id_in_test)
            assert exists(config_file)

            # read in the config file and make sure that the experiment
            # ID it contains matches what was in the test function
            with open(config_file, 'r') as configf:
                config_obj = json.loads(configf.read())
                if 'expID' in config_obj:
                    experiment_id_in_config = config_obj['expID']
                else:
                    experiment_id_in_config = config_obj['experiment_id']
            assert experiment_id_in_test == experiment_id_in_config

            # read in all the files under the 'output' directory
            # for this test and make sure that they all start
            # with the same experiment ID
            output_files = os.listdir('data/experiments/{}/output'.format(source))
            assert len(output_files) > 0
            non_matching_output_files = [f for f in output_files if not f.startswith(experiment_id_in_config)]
            assert len(non_matching_output_files) == 0
