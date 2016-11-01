"""
Script to compare two RSMTool experiments

:author: Nitin Madnani (nmadnani@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:organization: ETS
"""

#!/usr/bin/env python

import argparse
import glob
import logging
import os
import sys

from os.path import abspath, dirname, exists, join, normpath

from rsmtool.input import (read_json_file,
                           check_main_config,
                           locate_file,
                           locate_custom_sections)

from rsmtool.report import (create_comparison_report,
                            get_ordered_notebook_files)

from rsmtool.utils import LogFormatter
from rsmtool.version import __version__


def check_experiment_id(experiment_dir, experiment_id):
    """
    Check that the supplied ``experiment_dir`` contains
    the outputs for the supplied ``experiment_id``

    Parameters
    ----------
    experiment_id : str
        experiment_id of the original experiment used to generate the
        output
    experiment_dir : str
        path to the directory with the experiment output

    Raises
    FileNotFoundError
        if the ``experument_dir`` does not contain any outputs
        for the ``experiment_id``
    """

    # list all possible output files which start with
    # experiment_id
    outputs = glob.glob(join(experiment_dir,
                             'output',
                             '{}_*.*'.format(experiment_id)))

    # raise an error if none exists
    if len(outputs) == 0:
        raise FileNotFoundError("The directory {} does not contain "
                                "any outputs of an rsmtool experiment "
                                "{}".format(experiment_dir, experiment_id))



def run_comparison(config_file, output_dir):
    """
    Run an ``rsmcompare`` experiment using the given configuration
    file and generate the report in the given directory.

    Parameters
    ----------
    config_file : str
        Path to the experiment configuration file.
    output_dir : str
        Path to the experiment output directory.

    Raises
    ------
    ValueError
        If any of the required fields are missing or ill-specified.

    """

    logger = logging.getLogger(__name__)

    # load the information from the config file
    # read in the main config file
    config_obj = read_json_file(config_file)
    config_obj = check_main_config(config_obj, context='rsmcompare')

    # get the subgroups if any
    subgroups = config_obj.get('subgroups')

    # get the directory where the config file lives
    configpath = dirname(config_file)

    # get the comparison ID
    comparison_id = config_obj['comparison_id']

    # get the information about the "old" experiment
    description_old = config_obj['description_old']
    experiment_id_old = config_obj['experiment_id_old']
    experiment_dir_old = locate_file(config_obj['experiment_dir_old'], configpath)
    if not experiment_dir_old:
        raise FileNotFoundError("The directory {} "
                                "does not exist.".format(config_obj['experiment_dir_old']))
    else:
        csvdir_old = normpath(join(experiment_dir_old, 'output'))
        figdir_old = normpath(join(experiment_dir_old, 'figure'))
        if not exists(csvdir_old) or not exists(figdir_old):
            raise FileNotFoundError("The directory {} does not contain "
                                    "the output of an rsmtool "
                                    "experiment.".format(experiment_dir_old))

    check_experiment_id(experiment_dir_old, experiment_id_old)

    use_scaled_predictions_old = config_obj['use_scaled_predictions_old']

    # get the information about the "new" experiment
    description_new = config_obj['description_new']
    experiment_id_new = config_obj['experiment_id_new']
    experiment_dir_new = locate_file(config_obj['experiment_dir_new'], configpath)
    if not experiment_dir_new:
        raise FileNotFoundError("The directory {} "
                                "does not exist.".format(config_obj['experiment_dir_new']))
    else:
        csvdir_new = normpath(join(experiment_dir_new, 'output'))
        figdir_new = normpath(join(experiment_dir_new, 'figure'))
        if not exists(csvdir_new) or not exists(figdir_new):
            raise FileNotFoundError("The directory {} does not contain "
                                    "the output of an rsmtool "
                                    "experiment.".format(experiment_dir_new))

    check_experiment_id(experiment_dir_new, experiment_id_new)

    use_scaled_predictions_new = config_obj['use_scaled_predictions_new']

    # are there specific general report sections we want to include?
    general_report_sections = config_obj['general_sections']

    # what about the special or custom sections?
    special_report_sections = config_obj['special_sections']

    custom_report_section_paths = config_obj['custom_sections']

    if custom_report_section_paths:
        logger.info('Locating custom report sections')
        custom_report_sections = locate_custom_sections(custom_report_section_paths,
                                                        configpath)
    else:
        custom_report_sections = []

    section_order = config_obj['section_order']

    chosen_notebook_files = get_ordered_notebook_files(general_report_sections,
                                                       special_report_sections,
                                                       custom_report_sections,
                                                       section_order,
                                                       subgroups,
                                                       model_type=None,
                                                       context='rsmcompare')

    # now generate the comparison report
    logger.info('Starting report generation')
    create_comparison_report(comparison_id,
                             experiment_id_old,
                             description_old,
                             csvdir_old,
                             figdir_old,
                             experiment_id_new,
                             description_new,
                             csvdir_new,
                             figdir_new,
                             output_dir,
                             subgroups,
                             chosen_notebook_files,
                             use_scaled_predictions_old=use_scaled_predictions_old,
                             use_scaled_predictions_new=use_scaled_predictions_new)


def main():

    # set up the basic logging config
    fmt = LogFormatter()
    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setFormatter(fmt)
    logging.root.addHandler(hdlr)
    logging.root.setLevel(logging.INFO)

    # get a logger
    logger = logging.getLogger(__name__)

    # set up an argument parser
    parser = argparse.ArgumentParser(prog='rsmcompare')

    parser.add_argument('config_file', help="The JSON config file for "
                                            "this comparison")

    parser.add_argument('output_dir', nargs='?', default=os.getcwd(),
                        help="The output directory where the report "
                             "files for this comparison will be stored")

    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s {0}'.format(__version__))

    # parse given command line arguments
    args = parser.parse_args()
    logger.info('Output directory: {}'.format(args.output_dir))

    # convert all paths to absolute to make sure
    # all files can be found later
    config_file = abspath(args.config_file)
    output_dir = abspath(args.output_dir)

    # make sure that the given config file exists
    if not exists(config_file):
        raise FileNotFoundError("Main config file {} not "
                                "found.".format(config_file))

    # generate a comparison report
    run_comparison(config_file, output_dir)


if __name__ == '__main__':
    main()
