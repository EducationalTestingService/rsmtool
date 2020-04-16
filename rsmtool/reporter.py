"""
Classes for dealing with report generation.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import argparse
import logging
import json
import os

from os.path import (abspath,
                     basename,
                     dirname,
                     join,
                     splitext)

from traitlets.config import Config
from nbconvert.exporters import HTMLExporter

from . import HAS_RSMEXTRA
from .reader import DataReader

if HAS_RSMEXTRA:
    from rsmextra.settings import (special_section_list_rsmtool, # noqa
                                   special_section_list_rsmeval,
                                   special_section_list_rsmcompare,
                                   special_section_list_rsmsummarize,
                                   ordered_section_list_with_special_sections_rsmtool,
                                   ordered_section_list_with_special_sections_rsmeval,
                                   ordered_section_list_with_special_sections_rsmcompare,
                                   ordered_section_list_with_special_sections_rsmsummarize,
                                   special_notebook_path)

    ordered_section_list_rsmtool = ordered_section_list_with_special_sections_rsmtool
    ordered_section_list_rsmeval = ordered_section_list_with_special_sections_rsmeval
    ordered_section_list_rsmcompare = ordered_section_list_with_special_sections_rsmcompare
    ordered_section_list_rsmsummarize = ordered_section_list_with_special_sections_rsmsummarize

else:
    ordered_section_list_rsmtool = ['data_description',
                                    'data_description_by_group',
                                    'feature_descriptives',
                                    'features_by_group',
                                    'preprocessed_features',
                                    'dff_by_group',
                                    'consistency',
                                    'model',
                                    'evaluation',
                                    'true_score_evaluation',
                                    'evaluation_by_group',
                                    'fairness_analyses',
                                    'pca',
                                    'intermediate_file_paths',
                                    'sysinfo']

    ordered_section_list_rsmeval = ['data_description',
                                    'data_description_by_group',
                                    'consistency',
                                    'evaluation',
                                    'true_score_evaluation',
                                    'evaluation_by_group',
                                    'fairness_analyses',
                                    'intermediate_file_paths',
                                    'sysinfo']

    ordered_section_list_rsmcompare = ['feature_descriptives',
                                       'features_by_group',
                                       'preprocessed_features',
                                       'preprocessed_features_by_group',
                                       'consistency',
                                       'score_distributions',
                                       'model',
                                       'evaluation',
                                       'true_score_evaluation',
                                       'pca',
                                       'notes',
                                       'sysinfo']

    ordered_section_list_rsmsummarize = ['preprocessed_features',
                                         'model',
                                         'evaluation',
                                         'true_score_evaluation',
                                         'intermediate_file_paths',
                                         'sysinfo']

    special_section_list_rsmtool = []
    special_section_list_rsmcompare = []
    special_section_list_rsmeval = []
    special_section_list_rsmsummarize = []
    special_notebook_path = ""

package_path = dirname(__file__)
notebook_path = abspath(join(package_path, 'notebooks'))
template_path = join(notebook_path, 'templates')

javascript_path = join(notebook_path, 'javascript')
comparison_notebook_path = join(notebook_path, 'comparison')
summary_notebook_path = join(notebook_path, 'summary')


# Define the general section list

general_section_list_rsmtool = [section for section in ordered_section_list_rsmtool
                                if section not in special_section_list_rsmtool]

general_section_list_rsmeval = [section for section in ordered_section_list_rsmeval
                                if section not in special_section_list_rsmeval]

general_section_list_rsmcompare = [section for section in ordered_section_list_rsmcompare
                                   if section not in special_section_list_rsmcompare]

general_section_list_rsmsummarize = [section for section
                                     in ordered_section_list_rsmsummarize
                                     if section not in special_section_list_rsmsummarize]


# define a mapping from the tool name to the master
# list for both general and special sections
master_section_dict = {'general': {'rsmtool': general_section_list_rsmtool,
                                   'rsmeval': general_section_list_rsmeval,
                                   'rsmcompare': general_section_list_rsmcompare,
                                   'rsmsummarize': general_section_list_rsmsummarize},
                       'special': {'rsmtool': special_section_list_rsmtool,
                                   'rsmeval': special_section_list_rsmeval,
                                   'rsmcompare': special_section_list_rsmcompare,
                                   'rsmsummarize': special_section_list_rsmsummarize}}

# define the mapping for section paths
notebook_path_dict = {'general': {'rsmtool': notebook_path,
                                  'rsmeval': notebook_path,
                                  'rsmcompare': comparison_notebook_path,
                                  'rsmsummarize': summary_notebook_path},
                      'special': {'rsmtool': special_notebook_path,
                                  'rsmeval': special_notebook_path,
                                  'rsmcompare': special_notebook_path,
                                  'rsmsummarize': special_notebook_path}}


class Reporter:
    """
    A class for generating Jupyter notebook reports, and
    converting them to HTML.
    """

    @staticmethod
    def locate_custom_sections(custom_report_section_paths, configdir):
        """
        Get the absolute paths for custom report sections and check that
        the files exist. If a file does not exist, raise an exception.

        Parameters
        ----------
        custom_report_section_paths : list of str
            List of paths to IPython notebook
            files representing the custom sections.
        configdir : str
            Path to the experiment configuration directory.

        Returns
        -------
        custom_report_sections :  list of str
            List of absolute paths to the custom section
            notebooks.

        Raises
        ------
        FileNotFoundError
            If any of the files cannot be found.
        """

        custom_report_sections = []
        for cs_path in custom_report_section_paths:
            cs_location = DataReader.locate_files(cs_path, configdir)
            if not cs_location:
                raise FileNotFoundError("Error: custom section not found at "
                                        "{}.".format(cs_path))
            else:
                custom_report_sections.append(cs_location)
        return custom_report_sections

    @staticmethod
    def merge_notebooks(notebook_files, output_file):
        """
        Merge the given Jupyter notebooks into a single Jupyter notebook.

        Parameters
        ----------
        notebook_files : list of str
            List of paths to the input Jupyter notebook files.
        output_file : str
            Path to output Jupyter notebook file

        Note
        ----
        Adapted from: https://stackoverflow.com/q/20454668.
        """

        # Merging ipython notebooks basically means that we keep the
        # metadata from the "first" notebook and then add in the cells
        # from all the other notebooks.
        first_notebook = notebook_files[0]
        merged_notebook = json.loads(open(first_notebook, 'r', encoding='utf-8').read())
        for notebook in notebook_files[1:]:
            section_cells = json.loads(open(notebook, 'r', encoding='utf-8').read())["cells"]
            merged_notebook['cells'].extend(section_cells)

        # output the merged cells into a report
        with open(output_file, 'w') as outf:
            json.dump(merged_notebook, outf, indent=1)

    @staticmethod
    def check_section_names(specified_sections,
                            section_type,
                            context='rsmtool'):
        """
        Check whether the specified section names are valid
        and raise an exception if they are not.

        Parameters
        ----------
        specified_sections : list of str
            List of report section names.
        section_type : str
            'general' or 'special'
        context : str, optional
            Context in which we are validating the section
            names. Possible values are ::

                {'rsmtool', 'rsmeval', 'rsmcompare'}

            Defaults to 'rsmtool'.

        Raises
        ------
        ValueError
            If any of the section names of the given type
            are not valid in the context of the given tool.
        """
        master_section_list = master_section_dict[section_type][context]
        invalid_section_names = set(specified_sections).difference(master_section_list)
        if invalid_section_names:
            raise ValueError("The following {} report section "
                             "names are invalid or not supported for {}: {}\n"
                             "The following sections are currently "
                             "available: {}".format(section_type,
                                                    context,
                                                    invalid_section_names,
                                                    master_section_list))

    @staticmethod
    def check_section_order(chosen_sections, section_order):
        """
        Check the order of the specified sections.

        Parameters
        ----------
        chosen_sections : list of str
            List of chosen section names.
        section_order : list of str
            An ordered list of the chosen section names.

        Raises
        ------
        ValueError
            If any sections specified in the order are missing
            from the list of chosen sections or vice versa.
        """
        if sorted(chosen_sections) != sorted(section_order):

            # check for discrepancies and create a helpful error message
            missing_sections = set(chosen_sections).difference(set(section_order))
            if missing_sections:
                error_message_missing = ("'section_order' must list all "
                                         "sections selected for your experiment: "
                                         "Please edit section order to include the "
                                         "following missing sections: "
                                         "{}".format(', '.join(missing_sections)))

            extra_sections = set(section_order).difference(set(chosen_sections))
            if extra_sections:
                error_message_extra = ("'section order' can only include "
                                       "sections availabe for this experiment. "
                                       "The following sections are either unavailable "
                                       "or were not selected for this experiment "
                                       "{}".format(', '.join(extra_sections)))

            # raise an appropriate error message or a combination of messages
            if missing_sections and not extra_sections:
                raise ValueError(error_message_missing)
            elif extra_sections and not missing_sections:
                raise ValueError(error_message_extra)
            else:
                raise ValueError("{}\n{}".format(error_message_missing,
                                                 error_message_extra))

    @staticmethod
    def convert_ipynb_to_html(notebook_file, html_file):
        """
        Convert the given Jupyter notebook file (``.ipynb``)
        to HTML and write it out as the given ``.html`` file.

        Parameters
        ----------
        notebook_file : str
            Path to input Jupyter notebook file.
        html_file : str
            Path to output HTML file.

        Note
        ----
        This function is also exposed as the
        :ref:`render_notebook <render_notebook>` command-line utility.
        """

        # set a high timeout for datasets with a large number of features
        report_config = Config({'ExecutePreprocessor': {'enabled': True,
                                                        'timeout': 3600},
                                'HTMLExporter': {'template_path': [template_path],
                                                 'template_file': 'report.tpl'}})

        exportHtml = HTMLExporter(config=report_config)
        output, _ = exportHtml.from_filename(notebook_file)
        open(html_file, mode='w', encoding='utf-8').write(output)

    def determine_chosen_sections(self,
                                  general_sections,
                                  special_sections,
                                  custom_sections,
                                  subgroups,
                                  context='rsmtool'):
        """
        Determine the section names that have been chosen
        by the user and that will be generated in the report.

        Parameters
        ----------
        general_sections : list of str
            List of specified general section names.
        special_sections : str
            List of specified special section names, if any.
        custom_sections : list of str
            List of specified custom sections, if any.
        subgroups : list of str
            List of column names that contain grouping
            information.
        context : str, optional
            Context of the tool in which we are validating.
            Possible values are {'rsmtool', 'rsmeval', 'rsmcompare'}
            Defaults to 'rsmtool'.

        Returns
        -------
        chosen_sections : list of str
            Final list of chosen sections that are to
            be included in the HTML report.

        Raises
        ------
        ValueError
            If a subgroup report section is requested but
            no subgroups were specified in the configuration
            file.
        """

        # 1. Include all general sections unless we are asked to include
        # a specific (and valid) subset.
        general_section_list = master_section_dict['general'][context]
        chosen_general_sections = general_section_list
        all_general_sections = True

        if general_sections != ['all']:
            self.check_section_names(general_sections, 'general', context)
            chosen_general_sections = [s for s in general_sections
                                       if s in general_section_list]
            all_general_sections = False

        # 2. Exclude the subgroup sections if we do not have subgroup information.

        if len(subgroups) == 0:
            subgroup_sections = [section for section in chosen_general_sections
                                 if section.endswith('by_group') or section == 'fairness_analyses']
            # if we were given a list of general sections, raise an error if
            # that list included subgroup sections but no subgroups were specified

            if not all_general_sections and len(subgroup_sections) != 0:
                raise ValueError("You requested sections for subgroup analysis "
                                 "but did not specify any subgroups. "
                                 "Please amend the config files to define "
                                 "the subgroups or delete the following "
                                 "sections from the list of sections: {}"
                                 .format(', '.join(subgroup_sections)))

            # if we are using the default list, we simply remove the
            # subgroup sections
            chosen_general_sections = [section for section in chosen_general_sections
                                       if section not in subgroup_sections]

        # 3. Include the specified (and valid) subset of the special sections
        chosen_special_sections = []
        if special_sections:
            special_section_list = master_section_dict['special'][context]
            self.check_section_names(special_sections, 'special', context=context)
            chosen_special_sections = [s for s in special_sections
                                       if s in special_section_list]

        # 4. For the custom sections use the basename and strip off the `.ipynb` extension
        chosen_custom_sections = []
        if custom_sections:
            chosen_custom_sections = [splitext(basename(cs))[0] for cs in custom_sections]

        # return the final list of chosen sections
        chosen_sections = (chosen_general_sections +
                           chosen_special_sections +
                           chosen_custom_sections)

        return chosen_sections

    def get_section_file_map(self,
                             special_sections,
                             custom_sections,
                             model_type=None,
                             context='rsmtool'):
        """
        Map the section names to IPython notebook filenames.

        Parameters
        ----------
        special_sections : list of str
            List of special sections.
        custom_sections : list of str
            List of custom sections.
        model_type : None, optional
            Type of the model. Possible values are
            {'BUILTIN', 'SKLL'}. We allow None here so that
            RSMEval can use the same function.
        context : str, optional
            Context of the tool in which we are validating.
            Possible values are {'rsmtool', 'rsmeval', 'rsmcompare'}
            Defaults to 'rsmtool'

        Returns
        -------
        section_file_map : dict
            Dictionary mapping each section name to the
            corresponding IPython notebook filename.
        """

        # create the original section file map for general sections
        selected_notebook_path = notebook_path_dict['general'][context]
        general_sections = master_section_dict['general'][context]

        section_file_map = {s: join(selected_notebook_path, '{}.ipynb'.format(s))
                            for s in general_sections + ['header', 'footer']}

        # update the file map to point the 'model section to either the built-in
        # or the SKLL model notebook depending on the model type that
        # was passed in
        if context == 'rsmtool' and model_type:
            section_file_map['model'] = join(selected_notebook_path,
                                             '{}_model.ipynb'.format(model_type.lower()))

        # update the file map to include the special sections
        if special_sections:
            selected_special_notebook_path = notebook_path_dict['special'][context]
            section_file_map.update({ss: join(selected_special_notebook_path,
                                              "{}.ipynb".format(ss))
                                     for ss in special_sections})

        # update the file map to include the custom sections with
        # the file names (without the `.ipynb` extension) as the
        # names (keys) and full paths as values
        if custom_sections:
            section_file_map.update({splitext(basename(cs))[0]: cs
                                     for cs in custom_sections})

        return section_file_map

    def get_ordered_notebook_files(self,
                                   general_sections,
                                   special_sections=[],
                                   custom_sections=[],
                                   section_order=None,
                                   subgroups=[],
                                   model_type=None,
                                   context='rsmtool'):
        """
        Check all section names and section order,
        combine all section names with the appropriate file mapping,
        and generate an ordered list of notebook files that are
        needed to generate the final report.

        Parameters
        ----------
        general_sections : str
            List of specified general sections.
        special_sections : list, optional
            List of specified special sections, if any.
        custom_sections : list, optional
            List of specified custom sections, if any.
            Defaults to empty list.
        section_order : None, optional
            Ordered list in which the user wants the specified
            sections.
            Defaults to empty list
        subgroups : list, optional
            List of column names that contain grouping
            information.
            Defaults to empty list.
        model_type : None, optional
            Type of the model. Possible values are
            {'BUILTIN', 'SKLL'}. We allow None here so that
            RSMEval can use the same function.
            Defaults to None
        context : str, optional
            Context of the tool in which we are validating.
            Possible values are {'rsmtool', 'rsmeval', 'rsmcompare'}
            Defaults to 'rsmtool'

        Returns
        -------
        chosen_notebook_files : list of str
            List of the IPython notebook files that have
            to be rendered into the HTML report.
        """

        chosen_sections = self.determine_chosen_sections(general_sections,
                                                         special_sections,
                                                         custom_sections,
                                                         subgroups,
                                                         context=context)

        # check to make sure that if a custom section ordering is
        # specified by the user, that it actually contains
        # *all* of the sections that have been chosen for the
        # final report.
        if section_order:
            self.check_section_order(chosen_sections, section_order)

        # determine which order to use by default
        if context == 'rsmtool':
            ordered_section_list = ordered_section_list_rsmtool
        elif context == 'rsmeval':
            ordered_section_list = ordered_section_list_rsmeval
        elif context == 'rsmcompare':
            ordered_section_list = ordered_section_list_rsmcompare
        elif context == 'rsmsummarize':
            ordered_section_list = ordered_section_list_rsmsummarize

        # add all custom sections to the end of the default ordered list
        ordered_section_list.extend([splitext(basename(cs))[0] for cs in custom_sections])

        # get the section file map
        section_file_map = self.get_section_file_map(special_sections,
                                                     custom_sections,
                                                     model_type,
                                                     context=context)

        # order the section list either according to the default
        # order in `ordered_section_list` or according to the custom
        # order that has been passed in via `section_order`
        order_to_use = section_order if section_order else ordered_section_list
        chosen_sections = [s for s in order_to_use if s in chosen_sections]

        # add the header and the footer to the chosen sections
        chosen_sections = ['header'] + chosen_sections + ['footer']
        chosen_notebook_files = [section_file_map[cs] for cs in chosen_sections]

        return chosen_notebook_files

    def create_report(self,
                      config,
                      csvdir,
                      figdir,
                      context='rsmtool'):
        """
        The main driver function to generate the RSMTool HTML
        report the experiment as defined by the given arguments.

        Parameters
        ----------

        config : configuration_parser.Configuration
            A configuration object
        csvdir : str
            The CSV output directory.
        figdir : str
            The figure output directory
        context : str
            The context of the script
            Defaults to 'rsmtool'

        Returns
        -------
        notebook
            A jupyter notebook

        Raises
        ------
        KeyError
            If `test_file_location or `pred_file_location` not in config.
        """

        logger = logging.getLogger(__name__)

        test_file_location = config.get('test_file_location')
        if test_file_location is None:
            test_file_location = config.get('pred_file_location')

        # raise error if location is still None
        if test_file_location is None:
            raise KeyError('Could not find `test_file_location` or `pred_file_location` '
                           'in Configuration object. Please make sure you have included '
                           'one of these parameters in the configuration object.')

        # if the features subset file is None, just use empty string
        feature_subset_file = ('' if config.get('feature_subset_file') is None
                               else config['feature_subset_file'])

        # if the min items is None, just set it to zero
        min_items = (0 if config['min_items_per_candidate'] is None
                     else config['min_items_per_candidate'])

        # determine minimum and maximum scores for trimming
        (used_trim_min,
         used_trim_max,
         trim_tolerance) = config.get_trim_min_max_tolerance()
        min_score = used_trim_min - trim_tolerance
        max_score = used_trim_max + trim_tolerance

        environ_config = {'EXPERIMENT_ID': config['experiment_id'],
                          'DESCRIPTION': config['description'],
                          'MODEL_TYPE': config.get('model_type', ''),
                          'MODEL_NAME': config.get('model_name', ''),
                          'TRAIN_FILE_LOCATION': config.get('train_file_location', ''),
                          'TEST_FILE_LOCATION': test_file_location,
                          'SUBGROUPS': config.get('subgroups', []),
                          'GROUPS_FOR_DESCRIPTIVES': config.get('subgroups', []),
                          'GROUPS_FOR_EVALUATIONS': config.get('subgroups', []),
                          'MIN_N_PER_GROUP': config.get('min_n_per_group', {}),
                          'LENGTH_COLUMN': config.get('length_column', None),
                          'H2_COLUMN': config['second_human_score_column'],
                          'MIN_ITEMS': min_items,
                          'FEATURE_SUBSET_FILE': feature_subset_file,
                          'EXCLUDE_ZEROS': config.get('exclude_zero_scores', True),
                          'SCALED': config.get('use_scaled_predictions', False),
                          'MIN_SCORE': min_score,
                          'MAX_SCORE': max_score,
                          'STANDARDIZE_FEATURES': config.get('standardize_features', True),
                          'FILE_FORMAT': config.get('file_format', 'csv'),
                          'USE_THUMBNAILS': config.get('use_thumbnails', False),
                          'SKLL_FIXED_PARAMETERS': config.get('skll_fixed_parameters', {}),
                          'SKLL_OBJECTIVE': config.get('skll_objective', ''),
                          'PREDICT_EXPECTED_SCORES': config.get('predict_expected_scores', False),
                          'RATER_ERROR_VARIANCE': config.get('rater_error_variance', None),
                          'CONTEXT': context,
                          'JAVASCRIPT_PATH': javascript_path,
                          'OUTPUT_DIR': csvdir,
                          'FIGURE_DIR': figdir
                          }

        # get the report directory which is at the same level
        # as the output and the figure directory
        reportdir = abspath(join(csvdir, '..', 'report'))
        report_name = '{}_report'.format(config['experiment_id'])
        merged_notebook_file = join(reportdir, '{}.ipynb'.format(report_name))
        environ_config_file = join(reportdir, '.environ.json')

        # set the report directory as an environment variable
        os.environ['RSM_REPORT_DIR'] = reportdir

        # write out hidden environment JSON file
        with open(environ_config_file, 'w') as out_environ_config:
            json.dump(environ_config, out_environ_config)

        # merge all the given sections
        logger.info('Merging sections')
        self.merge_notebooks(config['chosen_notebook_files'], merged_notebook_file)

        # run the merged notebook and save the output as
        # an HTML file in the report directory
        logger.info('Exporting HTML')
        self.convert_ipynb_to_html(merged_notebook_file,
                                   join(reportdir, '{}.html'.format(report_name)))

    def create_comparison_report(self,
                                 config,
                                 csvdir_old,
                                 figdir_old,
                                 csvdir_new,
                                 figdir_new,
                                 output_dir):
        """
        The main driver function to generate a comparison
        report comparing the two RSMTool experiments as
        defined by the given arguments.

        Parameters
        ----------

        config : configuration_parser.Configuration
            A configuration object
        csvdir_old : str
            The old experiment CSV output directory.
        figdir_old : str
            The old figure output directory
        csvdir_new : str
            The new experiment CSV output directory.
        figdir_new : str
            The old figure output directory
        output_dir : str
            The output dir for the new report.

        Returns
        -------
        notebook
            A jupyter notebook
        """

        logger = logging.getLogger(__name__)

        # whether to use scaled predictions for both new and old; default to false
        use_scaled_predictions_old = config.get('use_scaled_predictions_old', False)
        use_scaled_predictions_new = config.get('use_scaled_predictions_new', False)

        environ_config = {'EXPERIMENT_ID_OLD': config['experiment_id_old'],
                          'EXPERIMENT_ID_NEW': config['experiment_id_new'],
                          'DESCRIPTION_OLD': config['description_old'],
                          'DESCRIPTION_NEW': config['description_new'],
                          'GROUPS_FOR_DESCRIPTIVES': config.get('subgroups', []),
                          'GROUPS_FOR_EVALUATIONS': config.get('subgroups', []),
                          'SCALED_OLD': use_scaled_predictions_old,
                          'SCALED_NEW': use_scaled_predictions_new,
                          'USE_THUMBNAILS': config.get('use_thumbnails', False),
                          'JAVASCRIPT_PATH': javascript_path,
                          'OUTPUT_DIR_NEW': csvdir_new,
                          'FIGURE_DIR_NEW': figdir_new,
                          'OUTPUT_DIR_OLD': csvdir_old,
                          'FIGURE_DIR_OLD': figdir_old,
                          }

        # create the output directory
        os.makedirs(output_dir, exist_ok=True)
        report_name = '{}_report'.format(config['comparison_id'])
        merged_notebook_file = join(output_dir, '{}.ipynb'.format(report_name))
        environ_config_file = join(output_dir, '.environ.json')

        # set the report directory as an environment variable
        os.environ['RSM_REPORT_DIR'] = abspath(output_dir)

        # write out hidden environment JSON file
        with open(environ_config_file, 'w') as out_environ_config:
            json.dump(environ_config, out_environ_config)

        # merge all the given sections
        logger.info('Merging sections')
        self.merge_notebooks(config['chosen_notebook_files'], merged_notebook_file)

        # run the merged notebook and save the output as
        # an HTML file in the report directory
        logger.info('Exporting HTML')
        self.convert_ipynb_to_html(merged_notebook_file,
                                   join(output_dir, '{}.html'.format(report_name)))

    def create_summary_report(self,
                              config,
                              all_experiments,
                              csvdir):
        """
        The main function to generate a summary
        report comparing several RSMTool experiments as
        defined by the given arguments.

        Parameters
        ----------

        config : configuration_parser.Configuration
            A configuration object
        all_experiments : list
            A list of experiments to summarize.
        csvdir : str
            The experiment CSV output directory.

        Returns
        -------
        notebook
            A jupyter notebook
        """

        logger = logging.getLogger(__name__)

        environ_config = {'SUMMARY_ID': config['summary_id'],
                          'DESCRIPTION': config['description'],
                          'GROUPS_FOR_DESCRIPTIVES': config.get('subgroups', []),
                          'GROUPS_FOR_EVALUATIONS': config.get('subgroups', []),
                          'USE_THUMBNAILS': config.get('use_thumbnails', False),
                          'FILE_FORMAT': config.get('file_format', 'csv'),
                          'JSONS': all_experiments,
                          'JAVASCRIPT_PATH': javascript_path,
                          'OUTPUT_DIR': csvdir
                          }

        report_name = '{}_report'.format(config['summary_id'])
        reportdir = abspath(join(csvdir, '..', 'report'))
        merged_notebook_file = join(reportdir, '{}.ipynb'.format(report_name))
        environ_config_file = join(reportdir, '.environ.json')

        # set the report directory as an environment variable
        os.environ['RSM_REPORT_DIR'] = reportdir

        # write out hidden environment JSON file
        with open(environ_config_file, 'w') as out_environ_config:
            json.dump(environ_config, out_environ_config)

        # merge all the given sections
        logger.info('Merging sections')
        self.merge_notebooks(config['chosen_notebook_files'], merged_notebook_file)

        # run the merged notebook and save the output as
        # an HTML file in the report directory
        logger.info('Exporting HTML')
        self.convert_ipynb_to_html(merged_notebook_file,
                                   join(reportdir, '{}.html'.format(report_name)))


def main():

    # set up an argument parser
    parser = argparse.ArgumentParser(prog='render_notebook')
    parser.add_argument('ipynb_file', help="IPython notebook file")
    parser.add_argument('html_file', help="output HTML file")

    # parse given command line arguments
    args = parser.parse_args()

    # convert notebook to HTML
    Reporter.convert_ipynb_to_html(args.ipynb_file, args.html_file)


if __name__ == '__main__':

    main()
