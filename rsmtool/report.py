"""
Functions dealing with report generation.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:organization: ETS
"""

import argparse
import logging
import json
import os

from os.path import abspath, basename, dirname, join, splitext

from traitlets.config import Config
from nbconvert.exporters import HTMLExporter

from rsmtool import HAS_RSMEXTRA

# Check if we have local add-ons
if HAS_RSMEXTRA:
    from rsmextra.settings import (special_section_list_rsmtool,
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
                                    'evaluation_by_group',
                                    'pca',
                                    'sysinfo']

    ordered_section_list_rsmeval = ['data_description',
                                    'data_description_by_group',
                                    'consistency',
                                    'evaluation',
                                    'evaluation_by_group',
                                    'sysinfo']

    ordered_section_list_rsmcompare = ['feature_descriptives',
                                       'features_by_group',
                                       'preprocessed_features',
                                       'preprocessed_features_by_group',
                                       'consistency',
                                       'score_distributions',
                                       'model',
                                       'evaluation',
                                       'pca',
                                       'notes',
                                       'sysinfo']

    ordered_section_list_rsmsummarize = ['preprocessed_features',
                                         'model',
                                         'evaluation',
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

general_section_list_rsmsummarize = [section for section in ordered_section_list_rsmsummarize
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
    Adapted from: http://stackoverflow.com/questions/20454668/how-to-merge-two-ipython-notebooks-correctly-without-getting-json-error.

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
        names. Possible values are {'rsmtool',
                                    'rsmeval',
                                    'rsmcompare'}
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
                         "The following sections are currently available: {}".format(section_type,
                                                                                     context,
                                                                                     invalid_section_names,
                                                                                     master_section_list))


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
        If any sections specifed in the order are missing
        from the list of chosen sections or vice versa.
    """
    if sorted(chosen_sections) != sorted(section_order):

        # check for discrepancies and create a helpful error message
        missing_sections = set(chosen_sections).difference(set(section_order))
        if missing_sections:
            error_message_missing = ("'section_order' must list all "
                                     "sections selected for your experiment: "
                                     "Please edit section order to include the following"
                                     " missing sections: "
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


def determine_chosen_sections(general_sections,
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
    #    a specific (and valid) subset.
    general_section_list = master_section_dict['general'][context]
    chosen_general_sections = general_section_list
    all_general_sections = True

    if general_sections != ['all']:
        check_section_names(general_sections, 'general', context)
        chosen_general_sections = [s for s in general_sections
                                   if s in general_section_list]
        all_general_sections = False

    # 2. Exclude the subgroup sections if we do not have subgroup information.

    if len(subgroups) == 0:
        subgroup_sections = [section for section in chosen_general_sections
                             if section.endswith('by_group')]
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
                                   if not section in subgroup_sections]

    # 3. Include the specified (and valid) subset of the special sections
    chosen_special_sections = []
    if special_sections:
        special_section_list = master_section_dict['special'][context]
        check_section_names(special_sections, 'special', context=context)
        chosen_special_sections = [s for s in special_sections
                                   if s in special_section_list]

    # 4. For the custom sections use the basename and strip off the `.ipynb` extension
    chosen_custom_sections = []
    if custom_sections:
        chosen_custom_sections = [splitext(basename(cs))[0] for cs in custom_sections]

    # return the final list of chosen sections
    chosen_sections = chosen_general_sections + \
                      chosen_special_sections + \
                      chosen_custom_sections

    return chosen_sections


def get_section_file_map(special_sections,
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
        section_file_map.update({ss: join(selected_special_notebook_path, "{}.ipynb".format(ss))
                                 for ss in special_sections})

    # update the file map to include the custom sections with
    # the file names (without the `.ipynb` extension) as the
    # names (keys) and full paths as values
    if custom_sections:
        section_file_map.update({splitext(basename(cs))[0]: cs for cs in custom_sections})

    return section_file_map


def get_ordered_notebook_files(general_sections,
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
    section_order : None, optional
        Ordered list in which the user wants the specified
        sections.
    subgroups : list, optional
        List of column names that contain grouping
        information.
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
    chosen_notebook_files : list of str
        List of the IPython notebook files that have
        to be rendered into the HTML report.
    """

    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups,
                                                context=context)

    # check to make sure that if a custom section ordering is
    # specified by the user, that it actually contains
    # *all* of the sections that have been chosen for the
    # final report.
    if section_order:
        check_section_order(chosen_sections, section_order)

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
    section_file_map = get_section_file_map(special_sections,
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


def create_report(experiment_id, description,
                  model_type, model_name,
                  train_file_location,
                  test_file_location,
                  csvdir, figdir,
                  subgroups,
                  length_column,
                  second_human_score_column,
                  min_items_per_candidate,
                  chosen_notebook_files,
                  feature_subset_file=None,
                  exclude_zero_scores=True,
                  use_scaled_predictions=False,
                  context='rsmtool'):
    """
    The main driver function to generate the RSMTool HTML
    report the experiment as defined by the given arguments.
    """

    logger = logging.getLogger(__name__)

    # set the environment variables we want
    os.environ['EXPERIMENT_ID'] = experiment_id
    os.environ['DESCRIPTION'] = description
    os.environ['CONTEXT'] = context
    os.environ['TRAIN_FILE_LOCATION'] = train_file_location
    os.environ['TEST_FILE_LOCATION'] = test_file_location
    os.environ['OUTPUT_DIR'] = csvdir
    os.environ['FIGURE_DIR'] = figdir
    os.environ['MODEL_NAME'] = model_name
    os.environ['MODEL_TYPE'] = model_type
    os.environ['SCALED'] = '1' if use_scaled_predictions else '0'
    os.environ['EXCLUDE_ZEROS'] = '1' if exclude_zero_scores else '0'
    os.environ['LENGTH_COLUMN'] = '' if length_column is None else length_column
    os.environ['H2_COLUMN'] = '' if second_human_score_column is None else second_human_score_column
    os.environ['MIN_ITEMS'] = '0' if min_items_per_candidate is None else str(min_items_per_candidate)
    os.environ['FEATURE_SUBSET_FILE'] = '' if feature_subset_file is None else feature_subset_file
    os.environ['JAVASCRIPT_PATH'] = javascript_path

    # we define separate groups to allow future flexibility in defining
    # what groups are used for descriptives and evaluations
    os.environ['GROUPS_FOR_DESCRIPTIVES'] = '%%'.join(subgroups)
    os.environ['GROUPS_FOR_EVALUATIONS'] = '%%'.join(subgroups)

    # get the report directory which is at the same level
    # as the output and the figure directory
    reportdir = abspath(join(csvdir, '..', 'report'))
    report_name = '{}_report'.format(experiment_id)
    merged_notebook_file = join(reportdir, '{}.ipynb'.format(report_name))

    # merge all the given sections
    logger.info('Merging sections')
    merge_notebooks(chosen_notebook_files, merged_notebook_file)

    # run the merged notebook and save the output as
    # an HTML file in the report directory
    logger.info('Exporting HTML')
    convert_ipynb_to_html(merged_notebook_file,
                          join(reportdir, '{}.html'.format(report_name)))


def create_comparison_report(comparison_id,
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
                             use_scaled_predictions_old=False,
                             use_scaled_predictions_new=False):
    """
    The main driver function to generate a comparison
    report comparing the two RSMTool experiments as
    defined by the given arguments.
    """

    logger = logging.getLogger(__name__)

    # set the environment variables we want
    os.environ['COMPARISON_ID'] = comparison_id
    os.environ['EXPERIMENT_ID_OLD'] = experiment_id_old
    os.environ['DESCRIPTION_OLD'] = description_old
    os.environ['OUTPUT_DIR_OLD'] = csvdir_old
    os.environ['FIGURE_DIR_OLD'] = figdir_old
    os.environ['SCALED_OLD'] = '1' if use_scaled_predictions_old else '0'

    os.environ['EXPERIMENT_ID_NEW'] = experiment_id_new
    os.environ['DESCRIPTION_NEW'] = description_new
    os.environ['OUTPUT_DIR_NEW'] = csvdir_new
    os.environ['FIGURE_DIR_NEW'] = figdir_new
    os.environ['SCALED_NEW'] = '1' if use_scaled_predictions_new else '0'
    os.environ['JAVASCRIPT_PATH'] = javascript_path

    # we define separate groups to allow future flexibility in defining
    # what groups are used for descriptives and evaluations
    os.environ['GROUPS_FOR_DESCRIPTIVES'] = '%%'.join(subgroups)
    os.environ['GROUPS_FOR_EVALUATIONS'] = '%%'.join(subgroups)

    # create the output directory
    os.makedirs(output_dir, exist_ok=True)
    report_name = '{}_report'.format(comparison_id)
    merged_notebook_file = join(output_dir, '{}.ipynb'.format(report_name))

    # merge all the given sections
    logger.info('Merging sections')
    merge_notebooks(chosen_notebook_files, merged_notebook_file)

    # run the merged notebook and save the output as
    # an HTML file in the report directory
    logger.info('Exporting HTML')
    convert_ipynb_to_html(merged_notebook_file,
                          join(output_dir, '{}.html'.format(report_name)))


def create_summary_report(summary_id, description,
                          all_experiments,
                          csvdir, subgroups,
                          chosen_notebook_files):
    """
    The main function to generate a summary
    report comparing several RSMTool experiments as
    defined by the given arguments.
    """

    logger = logging.getLogger(__name__)

    # set the environment variables we want
    os.environ['SUMMARY_ID'] = summary_id
    os.environ['DESCRIPTION'] = description
    os.environ['JSONS'] = '%%'.join(all_experiments)
    os.environ['OUTPUT_DIR'] = csvdir
    os.environ['JAVASCRIPT_PATH'] = javascript_path


    # we define separate groups to allow future flexibility in defining
    # what groups are used for descriptives and evaluations
    os.environ['GROUPS_FOR_DESCRIPTIVES'] = '%%'.join(subgroups)
    os.environ['GROUPS_FOR_EVALUATIONS'] = '%%'.join(subgroups)

    report_name = '{}_report'.format(summary_id)
    reportdir = abspath(join(csvdir, '..', 'report'))
    merged_notebook_file = join(reportdir, '{}.ipynb'.format(report_name))

    # merge all the given sections
    logger.info('Merging sections')
    merge_notebooks(chosen_notebook_files, merged_notebook_file)

    # run the merged notebook and save the output as
    # an HTML file in the report directory
    logger.info('Exporting HTML')
    convert_ipynb_to_html(merged_notebook_file,
                          join(reportdir, '{}.html'.format(report_name)))



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
    This function is also exposed as the :ref:`render_notebook <render_notebook>` command-line utility.

    """

    # set a high timeout for datasets with a large number of features
    report_config = Config({'ExecutePreprocessor': {'enabled': True,
                                                    'timeout': 3600},
                            'HTMLExporter': {'template_path': [template_path],
                                             'template_file': 'report.tpl'}})

    exportHtml = HTMLExporter(config=report_config)
    output, resources = exportHtml.from_filename(notebook_file)
    open(html_file, mode='w', encoding='utf-8').write(output)


def main():

    # set up an argument parser
    parser = argparse.ArgumentParser(prog='render_notebook')
    parser.add_argument('ipynb_file', help="IPython notebook file")
    parser.add_argument('html_file', help="output HTML file")

    # parse given command line arguments
    args = parser.parse_args()

    # convert notebook to HTML
    convert_ipynb_to_html(args.ipynb_file, args.html_file)

if __name__ == '__main__':
    main()
