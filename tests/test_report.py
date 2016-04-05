from nose.tools import eq_, raises, ok_

from os.path import join, normpath

from rsmtool.report import (check_section_names,
                            check_section_order,
                            determine_chosen_sections,
                            get_ordered_notebook_files,
                            get_section_file_map,
                            master_section_dict,
                            notebook_path,
                            comparison_notebook_path,
                            notebook_path_dict)

# Since we are in test mode, we want to add a placeholder
# special section to the master section list so that the tests
# will run irrespective of whether or not rsmextra is installed
# We also set a placeholder special_notebook path

for context in ['rsmtool', 'rsmeval', 'rsmcompare']:
    master_section_dict['special'][context].append('placeholder_special_section')
    notebook_path_dict['special'].update({context:'special_notebook_path'})

# define the general sections lists to keep tests more readable

general_section_list_rsmtool = master_section_dict['general']['rsmtool']
general_section_list_rsmeval = master_section_dict['general']['rsmeval']
general_section_list_rsmcompare = master_section_dict['general']['rsmcompare']


def check_section_lists(context):
    general_sections = master_section_dict['general'][context]
    special_sections = master_section_dict['special'][context]
    overlap = set(general_sections) & set(special_sections)
    # check that there are general section
    ok_(len(general_sections) > 0)
    # check that there is no overlap between general and special section
    # list
    eq_(len(overlap), 0)


def test_check_section_lists_rsmtool():
    # sanity checks to make sure nothing went wrong when generating
    # master section list
    for context in ['rsmtool', 'rsmeval', 'rsmcompare']:
        yield check_section_lists, context


@raises(ValueError)
def test_check_section_order_not_enough_sections():
    general_sections = ['evaluation', 'sysinfo']
    special_sections = ['placeholder_special_section']
    custom_sections = ['custom.ipynb']
    subgroups = ['prompt', 'gender']
    section_order = general_sections
    get_ordered_notebook_files(general_sections,
                               special_sections=special_sections,
                               custom_sections=custom_sections,
                               section_order=section_order,
                               subgroups=subgroups)


@raises(ValueError)
def test_check_section_order_extra_sections():
    general_sections = ['evaluation', 'sysinfo']
    special_sections = ['placeholder_special_section']
    custom_sections = ['custom.ipynb']
    subgroups = []
    section_order = general_sections + special_sections + custom_sections + ['extra_section']
    get_ordered_notebook_files(general_sections,
                               special_sections=special_sections,
                               custom_sections=custom_sections,
                               section_order=section_order,
                               subgroups=subgroups)


@raises(ValueError)
def test_check_section_order_wrong_sections():
    general_sections = ['evaluation', 'sysinfo']
    special_sections = ['placeholder_special_section']
    custom_sections = ['custom.ipynb']
    subgroups = []
    section_order = ['extra_section1', 'extra_section2']
    get_ordered_notebook_files(general_sections,
                               special_sections=special_sections,
                               custom_sections=custom_sections,
                               section_order=section_order,
                               subgroups=subgroups)


def test_check_section_order():
    general_sections = ['evaluation', 'sysinfo']
    special_sections = ['placeholder_special_section']
    custom_sections = ['foobar']
    section_order = (['foobar'] +
                     special_sections +
                     general_sections)
    check_section_order(general_sections + \
                        special_sections + \
                        custom_sections, section_order)


def test_check_general_section_names_rsmtool():
    specified_list = ['data_description', 'preprocessed_features']
    check_section_names(specified_list, 'general')


@raises(ValueError)
def test_check_general_section_names_wrong_names():
    specified_list = ['data_description', 'feature_stats']
    check_section_names(specified_list, 'general')


def test_check_general_section_names_rsmeval():
    specified_list = ['data_description', 'evaluation']
    check_section_names(specified_list, 'general', context='rsmeval')


@raises(ValueError)
def test_check_general_section_names_rsmeval():
    specified_list = ['data_description', 'preprocessed_features']
    check_section_names(specified_list, 'general', context='rsmeval')


def test_check_general_section_names_rsmcompare():
    specified_list = ['feature_descriptives', 'evaluation']
    check_section_names(specified_list, 'general', context='rsmcompare')


@raises(ValueError)
def test_check_general_section_names_wrong_names():
    specified_list = ['data_description', 'evaluation']
    check_section_names(specified_list, 'general', context='rsmcompare')


def test_determine_chosen_sections_default_general():
    general_sections = ['all']
    special_sections = []
    custom_sections = []
    subgroups = ['prompt']
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups)
    eq_(chosen_sections, general_section_list_rsmtool)

def test_determine_chosen_sections_default_general_no_subgroups():
    general_sections = ['all']
    special_sections = []
    custom_sections = []
    subgroups = []
    no_subgroup_list = [s for s in general_section_list_rsmtool
                        if not s.endswith('by_group')]
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups)
    eq_(chosen_sections, no_subgroup_list)


@raises(ValueError)
def test_determine_chosen_sections_invalid_general():
    general_sections = ['data_description', 'foobar']
    special_sections = []
    custom_sections = []
    subgroups = []
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups)
    eq_(chosen_sections, general_section_list_rsmtool)


@raises(ValueError)
def test_determine_chosen_sections_no_subgroups():
    general_sections = ['data_description', 'data_description_by_group']
    special_sections = []
    custom_sections = []
    subgroups = []
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups)
    eq_(chosen_sections, general_section_list_rsmtool)


def test_determine_chosen_sections_custom_general():
    general_sections = ['data_description', 'evaluation']
    special_sections = []
    custom_sections = []
    subgroups = []
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups)
    eq_(sorted(chosen_sections), sorted(general_sections))


def test_determine_chosen_sections_default_general_with_special():
    general_sections = ['all']
    special_sections = ['placeholder_special_section']
    custom_sections = []
    subgroups = ['prompt']
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups)
    eq_(sorted(chosen_sections),
        sorted(general_section_list_rsmtool + special_sections))


@raises(ValueError)
def test_determine_chosen_sections_invalid_special():
    general_sections = ['all']
    special_sections = ['placeholder_special_section', 'foobar']
    custom_sections = []
    subgroups = ['prompt']
    _ = determine_chosen_sections(general_sections,
                                  special_sections,
                                  custom_sections,
                                  subgroups)


def test_determine_chosen_sections_custom_general_with_special():
    general_sections = ['data_description', 'evaluation']
    special_sections = ['placeholder_special_section']
    custom_sections = []
    subgroups = []
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups)
    eq_(sorted(chosen_sections),
        sorted(general_sections + special_sections))


def test_determine_chosen_sections_default_general_with_subgroups():
    general_sections = ['all']
    special_sections = []
    custom_sections = []
    subgroups = ['prompt', 'gender']
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups)
    eq_(sorted(chosen_sections),
        sorted(general_section_list_rsmtool))


def test_determine_chosen_sections_custom_general_with_special_subgroups_and_custom():
    general_sections = ['evaluation', 'sysinfo', 'evaluation_by_group']
    special_sections = ['placeholder_special_section']
    custom_sections = ['foobar.ipynb']
    subgroups = ['prompt', 'gender']
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups)
    eq_(sorted(chosen_sections),
        sorted(general_sections +
               special_sections +
               ['foobar']))


def test_determine_chosen_sections_eval_default_general():
    general_sections = ['all']
    special_sections = []
    custom_sections = []
    subgroups = ['prompt']
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups,
                                                context='rsmeval')
    eq_(sorted(chosen_sections), sorted(general_section_list_rsmeval))


def test_determine_chosen_sections_eval_custom_general():
    general_sections = ['data_description', 'consistency']
    special_sections = []
    custom_sections = []
    subgroups = []
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups,
                                                context='rsmeval')
    eq_(sorted(chosen_sections), sorted(general_sections))


def test_determine_chosen_sections_eval_default_general_with_no_subgroups():
    general_sections = ['all']
    special_sections = []
    custom_sections = []
    subgroups = []
    no_subgroup_list = [s for s in general_section_list_rsmeval
                        if not s.endswith('by_group')]
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups,
                                                context='rsmeval')
    eq_(sorted(chosen_sections), sorted(no_subgroup_list))


def test_determine_chosen_sections_eval_custom_general_with_special_and_subgroups():
    general_sections = ['data_description', 'consistency', 'data_description_by_group']
    special_sections = ['placeholder_special_section']
    custom_sections = []
    subgroups = ['prompt', 'gender']
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups,
                                                context='rsmeval')
    eq_(sorted(chosen_sections), sorted(general_sections +
                                        special_sections))


def test_determine_chosen_sections_compare_default_general():
    general_sections = ['all']
    special_sections = []
    custom_sections = []
    subgroups = ['prompt']
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups,
                                                context='rsmcompare')
    eq_(sorted(chosen_sections), sorted(general_section_list_rsmcompare))


def test_determine_chosen_sections_rsmcompare_custom_general():
    general_sections = ['feature_descriptives',
                        'evaluation']
    special_sections = []
    custom_sections = []
    subgroups = []
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups,
                                                context='rsmcompare')
    eq_(sorted(chosen_sections), sorted(general_sections))


def test_determine_chosen_sections_rsmcompare_default_general_with_no_subgroups():

    general_sections = ['all']
    special_sections = []
    custom_sections = []
    subgroups = []
    no_subgroup_list = [s for s in general_section_list_rsmcompare
                        if not s.endswith('by_group')]
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups,
                                                context='rsmcompare')
    eq_(sorted(chosen_sections), sorted(no_subgroup_list))


def test_determine_chosen_sections_rsmcompare_custom_general_with_special_and_subgroups():
    general_sections = ['feature_descriptives',
                        'evaluation']
    special_sections = ['placeholder_special_section']
    custom_sections = []
    subgroups = ['prompt', 'gender']
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups,
                                                context='rsmcompare')
    eq_(sorted(chosen_sections), sorted(general_sections +
                                        special_sections))


def test_determine_chosen_sections_eval_custom_general_with_special_subgroups_and_custom():
    general_sections = ['data_description', 'consistency']
    special_sections = ['placeholder_special_section']
    custom_sections = ['foobar.ipynb']
    subgroups = ['prompt', 'gender']
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups,
                                                context='rsmeval')
    eq_(sorted(chosen_sections), sorted(general_sections +
                                        special_sections +
                                        ['foobar']))


def test_determine_chosen_sections_compare_custom_general_with_special_subgroups_and_custom():
    general_sections = ['feature_descriptives',
                        'evaluation']
    special_sections = ['placeholder_special_section']
    custom_sections = ['foobar.ipynb']
    subgroups = ['prompt', 'gender']
    chosen_sections = determine_chosen_sections(general_sections,
                                                special_sections,
                                                custom_sections,
                                                subgroups,
                                                context='rsmcompare')
    eq_(sorted(chosen_sections), sorted(general_sections +
                                        special_sections +
                                        ['foobar']))


def test_get_ordered_notebook_files_default_rsmtool():
    general_sections = ['all']
    notebook_files = get_ordered_notebook_files(general_sections,
                                                model_type='skll',
                                                context='rsmtool')
    no_subgroup_list = [s for s in general_section_list_rsmtool
                        if not s.endswith('by_group')]
    section_list = ['header'] + no_subgroup_list + ['footer']

    # replace model section with skll_model.

    updated_section_list = ['skll_'+sname if sname == 'model' else sname for sname in section_list]
    general_section_plus_extension = [s+'.ipynb' for s in updated_section_list]
    expected_notebook_files = [join(notebook_path, s)
                               for s in
                               general_section_plus_extension]
    eq_(notebook_files, expected_notebook_files)


def test_get_ordered_notebook_files_custom_rsmtool():

    # custom and general sections, custom order and subgroups
    general_sections = ['data_description', 'pca', 'data_description_by_group']
    custom_sections = ['/test_path/custom.ipynb']
    special_sections = ['placeholder_special_section']
    subgroups = ['prompt']
    section_order = ['custom',
                     'data_description',
                     'pca',
                     'data_description_by_group',
                     'placeholder_special_section']
    special_notebook_path = notebook_path_dict['special']['rsmtool']
    notebook_files = get_ordered_notebook_files(general_sections,
                                                custom_sections=custom_sections,
                                                special_sections=special_sections,
                                                section_order=section_order,
                                                subgroups=subgroups,
                                                model_type='skll',
                                                context='rsmtool')

    expected_notebook_files = ([join(notebook_path, 'header.ipynb')] +
                              ['/test_path/custom.ipynb'] +
                              [join(notebook_path, s)+'.ipynb' for s in ['data_description',
                                                                         'pca',
                                                                         'data_description_by_group']] +
                              [join(special_notebook_path, 'placeholder_special_section.ipynb')] +
                              [join(notebook_path, 'footer.ipynb')])
    eq_(notebook_files, expected_notebook_files)


def test_get_ordered_notebook_files_default_rsmeval():
    general_sections = ['all']
    notebook_files = get_ordered_notebook_files(general_sections,
                                                context='rsmeval')
    no_subgroup_list = [s for s in general_section_list_rsmeval
                        if not s.endswith('by_group')]
    section_list = ['header'] + no_subgroup_list + ['footer']

    # replace data_description section with data_description_eval
    updated_section_list = [sname+'_eval' if sname == 'data_description' else sname
                            for sname in section_list]
    general_section_plus_extension = ['{}.ipynb'.format(s) for s in updated_section_list]
    expected_notebook_files = [join(notebook_path_dict['general']['rsmeval'], s)
                               for s in
                               general_section_plus_extension]
    eq_(notebook_files, expected_notebook_files)


def test_get_ordered_notebook_files_custom_rsmeval():

    # custom and general sections, custom order and subgroups

    general_sections = ['evaluation', 'consistency', 'evaluation_by_group']
    custom_sections = ['/test_path/custom.ipynb']
    subgroups = ['prompt']
    section_order = ['evaluation',
                     'consistency',
                     'custom',
                     'evaluation_by_group']
    notebook_path = notebook_path_dict['general']['rsmeval']
    notebook_files = get_ordered_notebook_files(general_sections,
                                                custom_sections=custom_sections,
                                                section_order=section_order,
                                                subgroups=subgroups,
                                                context='rsmeval')

    expected_notebook_files = ([join(notebook_path, 'header.ipynb')] +
                               [join(notebook_path, s)+'.ipynb' for s in ['evaluation',
                                                                         'consistency']] +
                              ['/test_path/custom.ipynb'] +
                              [join(notebook_path, 'evaluation_by_group.ipynb')] +
                              [join(notebook_path, 'footer.ipynb')])
    eq_(notebook_files, expected_notebook_files)


def test_get_ordered_notebook_files_default_rsmcompare():
    general_sections = ['all']
    comparison_notebook_path = notebook_path_dict['general']['rsmcompare']
    notebook_files = get_ordered_notebook_files(general_sections,
                                                context='rsmcompare')
    no_subgroup_list = [s for s in general_section_list_rsmcompare
                        if not s.endswith('by_group')]
    section_list = ['header'] + no_subgroup_list + ['footer']

    general_section_plus_extension = [s+'.ipynb' for s in section_list]
    expected_notebook_files = [join(comparison_notebook_path, s)
                               for s in general_section_plus_extension]
    eq_(notebook_files, expected_notebook_files)


def test_get_ordered_notebook_files_custom_rsmcompare():
    # custom and general sections, custom order and subgroups
    general_sections = ['feature_descriptives',
                        'score_distributions',
                        'features_by_group']
    custom_sections = ['/test_path/custom.ipynb']
    subgroups = ['prompt']
    section_order = ['feature_descriptives',
                     'score_distributions',
                     'custom',
                     'features_by_group']
    comparison_notebook_path = notebook_path_dict['general']['rsmcompare']
    notebook_files = get_ordered_notebook_files(general_sections,
                                                custom_sections=custom_sections,
                                                section_order=section_order,
                                                subgroups=subgroups,
                                                context='rsmcompare')

    expected_notebook_files = ([join(comparison_notebook_path, 'header.ipynb')] +
                               [join(comparison_notebook_path, s)+'.ipynb' for s in ['feature_descriptives',
                                                                          'score_distributions']] +
                              ['/test_path/custom.ipynb'] +
                              [join(comparison_notebook_path, 'features_by_group.ipynb')] +
                              [join(comparison_notebook_path, 'footer.ipynb')])
    eq_(notebook_files, expected_notebook_files)


def test_get_section_file_map_rsmtool():
    special_sections = ['placeholder']
    custom_sections = ['/path/notebook.ipynb']
    section_file_map = get_section_file_map(special_sections,
                                            custom_sections,
                                            model_type='R')
    eq_(section_file_map['model'], join(notebook_path, 'r_model.ipynb'))
    eq_(section_file_map['notebook'], '/path/notebook.ipynb')
    eq_(section_file_map['placeholder'], normpath('special_notebook_path/placeholder.ipynb'))


def test_get_section_file_map_rsmeval():
    special_sections = ['placeholder']
    custom_sections = ['/path/notebook.ipynb']
    section_file_map = get_section_file_map(special_sections,
                                            custom_sections,
                                            context='rsmeval')
    eq_(section_file_map['data_description'], join(notebook_path, 'data_description_eval.ipynb'))
    eq_(section_file_map['notebook'], '/path/notebook.ipynb')
    eq_(section_file_map['placeholder'], normpath('special_notebook_path/placeholder.ipynb'))


def test_get_section_file_map_rsmcompare():
    special_sections = ['placeholder']
    custom_sections = ['/path/notebook.ipynb']
    section_file_map = get_section_file_map(special_sections,
                                            custom_sections,
                                            context='rsmcompare')
    eq_(section_file_map['evaluation'], join(comparison_notebook_path, 'evaluation.ipynb'))
    eq_(section_file_map['notebook'], '/path/notebook.ipynb')
    eq_(section_file_map['placeholder'], normpath('special_notebook_path/placeholder.ipynb'))
