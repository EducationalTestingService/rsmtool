from os.path import join

from nose.tools import raises
from parameterized import param, parameterized

from rsmtool.test_utils import check_run_experiment, do_run_experiment, test_dir


@parameterized([
    param('lr', 'lr'),
    param('lr-subset-features', 'lr_subset'),
    param('lr-with-feature-subset-file', 'lr_with_feature_subset_file'),
    param('lr-subgroups', 'lr_subgroups', subgroups=['L1']),
    param('lr-with-numeric-subgroup', 'lr_with_numeric_subgroup', subgroups=['ITEM', 'QUESTION']),
    param('lr-with-id-with-leading-zeros', 'lr_with_id_with_leading_zeros', subgroups=['ITEM', 'QUESTION']),
    param('lr-subgroups-with-edge-cases', 'lr_subgroups_with_edge_cases', subgroups=['group_edge_cases']),
    param('lr-missing-values', 'lr_missing_values'),
    param('lr-include-zeros', 'lr_include_zeros'),
    param('lr-with-length', 'lr_with_length'),
    param('lr-subgroups-with-length', 'lr_subgroups_with_length', subgroups=['L1', 'QUESTION']),
    param('lr-with-large-integer-value', 'lr_with_large_integer_value'),
    param('lr-with-missing-length-values', 'lr_with_missing_length_values'),
    param('lr-with-length-zero-sd', 'lr_with_length_zero_sd'),
    param('lr-with-h2', 'lr_with_h2', consistency=True),
    param('lr-subgroups-with-h2', 'lr_subgroups_with_h2', ['L1', 'QUESTION'], consistency=True),
    param('lr-with-h2-include-zeros', 'lr_with_h2_include_zeros', consistency=True),
    param('lr-with-h2-and-length', 'lr_with_h2_and_length', consistency=True),
    param('lr-with-h2-named-sc1', 'lr_with_h2_named_sc1', consistency=True)
])
def test_run_experiment(*args, **kwargs):
    check_run_experiment(*args, **kwargs)


@raises(ValueError)
def test_run_experiment_lr_subset_feature_file_and_feature_file():
    # basic experiment with LinearRegression model and a feature file and
    # also a subset file. This is not allowed and so should raise a ValueError.

    source = 'lr-with-feature-subset-file-and-feature-file'
    experiment_id = 'lr_with_feature_subset_file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_with_sc2_as_feature_name():

    # rsmtool experiment with sc2 used as a feature name
    # when the user also requests h2 analysis using a different
    # column
    source = 'lr-with-sc2-as-feature-name'
    experiment_id = 'lr_with_sc2_as_feature_name'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_with_subgroup_as_feature_name():

    # rsmtool experiment with a subgroup name used as feature
    # when the user also requests subgroup analysis with that subgroup

    source = 'lr-with-subgroup-as-feature-name'
    experiment_id = 'lr_with_subgroup_as_feature_name'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_all_non_numeric_scores():

    # rsmtool experiment with all values for `sc1`
    # being non-numeric and all getting filtered out
    # which should raise an exception

    source = 'lr-with-all-non-numeric-scores'
    experiment_id = 'lr_with_all_non_numeric_scores'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_one_fully_non_numeric_feature():

    # rsmtool experiment with all values for one of the
    # features being non-numeric and all getting filtered out
    # which should raise an exception

    source = 'lr-with-one-fully-non-numeric-feature'
    experiment_id = 'lr_with_one_fully_non_numeric_feature'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_none_flagged():

    # rsmtool experiment where all responses have the bad flag
    # value and so they all get filtered out which should
    # raise an exception

    source = 'lr-with-none-flagged'
    experiment_id = 'lr_with_none_flagged'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_wrong_model_name():

    # rsmtool experiment with incorrect model name
    source = 'wrong-model-name'
    experiment_id = 'wrong_model_name'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
