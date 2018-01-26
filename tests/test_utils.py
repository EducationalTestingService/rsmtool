
import tempfile
import os

from nose.tools import assert_equal, eq_, raises

from rsmtool.utils import (float_format_func,
                           int_or_float_format_func,
                           custom_highlighter,
                           bold_highlighter,
                           color_highlighter,
                           int_to_float,
                           convert_to_float,
                           compute_subgroup_plot_params,
                           parse_json_with_comments,
                           has_files_with_extension,
                           get_output_directory_extension)


def test_int_to_float():

    eq_(int_to_float(5), 5.0)
    eq_(int_to_float('5'), '5')
    eq_(int_to_float(5.0), 5.0)


def test_convert_to_float():

    eq_(convert_to_float(5), 5.0)
    eq_(convert_to_float('5'), 5.0)
    eq_(convert_to_float(5.0), 5.0)


def test_parse_json_with_comments():

    # Need to add comments
    json_with_comments = ("""{"key1": "value1", /*some comments */\n"""
                          """/*more comments */\n"""
                          """"key2": "value2", "key3": 5}""")

    tempf = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    filename = tempf.name
    tempf.close()

    with open(filename, 'w') as buff:
        buff.write(json_with_comments)

    result = parse_json_with_comments(filename)

    # get rid of the file now that have read it into memory
    os.unlink(filename)

    eq_(result, {'key1': 'value1', 'key2': 'value2', 'key3': 5})


def test_float_format_func_default_prec():
    x = 1 / 3
    ans = '0.333'
    assert_equal(float_format_func(x), ans)


def test_float_format_func_custom_prec():
    x = 1 / 3
    ans = '0.3'
    assert_equal(float_format_func(x, 1), ans)


def test_float_format_func_add_extra_zeros():
    x = 0.5
    ans = '0.500'
    assert_equal(float_format_func(x), ans)


def test_int_or_float_format_func_with_integer_as_float():
    x = 3.0
    ans = '3'
    assert_equal(int_or_float_format_func(x), ans)


def test_int_or_float_format_func_with_float_and_custom_precision():
    x = 1 / 3
    ans = '0.33'
    assert_equal(int_or_float_format_func(x, 2), ans)


def test_custom_highlighter_not_bold_default_values():
    x = 1 / 3
    ans = '0.333'
    assert_equal(custom_highlighter(x), ans)


def test_custom_highlighter_bold_default_values():
    x = -1 / 3
    ans = '<span class="highlight_bold">-0.333</span>'
    assert_equal(custom_highlighter(x), ans)


def test_custom_highlighter_bold_custom_low():
    x = 1 / 3
    ans = '<span class="highlight_bold">0.333</span>'
    assert_equal(custom_highlighter(x, low=0.5), ans)


def test_custom_highlighter_bold_custom_high():
    x = 1 / 3
    ans = '<span class="highlight_bold">0.333</span>'
    assert_equal(custom_highlighter(x, high=0.2), ans)


def test_custom_highlighter_bold_custom_prec():
    x = -1 / 3
    ans = '<span class="highlight_bold">-0.3</span>'
    assert_equal(custom_highlighter(x, prec=1), ans)


def test_custom_highlighter_bold_use_absolute():
    x = -4 / 3
    ans = '<span class="highlight_bold">-1.333</span>'
    assert_equal(custom_highlighter(x, absolute=True), ans)


def test_custom_highlighter_not_bold_custom_low():
    x = -1 / 3
    ans = '-0.333'
    assert_equal(custom_highlighter(x, low=-1), ans)


def test_custom_highlighter_not_bold_custom_high():
    x = 1 / 3
    ans = '0.333'
    assert_equal(custom_highlighter(x, high=0.34), ans)


def test_custom_highlighter_not_bold_custom_prec():
    x = 1 / 3
    ans = '0.3'
    assert_equal(custom_highlighter(x, prec=1), ans)


def test_custom_highlighter_not_bold_use_absolute():
    x = -1 / 3
    ans = '-0.333'
    assert_equal(custom_highlighter(x, absolute=True), ans)


def test_custom_highlighter_not_colored_default_values():
    x = 1 / 3
    ans = '0.333'
    assert_equal(custom_highlighter(x, span_class='color'), ans)


def test_custom_highlighter_color_default_values():
    x = -1 / 3
    ans = '<span class="highlight_color">-0.333</span>'
    assert_equal(custom_highlighter(x, span_class='color'), ans)


def test_bold_highlighter_custom_values_not_bold():
    x = -100.33333
    ans = '-100.3'
    assert_equal(bold_highlighter(x, 100, 101, 1, absolute=True), ans)


def test_bold_highlighter_custom_values_bold():
    x = -100.33333
    ans = '<span class="highlight_bold">-100.3</span>'
    assert_equal(bold_highlighter(x, 99, 100, 1, absolute=True), ans)


def test_color_highlighter_custom_values_not_color():
    x = -100.33333
    ans = '-100.3'
    assert_equal(color_highlighter(x, 100, 101, 1, absolute=True), ans)


def test_color_highlighter_custom_values_color():
    x = -100.33333
    ans = '<span class="highlight_color">-100.3</span>'
    assert_equal(color_highlighter(x, 99, 100, 1, absolute=True), ans)


def test_compute_subgroup_params_with_two_groups():
    figure_width = 4
    figure_height = 8
    num_rows, num_cols = 2, 2
    group_names = ['A', 'B']

    expected_subgroup_plot_params = (figure_width, figure_height,
                                     num_rows, num_cols,
                                     group_names)

    subgroup_plot_params = compute_subgroup_plot_params(group_names, 3)
    eq_(expected_subgroup_plot_params, subgroup_plot_params)


def test_compute_subgroup_params_with_10_groups():
    figure_width = 10
    figure_height = 18
    num_rows, num_cols = 3, 1
    group_names = [i for i in range(10)]
    wrapped_group_names = [str(i) for i in group_names]

    expected_subgroup_plot_params = (figure_width, figure_height,
                                     num_rows, num_cols,
                                     wrapped_group_names)

    subgroup_plot_params = compute_subgroup_plot_params(group_names, 3)
    eq_(expected_subgroup_plot_params, subgroup_plot_params)


def test_compute_subgroups_with_wrapping_and_five_plots():
    figure_width = 10
    figure_height = 30
    num_rows, num_cols = 5, 1
    group_names = ['this is a very long string that will '
                   'ultimately be wrapped I assume {}'.format(i)
                   for i in range(10)]

    wrapped_group_names = ['this is a very long\nstring that will\n'
                           'ultimately be\nwrapped I assume {}'.format(i)
                           for i in range(10)]

    expected_subgroup_plot_params = (figure_width, figure_height,
                                     num_rows, num_cols,
                                     wrapped_group_names)

    subgroup_plot_params = compute_subgroup_plot_params(group_names, 5)
    eq_(expected_subgroup_plot_params, subgroup_plot_params)


def test_has_files_with_extension_true():
    directory = 'tests/data/files'
    result = has_files_with_extension(directory, 'csv')
    eq_(result, True)


def test_has_files_with_extension_flase():
    directory = 'tests/data/files'
    result = has_files_with_extension(directory, 'ppt')
    eq_(result, False)


def test_get_output_directory_extension():
    directory = 'tests/data/experiments/lr/output'
    result = get_output_directory_extension(directory, 'id_1')
    eq_(result, 'csv')


@raises(ValueError)
def test_get_output_directory_extension_error():
    directory = 'tests/data/files'
    get_output_directory_extension(directory, 'id_1')
