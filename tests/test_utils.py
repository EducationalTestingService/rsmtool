
import tempfile
import warnings

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from itertools import count
from nose.tools import assert_equal, eq_, raises
from os import unlink, listdir
from os.path import abspath, dirname, join, relpath
from pandas.util.testing import assert_frame_equal

from rsmtool.utils import (difference_of_standardized_means,
                           float_format_func,
                           int_or_float_format_func,
                           custom_highlighter,
                           bold_highlighter,
                           color_highlighter,
                           int_to_float,
                           convert_to_float,
                           compute_subgroup_plot_params,
                           partial_correlations,
                           parse_json_with_comments,
                           has_files_with_extension,
                           get_output_directory_extension,
                           get_thumbnail_as_html,
                           get_files_as_html,
                           compute_expected_scores_from_model,
                           standardized_mean_difference,
                           quadratic_weighted_kappa)

from sklearn.datasets import make_classification
from sklearn.metrics import cohen_kappa_score
from skll import FeatureSet, Learner

from skll.metrics import kappa

# get the directory containing the tests
test_dir = dirname(__file__)


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
    unlink(filename)

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
    directory = join(test_dir, 'data', 'files')
    result = has_files_with_extension(directory, 'csv')
    eq_(result, True)


def test_has_files_with_extension_false():
    directory = join(test_dir, 'data', 'files')
    result = has_files_with_extension(directory, 'ppt')
    eq_(result, False)


def test_get_output_directory_extension():
    directory = join(test_dir, 'data', 'experiments', 'lr', 'output')
    result = get_output_directory_extension(directory, 'id_1')
    eq_(result, 'csv')


@raises(ValueError)
def test_get_output_directory_extension_error():
    directory = join(test_dir, 'data', 'files')
    get_output_directory_extension(directory, 'id_1')


def test_standardized_mean_difference():

    # test SMD
    expected = 1 / 4
    smd = standardized_mean_difference(8, 9, 4, 4, method='williamson')
    eq_(smd, expected)


def test_standardized_mean_difference_zero_denominator():

    # test SMD with zero denominator
    smd = standardized_mean_difference(3.2, 4.2, 0, 0)
    assert np.isnan(smd)


def test_standardized_mean_difference_zero_difference():

    # test SMD with zero difference between groups
    expected = 0.0
    smd = standardized_mean_difference(4.2, 4.2, 1.1, 1.1, method='williamson')
    eq_(smd, expected)


@raises(ValueError)
def test_standardized_mean_difference_fake_method():

    # test SMD with fake method
    standardized_mean_difference(4.2, 4.2, 1.1, 1.1,
                                 method='foobar')


def test_standardized_mean_difference_pooled():

    expected = 0.8523247028586811
    smd = standardized_mean_difference([8, 4, 6, 3],
                                       [9, 4, 5, 12],
                                       method='pooled',
                                       ddof=0)
    eq_(smd, expected)


def test_standardized_mean_difference_unpooled():

    expected = 1.171700198827415
    smd = standardized_mean_difference([8, 4, 6, 3],
                                       [9, 4, 5, 12],
                                       method='unpooled',
                                       ddof=0)
    eq_(smd, expected)


def test_standardized_mean_difference_johnson():

    expected = 0.9782608695652175
    smd = standardized_mean_difference([8, 4, 6, 3],
                                       [9, 4, 5, 12],
                                       method='johnson',
                                       population_y_true_observed_sd=2.3,
                                       ddof=0)
    eq_(smd, expected)


@raises(ValueError)
def test_standardized_mean_difference_johnson_error():

    standardized_mean_difference([8, 4, 6, 3],
                                 [9, 4, 5, 12],
                                 method='johnson',
                                 ddof=0)


@raises(AssertionError)
def test_difference_of_standardized_means_unequal_lengths():

    difference_of_standardized_means([8, 4, 6, 3],
                                     [9, 4, 5, 12, 17])


@raises(ValueError)
def test_difference_of_standardized_means_with_y_true_mn_but_no_sd():

    difference_of_standardized_means([8, 4, 6, 3],
                                     [9, 4, 5, 12],
                                     population_y_true_observed_mn=4.5)


@raises(ValueError)
def test_difference_of_standardized_means_with_y_true_sd_but_no_mn():

    difference_of_standardized_means([8, 4, 6, 3],
                                     [9, 4, 5, 12],
                                     population_y_true_observed_sd=1.5)


@raises(ValueError)
def test_difference_of_standardized_means_with_y_pred_mn_but_no_sd():

    difference_of_standardized_means([8, 4, 6, 3],
                                     [9, 4, 5, 12],
                                     population_y_pred_mn=4.5)


@raises(ValueError)
def test_difference_of_standardized_means_with_y_pred_sd_but_no_mn():

    difference_of_standardized_means([8, 4, 6, 3],
                                     [9, 4, 5, 12],
                                     population_y_pred_sd=1.5)


def test_difference_of_standardized_means_with_all_values():

    expected = 0.7083333333333336
    y_true, y_pred = np.array([8, 4, 6, 3]), np.array([9, 4, 5, 12])
    diff_std_means = difference_of_standardized_means(y_true, y_pred,
                                                      population_y_true_observed_mn=4.5,
                                                      population_y_pred_mn=5.1,
                                                      population_y_true_observed_sd=1.2,
                                                      population_y_pred_sd=1.8)
    eq_(diff_std_means, expected)


def test_difference_of_standardized_means_with_no_population_info():
    expected = -1.7446361815538174e-16
    y_true, y_pred = (np.array([98, 18, 47, 64, 32, 11, 100]),
                      np.array([94, 42, 54, 12, 92, 10, 77]))
    diff_std_means = difference_of_standardized_means(y_true, y_pred)
    eq_(diff_std_means, expected)


def test_quadratic_weighted_kappa():

    expected_qwk = -0.09210526315789469
    computed_qwk = quadratic_weighted_kappa(np.array([8, 4, 6, 3]),
                                   np.array([9, 4, 5, 12]))
   assert_almost_equal(computed_qwk, expected_qwk)


def test_quadratic_weighted_kappa_discrete_values_match_skll():
    data = (np.array([8, 4, 6, 3]),
            np.array([9, 4, 5, 12]))
    qwk_rsmtool = quadratic_weighted_kappa(data[0], data[1])
    qwk_skll = kappa(data[0], data[1], weights='quadratic')
    npt.assert_almost_equal(qwk_rsmtool, qwk_skll)


def test_quadratic_weighted_kappa_discrete_values_match_sklearn():
    data = (np.array([8, 4, 6, 3]),
            np.array([9, 4, 5, 12]))
    qwk_rsmtool = quadratic_weighted_kappa(data[0], data[1])
    qwk_sklearn = cohen_kappa_score(data[0], data[1],
                                    weights='quadratic',
                                    labels=[3, 4, 5, 6, 7,
                                            8, 9, 10, 11, 12])
    assert_almost_equal(qwk_rsmtool, qwk_sklearn)

@raises(AssertionError)
def test_quadratic_weighted_kappa_error():

    quadratic_weighted_kappa(np.array([8, 4, 6, 3]),
                             np.array([9, 4, 5, 12, 11]))


def test_partial_correlations_with_singular_matrix():

    expected = pd.DataFrame({0: [1.0, -1.0], 1: [-1.0, 1.0]})
    df_singular = pd.DataFrame(np.tile(np.random.randn(100), (2, 1))).T
    assert_frame_equal(partial_correlations(df_singular), expected)


def test_partial_correlations_pinv():

    msg = ('The inverse of the variance-covariance matrix was calculated '
           'using the Moore-Penrose generalized matrix inversion, due to '
           'its determinant being at or very close to zero.')
    df_small_det = pd.DataFrame({'X1': [1.3, 1.2, 1.5, 1.7, 1.8, 1.9, 2.0],
                                 'X2': [1.3, 1.2, 1.5, 1.7001, 1.8, 1.9, 2.0]})

    with warnings.catch_warnings(record=True) as wrn:
        warnings.simplefilter("always")
        partial_correlations(df_small_det)
        eq_(str(wrn[-1].message), msg)


class TestIntermediateFiles:

    def get_files(self, file_format='csv'):
        directory = join(test_dir, 'data', 'output')
        files = sorted([f for f in listdir(directory)
                        if f.endswith(file_format)])
        return files, directory

    def test_get_files_as_html(self):

        files, directory = self.get_files()
        html_string = ("""<li><b>Betas</b>: <a href="{}" download>csv</a></li>"""
                       """<li><b>Eval</b>: <a href="{}" download>csv</a></li>""")

        html_expected = html_string.format(join('..', 'output', files[0]),
                                           join('..', 'output', files[1]))
        html_expected = "".join(html_expected.strip().split())
        html_expected = """<ul><html>""" + html_expected + """</ul></html>"""
        html_result = get_files_as_html(directory, 'lr', 'csv')
        html_result = "".join(html_result.strip().split())
        eq_(html_expected, html_result)

    def test_get_files_as_html_replace_dict(self):

        files, directory = self.get_files()
        html_string = ("""<li><b>THESE BETAS</b>: <a href="{}" download>csv</a></li>"""
                       """<li><b>THESE EVALS</b>: <a href="{}" download>csv</a></li>""")

        replace_dict = {'betas': 'THESE BETAS',
                        'eval': 'THESE EVALS'}
        html_expected = html_string.format(join('..', 'output', files[0]),
                                           join('..', 'output', files[1]))
        html_expected = "".join(html_expected.strip().split())
        html_expected = """<ul><html>""" + html_expected + """</ul></html>"""
        html_result = get_files_as_html(directory, 'lr', 'csv', replace_dict)
        html_result = "".join(item for item in html_result)
        html_result = "".join(html_result.strip().split())
        eq_(html_expected, html_result)


class TestThumbnail:

    def get_result(self, path, id_num='1', other_path=None):

        if other_path is None:
            other_path = path

        # get the expected HTML output

        result = """
        <img id='{}' src='{}'
        onclick='getPicture("{}")'
        title="Click to enlarge">
        </img>
        <style>
        img {{
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            width: 150px;
            cursor: pointer;
        }}
        </style>

        <script>
        function getPicture(picpath) {{
            window.open(picpath, 'Image', resizable=1);
        }};
        </script>""".format(id_num, path, other_path)
        return "".join(result.strip().split())

    def test_convert_to_html(self):

        # simple test of HTML thumbnail conversion

        path = relpath(join(test_dir, 'data', 'figures', 'figure1.svg'))
        image = get_thumbnail_as_html(path, 1)

        clean_image = "".join(image.strip().split())
        clean_thumb = self.get_result(path)

        eq_(clean_image, clean_thumb)

    def test_convert_to_html_with_png(self):

        # simple test of HTML thumbnail conversion
        # with a PNG file instead of SVG

        path = relpath(join(test_dir, 'data', 'figures', 'figure3.png'))
        image = get_thumbnail_as_html(path, 1)

        clean_image = "".join(image.strip().split())
        clean_thumb = self.get_result(path)

        eq_(clean_image, clean_thumb)

    def test_convert_to_html_with_two_images(self):

        # test converting two images to HTML thumbnails

        path1 = relpath(join(test_dir, 'data', 'figures', 'figure1.svg'))
        path2 = relpath(join(test_dir, 'data', 'figures', 'figure2.svg'))

        counter = count(1)
        image = get_thumbnail_as_html(path1, next(counter))
        image = get_thumbnail_as_html(path2, next(counter))

        clean_image = "".join(image.strip().split())
        clean_thumb = self.get_result(path2, 2)

        eq_(clean_image, clean_thumb)

    def test_convert_to_html_with_absolute_path(self):

        # test converting image to HTML with absolute path

        path = relpath(join(test_dir, 'data', 'figures', 'figure1.svg'))
        path_absolute = abspath(path)

        image = get_thumbnail_as_html(path_absolute, 1)

        clean_image = "".join(image.strip().split())
        clean_thumb = self.get_result(path)

        eq_(clean_image, clean_thumb)

    @raises(FileNotFoundError)
    def test_convert_to_html_file_not_found_error(self):

        # test FileNotFound error properly raised

        path = 'random/path/asftesfa/to/figure1.svg'
        get_thumbnail_as_html(path, 1)

    def test_convert_to_html_with_different_thumbnail(self):

        # test converting image to HTML with different thumbnail

        path1 = relpath(join(test_dir, 'data', 'figures', 'figure1.svg'))
        path2 = relpath(join(test_dir, 'data', 'figures', 'figure2.svg'))

        image = get_thumbnail_as_html(path1, 1, path_to_thumbnail=path2)

        clean_image = "".join(image.strip().split())
        clean_thumb = self.get_result(path1, other_path=path2)

        eq_(clean_image, clean_thumb)

    @raises(FileNotFoundError)
    def test_convert_to_html_thumbnail_not_found_error(self):

        # test FileNotFound error properly raised for thumbnail

        path1 = relpath(join(test_dir, 'data', 'figures', 'figure1.svg'))
        path2 = 'random/path/asftesfa/to/figure1.svg'
        _ = get_thumbnail_as_html(path1, 1, path_to_thumbnail=path2)


class TestExpectedScores():

    @classmethod
    def setUpClass(cls):

        # create a dummy train and test feature set
        X, y = make_classification(n_samples=525, n_features=10,
                                   n_classes=5, n_informative=8, random_state=123)
        X_train, y_train = X[:500], y[:500]
        X_test = X[500:]

        train_ids = list(range(1, len(X_train) + 1))
        train_features = [dict(zip(['FEATURE_{}'.format(i + 1) for i in range(X_train.shape[1])], x)) for x in X_train]
        train_labels = list(y_train)

        test_ids = list(range(1, len(X_test) + 1))
        test_features = [dict(zip(['FEATURE_{}'.format(i + 1) for i in range(X_test.shape[1])], x)) for x in X_test]

        cls.train_fs = FeatureSet('train', ids=train_ids, features=train_features, labels=train_labels)
        cls.test_fs = FeatureSet('test', ids=test_ids, features=test_features)

        # train some test SKLL learners that we will use in our tests
        cls.linearsvc = Learner('LinearSVC')
        _ = cls.linearsvc.train(cls.train_fs, grid_search=False)

        cls.svc = Learner('SVC')
        _ = cls.svc.train(cls.train_fs, grid_search=False)

        cls.svc_with_probs = Learner('SVC', probability=True)
        _ = cls.svc_with_probs.train(cls.train_fs, grid_search=False)

    @raises(ValueError)
    def test_wrong_model(self):
        compute_expected_scores_from_model(self.linearsvc, self.test_fs, 0, 4)

    @raises(ValueError)
    def test_svc_model_trained_with_no_probs(self):
        compute_expected_scores_from_model(self.svc, self.test_fs, 0, 4)

    @raises(ValueError)
    def test_wrong_score_range(self):
        compute_expected_scores_from_model(self.svc_with_probs, self.test_fs, 0, 3)

    def test_expected_scores(self):
        computed_predictions = compute_expected_scores_from_model(self.svc_with_probs, self.test_fs, 0, 4)
        assert len(computed_predictions) == len(self.test_fs)
        assert np.all([((prediction >= 0) and (prediction <= 4)) for prediction in computed_predictions])
