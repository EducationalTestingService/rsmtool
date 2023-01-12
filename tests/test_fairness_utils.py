"""Test functions in rsmtool.fairness_utils."""


from os.path import dirname

import pandas as pd
from nose.tools import assert_almost_equal, eq_, ok_, raises

from rsmtool.fairness_utils import convert_to_ordered_category, get_fairness_analyses

# get the directory containing the tests
test_dir = dirname(__file__)


def test_convert_to_ordered_category():
    values = pd.Series(["a", "a", "b", "b", "b", "c", "c", "d"])
    cat_values = convert_to_ordered_category(values)
    eq_(cat_values.cat.categories[0], "b")


def test_convert_to_ordered_category_several_maximums():
    values = pd.Series(["a_2", "a_3", "a_1", "a_2", "a_3", "a_1", "a_2", "a_3", "a_1"])
    cat_values = convert_to_ordered_category(values)
    eq_(cat_values.cat.categories[0], "a_1")


def test_convert_to_ordered_category_base_category():
    values = pd.Series(["a", "a", "b", "b", "b", "c", "c", "d"])
    cat_values = convert_to_ordered_category(values, "a")
    eq_(cat_values.cat.categories[0], "a")


@raises(ValueError)
def test_convert_to_ordered_category_wrong_base_category():
    values = pd.Series(["a", "a", "b", "b", "b", "c", "c", "d"])
    convert_to_ordered_category(values, "e")


def test_get_fairness_analyses_no_effect():
    df = pd.DataFrame(
        {
            "sc1": [1, 1, 4, 4],
            "raw": [0.9, 0.99, 3.99, 3.9],
            "L1": ["Klingon", "Esperanto", "Klingon", "Esperanto"],
        }
    )
    (model_dict, dc) = get_fairness_analyses(df, "L1", "raw")
    eq_(len(model_dict), 3)
    eq_(len(dc), 4)
    sig_osa = dc["fairness_metrics_by_L1"].loc["sig", "Overall score accuracy"]
    sig_osd = dc["fairness_metrics_by_L1"].loc["sig", "Overall score difference"]
    sig_csd = dc["fairness_metrics_by_L1"].loc["sig", "Conditional score difference"]
    eq_(sig_osa, 1)
    eq_(sig_osd, 1)
    eq_(sig_csd, 1)


def test_get_fairness_analyses_custom_human_score_label():
    df = pd.DataFrame(
        {
            "human": [1, 1, 4, 4],
            "raw": [0.9, 0.99, 3.99, 3.9],
            "L1": ["Klingon", "Esperanto", "Klingon", "Esperanto"],
        }
    )
    (model_dict, dc) = get_fairness_analyses(df, "L1", "raw", "human")
    eq_(len(model_dict), 3)
    eq_(len(dc), 4)
    sig_osa = dc["fairness_metrics_by_L1"].loc["sig", "Overall score accuracy"]
    sig_osd = dc["fairness_metrics_by_L1"].loc["sig", "Overall score difference"]
    sig_csd = dc["fairness_metrics_by_L1"].loc["sig", "Conditional score difference"]
    print(dc["estimates_osa_by_L1"])
    eq_(sig_osa, 1)
    eq_(sig_osd, 1)
    eq_(sig_csd, 1)


def test_gest_fairness_analyses_coefficients():
    df = pd.DataFrame(
        {
            "sc1": [2, 3, 2, 3],
            "raw": [1, 1.5, 3, 3.5],
            "L1": ["Klingon", "Klingon", "Esperanto", "Esperanto"],
        }
    )
    (model_dict, dc) = get_fairness_analyses(df, "L1", "raw")
    coefs_osa = dc["estimates_osa_by_L1"]
    coefs_osd = dc["estimates_osd_by_L1"]
    coefs_csd = dc["estimates_csd_by_L1"]
    assert_almost_equal(coefs_osa.loc["Klingon", "estimate"], 1)
    assert_almost_equal(coefs_osd.loc["Klingon", "estimate"], -2)
    assert_almost_equal(coefs_csd.loc["Klingon", "estimate"], -2)
    assert_almost_equal(coefs_osa.loc["Intercept (Esperanto)", "estimate"], 0.625)
    assert_almost_equal(coefs_osd.loc["Intercept (Esperanto)", "estimate"], 0.75)
    assert_almost_equal(coefs_csd.loc["Intercept (Esperanto)", "estimate"], 1)


def test_get_fairness_analyses_osa_difference():
    df = pd.DataFrame(
        {
            "sc1": [1, 1, 4, 4],
            "raw": [0.9, 5, 3.99, 0.01],
            "L1": ["Klingon", "Esperanto", "Klingon", "Esperanto"],
        }
    )
    (model_dict, dc) = get_fairness_analyses(df, "L1", "raw")
    eq_(len(model_dict), 3)
    eq_(len(dc), 4)
    sig_osa = dc["fairness_metrics_by_L1"].loc["sig", "Overall score accuracy"]
    assert_almost_equal(sig_osa, 0, places=4)


def test_get_fairness_analyses_osd_difference():
    df = pd.DataFrame(
        {
            "sc1": [1, 1, 4, 4],
            "raw": [0.49, 1.5, 3.5, 4.51],
            "L1": ["Klingon", "Esperanto", "Klingon", "Esperanto"],
        }
    )
    (model_dict, dc) = get_fairness_analyses(df, "L1", "raw")
    eq_(len(model_dict), 3)
    eq_(len(dc), 4)
    sig_osa = dc["fairness_metrics_by_L1"].loc["sig", "Overall score accuracy"]
    sig_osd = dc["fairness_metrics_by_L1"].loc["sig", "Overall score difference"]
    eq_(sig_osa, 1)
    assert_almost_equal(sig_osd, 0, places=4)


def test_get_fairness_analyses_csd_difference():
    df = pd.DataFrame(
        {
            "sc1": [1, 1, 1, 1, 4, 4, 4, 4],
            "raw": [1.21, 1.2, 1.21, 1.32, 3.78, 3.91, 3.9, 3.9],
            "L1": [
                "Klingon",
                "Klingon",
                "Klingon",
                "Esperanto",
                "Klingon",
                "Esperanto",
                "Esperanto",
                "Esperanto",
            ],
        }
    )
    (model_dict, dc) = get_fairness_analyses(df, "L1", "raw")
    eq_(len(model_dict), 3)
    eq_(len(dc), 4)
    sig_osa = dc["fairness_metrics_by_L1"].loc["sig", "Overall score accuracy"]
    sig_osd = dc["fairness_metrics_by_L1"].loc["sig", "Overall score difference"]
    sig_csd = dc["fairness_metrics_by_L1"].loc["sig", "Conditional score difference"]
    ok_(sig_osa > 0.5)
    ok_(sig_osd > 0.5)
    assert_almost_equal(sig_csd, 0, places=4)


def test_get_fairness_analyses_custom_reference():
    df = pd.DataFrame(
        {
            "sc1": [1, 1, 4, 4, 3],
            "raw": [0.9, 0.99, 3.99, 3.9, 2.9],
            "L1": ["Klingon", "Esperanto", "Klingon", "Esperanto", "Klingon"],
        }
    )
    (model_dict, dc) = get_fairness_analyses(df, "L1", "raw", base_group="Esperanto")
    eq_(len(model_dict), 3)
    eq_(len(dc), 4)
    base_group = dc["fairness_metrics_by_L1"]["base_category"].values[0]
    intercept = dc["estimates_osa_by_L1"].index.values[0]
    eq_(base_group, "Esperanto")
    eq_(intercept, "Intercept (Esperanto)")
