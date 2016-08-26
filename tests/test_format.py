"""
Unit tests for testing functions in format.py

:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

from nose.tools import assert_equal, assert_raises, eq_, ok_, raises

from rsmtool.format_utils import (float_format_func,
                                  int_or_float_format_func,
                                  custom_highlighter,
                                  bold_highlighter,
                                  color_highlighter)


def test_float_format_func_default_prec():
    x = 1/3
    ans = '0.333'
    assert_equal(float_format_func(x), ans)


def test_float_format_func_custom_prec():
    x = 1/3
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
    x = 1/3
    ans = '0.33'
    assert_equal(int_or_float_format_func(x, 2), ans)


def test_custom_highlighter_not_bold_default_values():
    x = 1/3
    ans = '0.333'
    assert_equal(custom_highlighter(x), ans)


def test_custom_highlighter_bold_default_values():
    x = -1/3
    ans = '<span class="highlight_bold">-0.333</span>'
    assert_equal(custom_highlighter(x), ans)


def test_custom_highlighter_bold_custom_low():
    x = 1/3
    ans = '<span class="highlight_bold">0.333</span>'
    assert_equal(custom_highlighter(x, low=0.5), ans)


def test_custom_highlighter_bold_custom_high():
    x = 1/3
    ans = '<span class="highlight_bold">0.333</span>'
    assert_equal(custom_highlighter(x, high=0.2), ans)


def test_custom_highlighter_bold_custom_prec():
    x = -1/3
    ans = '<span class="highlight_bold">-0.3</span>'
    assert_equal(custom_highlighter(x, prec=1), ans)


def test_custom_highlighter_bold_use_absolute():
    x = -4/3
    ans = '<span class="highlight_bold">-1.333</span>'
    assert_equal(custom_highlighter(x, absolute=True), ans)


def test_custom_highlighter_not_bold_custom_low():
    x = -1/3
    ans = '-0.333'
    assert_equal(custom_highlighter(x, low=-1), ans)


def test_custom_highlighter_not_bold_custom_high():
    x = 1/3
    ans = '0.333'
    assert_equal(custom_highlighter(x, high=0.34), ans)


def test_custom_highlighter_not_bold_custom_prec():
    x = 1/3
    ans = '0.3'
    assert_equal(custom_highlighter(x, prec=1), ans)


def test_custom_highlighter_not_bold_use_absolute():
    x = -1/3
    ans = '-0.333'
    assert_equal(custom_highlighter(x, absolute=True), ans)


def test_custom_highlighter_not_colored_default_values():
    x = 1/3
    ans = '0.333'
    assert_equal(custom_highlighter(x, span_class='color'), ans)


def test_custom_highlighter_color_default_values():
    x = -1/3
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