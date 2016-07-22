"""
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import re
import warnings

# do we have rsmextra installed?
try:
    import rsmextra
except ImportError:
    HAS_RSMEXTRA = False
else:
    HAS_RSMEXTRA = True

from .analysis import (compute_basic_descriptives,
                       compute_percentiles,
                       compute_outliers,
                       compute_pca,
                       correlation_helper,
                       metrics_helper)

from .create_features import (generate_default_specs,
                              find_feature_transformation)

from .input import parse_json_with_comments

from .model import (model_fit_to_dataframe,
                    ols_coefficients_to_dataframe,
                    skll_learner_params_to_dataframe,
                    train_builtin_model)

from .predict import predict_with_model

from .preprocess import (filter_on_column,
                         preprocess_feature,
                         remove_outliers,
                         transform_feature,
                         trim)

from .report import convert_ipynb_to_html, merge_notebooks

from .rsmtool import run_experiment

from .utils import (agreement,
                    partial_correlations,
                    write_experiment_output,
                    write_feature_json)

__all__ = ['compute_basic_descriptives',
           'compute_percentiles',
           'compute_outliers',
           'compute_pca',
           'correlation_helper',
           'metrics_helper',
           'generate_default_specs',
           'find_feature_transformation',
           'parse_json_with_comments',
           'model_fit_to_dataframe',
           'ols_coefficients_to_dataframe',
           'skll_learner_params_to_dataframe',
           'train_builtin_model',
           'predict_with_model',
           'filter_on_column',
           'remove_outliers',
           'preprocess_feature',
           'transform_feature',
           'trim',
           'convert_ipynb_to_html',
           'merge_notebooks',
           'run_experiment',
           'agreement',
           'partial_correlations',
           'write_experiment_output',
           'write_feature_json']

# Make sure that DeprecationWarnings are always shown
# within this package unless we are in test mode in
# which case do not enable them by default.
warnings.filterwarnings('always',
                        category=DeprecationWarning,
                        module='^{0}\.'.format(re.escape(__name__)))
