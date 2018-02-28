"""
:author: Jeremy Biggs {jbiggs@ets.org)}
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:date: 01/25/2017
:organization: ETS
"""

# do we have rsmextra installed? we try to import rsmextra here
# to avoid doing this in each module.

import re
import warnings

try:
    import rsmextra
except ImportError:
    HAS_RSMEXTRA = False
else:
    HAS_RSMEXTRA = True

from .version import __version__

if HAS_RSMEXTRA:
    from rsmextra.version import __version__ as rsmextra_version
    VERSION_STRING = '%(prog)s {}; rsmextra {}'.format(__version__,
                                                       rsmextra_version)
else:
    VERSION_STRING = '%(prog)s {}'.format(__version__)

from .analyzer import Analyzer

from .convert_feature_json import convert_feature_json_file

from .comparer import Comparer

from .container import DataContainer

from .modeler import Modeler

from .preprocessor import FeaturePreprocessor

from .reader import DataReader

from .reporter import Reporter

from .writer import DataWriter

from .utils import (agreement,
                    compute_expected_scores_from_model,
                    get_thumbnail_as_html,
                    partial_correlations,
                    show_thumbnail)

from .rsmcompare import run_comparison

from .rsmeval import run_evaluation

from .rsmtool import run_experiment

from .rsmpredict import compute_and_save_predictions

from .rsmsummarize import run_summary


# Make sure that DeprecationWarnings are always shown
# within this package unless we are in test mode in
# which case do not enable them by default.
warnings.filterwarnings('always',
                        category=DeprecationWarning,
                        module='^{0}\.'.format(re.escape(__name__)))
