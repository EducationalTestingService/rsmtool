"""
:author: Jeremy Biggs {jbiggs@ets.org)}
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

# do we have rsmextra installed? we try to import rsmextra here
# to avoid doing this in each module.

import re
import warnings

try:
    import rsmextra # noqa
except ImportError:
    HAS_RSMEXTRA = False
else:
    HAS_RSMEXTRA = True

from .version import __version__

if HAS_RSMEXTRA:
    from rsmextra.version import __version__ as rsmextra_version # noqa
    VERSION_STRING = '%(prog)s {}; rsmextra {}'.format(__version__,
                                                       rsmextra_version)
else:
    VERSION_STRING = '%(prog)s {}'.format(__version__)

from .analyzer import Analyzer  # noqa

from .convert_feature_json import convert_feature_json_file  # noqa

from .comparer import Comparer  # noqa

from .container import DataContainer  # noqa

from .modeler import Modeler  # noqa

from .preprocessor import FeaturePreprocessor  # noqa

from .reader import DataReader  # noqa

from .reporter import Reporter  # noqa

from .writer import DataWriter  # noqa

from .rsmcompare import run_comparison  # noqa

from .rsmeval import run_evaluation  # noqa

from .rsmtool import run_experiment  # noqa

from .rsmpredict import compute_and_save_predictions  # noqa

from .rsmsummarize import run_summary  # noqa

from .utils.metrics import (agreement,  # noqa
                            compute_expected_scores_from_model,
                            partial_correlations)

from .utils.notebook import get_thumbnail_as_html, show_thumbnail  # noqa

# Make sure that DeprecationWarnings are always shown
# within this package unless we are in test mode in
# which case do not enable them by default.
warnings.filterwarnings('always',
                        category=DeprecationWarning,
                        module=r'^{0}\.'.format(re.escape(__name__)))
