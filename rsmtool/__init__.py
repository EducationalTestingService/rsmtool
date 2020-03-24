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

from .rsmcompare import run_comparison  # noqa

from .rsmeval import run_evaluation  # noqa

from .rsmtool import run_experiment  # noqa

from .rsmpredict import compute_and_save_predictions  # noqa

from .rsmsummarize import run_summary  # noqa

__all__ = ['run_experiment', 'run_evaluation', 'run_comparison',
           'compute_and_save_predictions', 'run_summary']

# Make sure that DeprecationWarnings are always shown
# within this package unless we are in test mode in
# which case do not enable them by default.
warnings.filterwarnings('always',
                        category=DeprecationWarning,
                        module=r'^{0}\.'.format(re.escape(__name__)))
