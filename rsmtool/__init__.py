"""
Set up RSMTool version and global imports.

:author: Jeremy Biggs {jbiggs@ets.org)}
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS

isort:skip_file
"""

import re
import warnings

from .version import __version__

VERSION_STRING = f"%(prog)s {__version__}"

from .rsmcompare import run_comparison  # noqa

from .rsmeval import run_evaluation  # noqa

from .rsmtool import run_experiment  # noqa

from .rsmpredict import compute_and_save_predictions, fast_predict  # noqa

from .rsmsummarize import run_summary  # noqa

from .rsmexplain import generate_explanation  # noqa

__all__ = [
    "run_experiment",
    "run_evaluation",
    "run_comparison",
    "compute_and_save_predictions",
    "fast_predict",
    "run_summary",
    "generate_explanation",
]

# Make sure that DeprecationWarnings are always shown
# within this package unless we are in test mode in
# which case do not enable them by default.
warnings.filterwarnings(
    "always", category=DeprecationWarning, module=r"^{0}\.".format(re.escape(__name__))
)
