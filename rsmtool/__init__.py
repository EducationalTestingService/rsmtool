"""
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import warnings
import re

# do we have rsmextra installed?
try:
    import rsmextra
except ImportError:
    HAS_RSMEXTRA = False
else:
    HAS_RSMEXTRA = True

from .rsmtool import run_experiment
__all__ = ['run_experiment']

# Make sure that DeprecationWarnings are always shown
# within this package
warnings.filterwarnings('always',
                        category=DeprecationWarning,
                        module='^{0}\.'.format(re.escape(__name__)))
