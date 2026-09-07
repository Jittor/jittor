"""Legacy import alias for the native Python API composition."""

import sys
from jittor._core import api

sys.modules[__name__] = api
