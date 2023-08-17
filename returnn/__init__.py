"""
The main RETURNN package __init__.
We provide ``__version__`` and ``__long_version__``. See below.

You are supposed to explicitly import the specific sub-module/sub-package.
Just `import returnn` is not enough.

We also provide some helper code to keep older configs compatible,
which used our old-style module names, like ``import TFUtil`` or ``import returnn.TFUtil``.
"""

import os as _os

from .__setup__ import get_version_str as _get_version_str

__long_version__ = _get_version_str()  # `SemVer <https://semver.org/>`__ compatible
if "+" in __long_version__:
    __version__ = __long_version__[: __long_version__.index("+")]  # distutils.version.StrictVersion compatible
else:
    __version__ = __long_version__
__git_version__ = __long_version__  # just an alias, to keep similar to other projects


__root_dir__ = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # can be used as __path__
