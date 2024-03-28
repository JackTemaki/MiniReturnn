"""
Used by setup.py.
"""

from __future__ import annotations
from pprint import pprint
import os
import sys


VERSION = "0.4+git"

_my_dir = os.path.dirname(os.path.abspath(__file__))
# Use realpath to resolve any symlinks. We want the real root-dir, to be able to check the Git revision.
_root_dir = os.path.dirname(os.path.realpath(_my_dir))


def git_rev_version(git_dir=_root_dir):
    """
    :param str git_dir:
    :param bool long: see :func:`get_version_str`
    :rtype: str
    """
    from returnn.util.basic import git_commit_rev, git_is_dirty

    rev = git_commit_rev(git_dir=git_dir)
    version = VERSION + ".%s" % rev
    if git_is_dirty(git_dir=git_dir):
        version += ".dirty"
    return version


def get_version_str():
    """
    :param bool verbose: print exactly how we end up with some version
    :param bool verbose_error: print only any potential errors
        The format might change in the future.
        We will keep it `SemVer <https://semver.org/>`__ compatible.
    :rtype: str
    """

    if VERSION.endswith("+git"):
        if os.path.exists("%s/.git" % _root_dir):
            try:
                version = git_rev_version(git_dir=_root_dir)
                return version
            except Exception as exc:
                print("Exception while getting Git version:", exc)
                sys.excepthook(*sys.exc_info())
                raise  # no fallback anymore
    else:
        return VERSION
