# -*- coding: utf8 -*-

"""
Various generic utilities, which are shared across different backend engines.
"""

from __future__ import annotations

import shutil
from typing import Generic, TypeVar

import subprocess
from subprocess import CalledProcessError

import h5py
from collections import deque
import inspect
import os
import sys
import shlex
import math
import numpy as np
import re
import time
from multiprocessing import Lock

try:
    import thread
except ImportError:
    import _thread as thread
import threading

import filecmp
import typing
from returnn.log import log
import builtins

# Moved imports
# noinspection PyUnresolvedReferences
from .numbers_dict import NumbersDict

unicode = str
long = int
# noinspection PyShadowingBuiltins
input = builtins.input


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

returnn_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NotSpecified(object):
    """
    This is just a placeholder, to be used as default argument to mark that it is not specified.
    """

    def __str__(self):
        return "NotSpecified"

    def __repr__(self):
        return "<NotSpecified>"

    @classmethod
    def resolve(cls, value, default):
        """
        :param T|NotSpecified|type[NotSpecified] value:
        :param U default:
        :rtype: T|U
        """
        if value is NotSpecified:
            return default
        return value


class OptionalNotImplementedError(NotImplementedError):
    """
    This can optionally be implemented, but it is not required by the API.
    """


def get_checkpoint_filepattern(filepath):
    """
    Removes optional .index or .meta extension

    :param str filepath:
    :return: CheckpointLoader compatible filepattern
    :rtype: str
    """
    if filepath.endswith(".meta"):
        return filepath[: -len(".meta")]
    elif filepath.endswith(".index"):
        return filepath[: -len(".index")]
    elif filepath.endswith(".pt"):
        return filepath[: -len(".pt")]
    return filepath


def sys_cmd_out_lines(s):
    """
    :param str s: shell command
    :rtype: list[str]
    :return: all stdout split by newline. Does not cover stderr.
    Raises CalledProcessError on error.
    """
    p = subprocess.Popen(
        s,
        stdout=subprocess.PIPE,
        shell=True,
        close_fds=True,
        env=dict(os.environ, LANG="en_US.UTF-8", LC_ALL="en_US.UTF-8"),
    )
    stdout = p.communicate()[0]
    stdout = stdout.decode("utf8")
    result = [line.strip() for line in stdout.split("\n")[:-1]]
    p.stdout.close()
    if p.returncode != 0:
        raise CalledProcessError(p.returncode, s, stdout)
    return result


def sys_exec_out(*args, **kwargs):
    """
    :param str args: for subprocess.Popen
    :param kwargs: for subprocess.Popen
    :return: stdout as str (assumes utf8)
    :rtype: str
    """
    from subprocess import Popen, PIPE

    kwargs.setdefault("shell", False)
    p = Popen(args, stdin=PIPE, stdout=PIPE, **kwargs)
    out, _ = p.communicate()
    if p.returncode != 0:
        raise CalledProcessError(p.returncode, args)
    out = unicode_to_str(out)
    return out


def sys_exec_ret_code(*args, **kwargs):
    """
    :param str args: for subprocess.call
    :param kwargs: for subprocess.call
    :return: return code
    :rtype: int
    """
    import subprocess

    res = subprocess.call(args, shell=False, **kwargs)
    valid = kwargs.get("valid", (0, 1))
    if valid is not None:
        if res not in valid:
            raise CalledProcessError(res, args)
    return res


def git_commit_rev(commit="HEAD", git_dir=".", length=None):
    """
    :param str commit:
    :param str git_dir:
    :param int|None length:
    :rtype: str
    """
    if commit is None:
        commit = "HEAD"
    return sys_exec_out("git", "rev-parse", "--short=%i" % length if length else "--short", commit, cwd=git_dir).strip()


def git_is_dirty(git_dir="."):
    """
    :param str git_dir:
    :rtype: bool
    """
    r = sys_exec_ret_code("git", "diff", "--no-ext-diff", "--quiet", "--exit-code", cwd=git_dir)
    if r == 0:
        return False
    if r == 1:
        return True
    assert False, "bad return %i" % r


def git_commit_date(commit="HEAD", git_dir="."):
    """
    :param str commit:
    :param str git_dir:
    :rtype: str
    """
    return (
        sys_exec_out("git", "show", "-s", "--format=%ci", commit, cwd=git_dir)
        .strip()[:-6]
        .replace(":", "")
        .replace("-", "")
        .replace(" ", ".")
    )


def git_describe_head_version(git_dir="."):
    """
    :param str git_dir:
    :rtype: str
    """
    cdate = git_commit_date(git_dir=git_dir)
    rev = git_commit_rev(git_dir=git_dir)
    is_dirty = git_is_dirty(git_dir=git_dir)
    return "%s--git-%s%s" % (cdate, rev, "-dirty" if is_dirty else "")


def describe_returnn_version():
    """
    :rtype: str
    :return: string like "1.20171017.163840+git-ab2a1da"
    """
    from returnn import __long_version__

    return __long_version__


def describe_torch_version():
    """
    :rtype: str
    """
    try:
        # noinspection PyPackageRequirements
        import torch
    except ImportError as exc:
        return "<PyTorch ImportError: %s>" % exc
    try:
        tdir = os.path.dirname(torch.__file__)
    except Exception as e:
        tdir = "<unknown(exception: %r)>" % e
    version = getattr(torch, "__version__", "<unknown version>")
    version += " (%s)" % getattr(torch.version, "git_version", "<unknown git version>")
    try:
        if tdir.startswith("<"):
            git_info = "<unknown-dir>"
        elif os.path.exists(tdir + "/../.git"):
            git_info = "git:" + git_describe_head_version(git_dir=tdir)
        elif "/site-packages/" in tdir:
            git_info = "<site-package>"
        else:
            git_info = "<not-under-git>"
    except Exception as e:
        git_info = "<unknown(git exception: %r)>" % e
    return "%s (%s in %s)" % (version, git_info, tdir)


def eval_shell_env(token):
    """
    :param str token:
    :return: if "$var", looks in os.environ, otherwise return token as is
    :rtype: str
    """
    if token.startswith("$"):
        return os.environ.get(token[1:], "")
    return token


def eval_shell_str(s):
    """
    :type s: str | list[str] | ()->str | list[()->str] | ()->list[str] | ()->list[()->str]
    :rtype: list[str]

    Parses `s` as shell like arguments (via shlex.split) and evaluates shell environment variables
    (:func:`eval_shell_env`).
    `s` or its elements can also be callable. In those cases, they will be called and the returned value is used.
    """
    tokens = []
    if callable(s):
        s = s()
    if isinstance(s, (list, tuple)):
        ls = s
    else:
        assert isinstance(s, (str, unicode))
        ls = shlex.split(s)
    for token in ls:
        if callable(token):
            token = token()
        assert isinstance(token, (str, unicode))
        if token.startswith("$"):
            tokens += eval_shell_str(eval_shell_env(token))
        else:
            tokens += [token]
    return tokens


def hdf5_dimension(filename, dimension):
    """
    :param str filename:
    :param str dimension:
    :rtype: numpy.ndarray|int
    """
    fin = h5py.File(filename, "r")
    if "/" in dimension:
        res = fin["/".join(dimension.split("/")[:-1])].attrs[dimension.split("/")[-1]]
    else:
        res = fin.attrs[dimension]
    fin.close()
    return res


def hdf5_group(filename, dimension):
    """
    :param str filename:
    :param str dimension:
    :rtype: dict[str]
    """
    fin = h5py.File(filename, "r")
    res = {k: fin[dimension].attrs[k] for k in fin[dimension].attrs}
    fin.close()
    return res


def hdf5_shape(filename, dimension):
    """
    :param str filename:
    :param dimension:
    :rtype: tuple[int]
    """
    fin = h5py.File(filename, "r")
    res = fin[dimension].shape
    fin.close()
    return res


def hdf5_strings(handle, name, data):
    """
    :param h5py.File handle:
    :param str name:
    :param numpy.ndarray|list[str] data:
    """
    # noinspection PyBroadException
    try:
        s = max([len(d) for d in data])
        dset = handle.create_dataset(name, (len(data),), dtype="S" + str(s))
        dset[...] = data
    except Exception:
        # noinspection PyUnresolvedReferences
        dt = h5py.special_dtype(vlen=unicode)
        del handle[name]
        dset = handle.create_dataset(name, (len(data),), dtype=dt)
        dset[...] = data


def deep_update_dict_values(d, key, new_value):
    """
    Visits all items in `d`.
    If the value is a dict, it will recursively visit it.

    :param dict[str,T|object|None|dict] d: will update inplace
    :param str key:
    :param T new_value:
    """
    for value in d.values():
        if isinstance(value, dict):
            deep_update_dict_values(value, key=key, new_value=new_value)
    if key in d:
        d[key] = new_value


def terminal_size(file=sys.stdout):
    """
    Returns the terminal size.
    This will probably work on linux only.

    :param io.File file:
    :return: (columns, lines), or (-1,-1)
    :rtype: (int,int)
    """
    import os
    import io

    if not hasattr(file, "fileno"):
        return -1, -1
    try:
        if not os.isatty(file.fileno()):
            return -1, -1
    except (io.UnsupportedOperation, ValueError):
        return -1, -1
    env = os.environ

    # noinspection PyShadowingNames
    def ioctl_gwinsz(fd):
        """
        :param int fd: file descriptor
        :rtype: tuple[int]
        """
        # noinspection PyBroadException
        try:
            import fcntl
            import termios
            import struct

            cr_ = struct.unpack("hh", fcntl.ioctl(fd, termios.TIOCGWINSZ, "1234"))  # noqa
        except Exception:
            return
        return cr_

    cr = ioctl_gwinsz(file.fileno) or ioctl_gwinsz(0) or ioctl_gwinsz(1) or ioctl_gwinsz(2)
    if not cr:
        # noinspection PyBroadException
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_gwinsz(fd)
            os.close(fd)
        except Exception:
            pass
    if not cr:
        cr = (env.get("LINES", 25), env.get("COLUMNS", 80))
    return int(cr[1]), int(cr[0])


def is_tty(file=sys.stdout):
    """
    :param io.File file:
    :rtype: bool
    """
    terminal_width, _ = terminal_size(file=file)
    return terminal_width > 0


def confirm(txt, exit_on_false=False):
    """
    :param str txt: e.g. "Delete everything?"
    :param bool exit_on_false: if True, will call sys.exit(1) if not confirmed
    :rtype: bool
    """
    while True:
        r = input("%s Confirm? [yes/no]" % txt)
        if not r:
            continue
        if r in ["y", "yes"]:
            return True
        if r in ["n", "no"]:
            if exit_on_false:
                sys.exit(1)
            return False
        print("Invalid response %r." % r)


def hms(s):
    """
    :param float|int s: seconds
    :return: e.g. "1:23:45" (hs:ms:secs). see hms_fraction if you want to get fractional seconds
    :rtype: str
    """
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def hms_fraction(s, decimals=4):
    """
    :param float s: seconds
    :param int decimals: how much decimals to print
    :return: e.g. "1:23:45.6789" (hs:ms:secs)
    :rtype: str
    """
    return hms(int(s)) + (("%%.0%if" % decimals) % (s - int(s)))[1:]


def human_size(n, factor=1000, frac=0.8, prec=1):
    """
    :param int|float n:
    :param int factor: for each of the units K, M, G, T
    :param float frac: when to go over to the next bigger unit
    :param int prec: how much decimals after the dot
    :return: human readable size, using K, M, G, T
    :rtype: str
    """
    postfixes = ["", "K", "M", "G", "T"]
    i = 0
    while i < len(postfixes) - 1 and n > (factor ** (i + 1)) * frac:
        i += 1
    if i == 0:
        return str(n)
    return ("%." + str(prec) + "f") % (float(n) / (factor**i)) + postfixes[i]


def human_bytes_size(n, factor=1024, frac=0.8, prec=1):
    """
    :param int|float n:
    :param int factor: see :func:`human_size`. 1024 by default for bytes
    :param float frac: see :func:`human_size`
    :param int prec: how much decimals after the dot
    :return: human readable byte size, using K, M, G, T, with "B" at the end
    :rtype: str
    """
    return human_size(n, factor=factor, frac=frac, prec=prec) + "B"


def _pp_extra_info(obj, depth_limit=3):
    """
    :param object obj:
    :param int depth_limit:
    :return: extra info (if available: len, some items, ...)
    :rtype: str
    """
    if isinstance(obj, np.ndarray):
        return "shape=%r" % (obj.shape,)
    s = []
    if hasattr(obj, "__len__"):
        # noinspection PyBroadException
        try:
            # noinspection PyTypeChecker
            if type(obj) in (str, unicode, list, tuple, dict) and len(obj) <= 5:
                pass  # don't print len in this case
            else:
                s += ["len = %i" % obj.__len__()]  # noqa
        except Exception:
            pass
    if depth_limit > 0 and hasattr(obj, "__getitem__"):
        # noinspection PyBroadException
        try:
            if type(obj) in (str, unicode):
                pass  # doesn't make sense to get sub items here
            else:
                sub_obj = obj.__getitem__(0)  # noqa
                extra_info = _pp_extra_info(sub_obj, depth_limit - 1)
                if extra_info != "":
                    s += ["_[0]: {%s}" % extra_info]
        except Exception:
            pass
    return ", ".join(s)


_pretty_print_limit = 300
_pretty_print_as_bytes = False


def set_pretty_print_default_limit(limit):
    """
    :param int|float limit: use float("inf") to disable
    """
    global _pretty_print_limit
    _pretty_print_limit = limit


def set_pretty_print_as_bytes(as_bytes):
    """
    :param bool as_bytes:
    """
    global _pretty_print_as_bytes
    _pretty_print_as_bytes = as_bytes


def pretty_print(obj, limit=None):
    """
    :param object obj:
    :param int|float limit: use float("inf") to disable. None will use the default, via set_pretty_print_default_limit
    :return: repr(obj), or some shorted version of that, maybe with extra info
    :rtype: str
    """
    if _pretty_print_as_bytes and isinstance(obj, np.ndarray):
        bs = obj.tobytes()
        import gzip

        bs = gzip.compress(bs)
        import base64

        if len(bs) > 57:
            parts = []
            while len(bs) > 0:
                parts.append(bs[:57])
                bs = bs[57:]
                if len(bs) == 0:
                    break
            s = "\n  " + "\n  ".join([repr(base64.encodebytes(bs).strip()) for bs in parts]) + "\n  "
        else:
            s = repr(base64.encodebytes(bs).strip())
        s = "numpy.frombuffer(gzip.decompress(base64.decodebytes(%s)), dtype=%r).reshape(%r)" % (
            s,
            str(obj.dtype),
            obj.shape,
        )
    else:
        s = repr(obj)
    if limit is None:
        limit = _pretty_print_limit
    if len(s) > limit:
        s = s[: int(limit) - 3]
        s += "..."
    extra_info = _pp_extra_info(obj)
    if extra_info != "":
        s += ", " + extra_info
    return s


def progress_bar(complete=1.0, prefix="", suffix="", file=None):
    """
    Prints some progress bar.

    :param float complete: from 0.0 to 1.0
    :param str prefix:
    :param str suffix:
    :param io.TextIOWrapper|typing.TextIO|None file: where to print. stdout by default
    :return: nothing, will print on ``file``
    """
    if file is None:
        file = sys.stdout
    terminal_width, _ = terminal_size(file=file)
    if terminal_width <= 0:
        return
    if complete == 1.0:
        file.write("\r%s" % (terminal_width * " "))
        file.flush()
        file.write("\r")
        file.flush()
        return
    progress = "%.02f%%" % (complete * 100)
    if prefix != "":
        prefix = prefix + " "
    if suffix != "":
        suffix = " " + suffix
    ntotal = terminal_width - len(progress) - len(prefix) - len(suffix) - 4
    bars = "|" * int(complete * ntotal)
    spaces = " " * (ntotal - int(complete * ntotal))
    bar = bars + spaces
    file.write(
        "\r%s" % prefix + "[" + bar[: len(bar) // 2] + " " + progress + " " + bar[len(bar) // 2 :] + "]" + suffix
    )
    file.flush()


class _ProgressBarWithTimeStats:
    """
    Global closure. Used by :func:`progress_bar_with_time`.
    """

    start_time = None
    last_complete = None


def progress_bar_with_time(complete=1.0, prefix="", **kwargs):
    """
    :func:`progress_bar` with additional remaining time estimation.

    :param float complete:
    :param str prefix:
    :param kwargs: passed to :func:`progress_bar`
    :return: nothing
    """
    stats = _ProgressBarWithTimeStats
    if stats.start_time is None:
        stats.start_time = time.time()
        stats.last_complete = complete
    if stats.last_complete > complete:
        stats.start_time = time.time()
    stats.last_complete = complete

    start_elapsed = time.time() - stats.start_time
    if complete > 0:
        total_time_estimated = start_elapsed / complete
        remaining_estimated = total_time_estimated - start_elapsed
        if prefix:
            prefix += ", " + hms(remaining_estimated)
        else:
            prefix = hms(remaining_estimated)
    progress_bar(complete, prefix=prefix, **kwargs)


def better_repr(o):
    """
    The main difference to :func:`repr`: this one is deterministic.
    The orig dict.__repr__ has the order undefined for dict or set.
    For big dicts/sets/lists, add "," at the end to make textual diffs nicer.

    :param object o:
    :rtype: str
    """
    if isinstance(o, list):
        return "[\n%s]" % "".join(map(lambda v: better_repr(v) + ",\n", o))
    if isinstance(o, deque):
        return "deque([\n%s])" % "".join(map(lambda v: better_repr(v) + ",\n", o))
    if isinstance(o, tuple):
        if len(o) == 1:
            return "(%s,)" % o[0]
        return "(%s)" % ", ".join(map(better_repr, o))
    if isinstance(o, dict):
        ls = [better_repr(k) + ": " + better_repr(v) for (k, v) in sorted(o.items())]
        if sum([len(v) for v in ls]) >= 40:
            return "{\n%s}" % "".join([v + ",\n" for v in ls])
        else:
            return "{%s}" % ", ".join(ls)
    if isinstance(o, set):
        return "set([\n%s])" % "".join(map(lambda v: better_repr(v) + ",\n", o))
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return "float('%s')" % repr(o)
    # fallback
    return repr(o)


def simple_obj_repr(obj):
    """
    :return: All self.__init__ args.
    :rtype: str
    """
    return obj.__class__.__name__ + "(%s)" % ", ".join(
        ["%s=%s" % (arg, better_repr(getattr(obj, arg))) for arg in inspect.getfullargspec(obj.__init__).args[1:]]
    )


class ObjAsDict(typing.Mapping[str, object]):
    """
    Wraps up any object as a dict, where the attributes becomes the keys.
    See also :class:`DictAsObj`.
    """

    def __init__(self, obj):
        self.__obj = obj

    def __getitem__(self, item):
        if not isinstance(item, (str, unicode)):
            raise KeyError(item)
        try:
            return getattr(self.__obj, item)
        except AttributeError as e:
            raise KeyError(e)

    def __len__(self):
        return len(vars(self.__obj))

    def __iter__(self):
        return iter(vars(self.__obj))

    def items(self):
        """
        :return: vars(..).items()
        :rtype: set[(str,object)]
        """
        return vars(self.__obj).items()


class DictAsObj:
    """
    Wraps up any dictionary as an object, where the keys becomes the attributes.
    See also :class:`ObjAsDict`.
    """

    def __init__(self, dikt):
        """
        :param dict[str] dikt:
        """
        self.__dict__ = dikt


def obj_diff_str(self, other, **kwargs):
    """
    :param object self:
    :param object other:
    :return: the difference described
    :rtype: str
    """
    diff_list = obj_diff_list(self, other, **kwargs)
    if not diff_list:
        return "No diff."
    return "\n".join(diff_list)


def obj_diff_list(self, other, **kwargs):
    """
    Note that we recurse to a certain degree to the items, but not fully.
    Some differences might just be summarized.

    :param object self:
    :param object other:
    :return: the difference described
    :rtype: list[str]
    """
    # Having them explicitly in kwargs because we cannot use `*` in Python 2,
    # but we do not want to allow them as positional args.
    prefix = kwargs.pop("_prefix", "")
    # If allowed_mapping(self, other),
    # they are assigned to the equal_mapping, if possible to do so unambiguously.
    allowed_mapping = kwargs.pop("allowed_mapping", None)
    equal_map_s2o = kwargs.pop("_equal_map_s2o", None)  # self -> other
    equal_map_o2s = kwargs.pop("_equal_map_o2s", None)  # other -> self
    equal_map_finished = kwargs.pop("_equal_map_finished", False)
    if kwargs:
        raise TypeError("obj_diff_list: invalid kwargs %r" % kwargs)

    if self is None and other is None:
        return []
    if self is None and other is not None:
        return ["%sself is None and other is %r" % (prefix, other)]
    if self is not None and other is None:
        return ["%sother is None and self is %r" % (prefix, self)]
    if type(self) != type(other):
        return ["%stype diff: self is %s but other is %s" % (prefix, type(self).__name__, type(other).__name__)]

    if allowed_mapping:
        if equal_map_s2o is None:
            # Build up the unequal_map, by going through the objects.
            equal_map_s2o, equal_map_o2s = {}, {}
            obj_diff_list(
                self, other, allowed_mapping=allowed_mapping, _equal_map_s2o=equal_map_s2o, _equal_map_o2s=equal_map_o2s
            )
            equal_map_finished = True
    else:
        equal_map_finished = True
    if equal_map_s2o is None or equal_map_o2s is None:
        equal_map_s2o, equal_map_o2s = {}, {}  # simplifies the code below

    sub_kwargs = dict(
        allowed_mapping=allowed_mapping,
        _equal_map_s2o=equal_map_s2o,
        _equal_map_o2s=equal_map_o2s,
        _equal_map_finished=equal_map_finished,
    )

    if isinstance(self, (list, tuple)):
        assert isinstance(other, (list, tuple))
        if len(self) != len(other):
            return ["%slist diff len: len self: %i, len other: %i" % (prefix, len(self), len(other))]
        s = []
        for i, (a, b) in enumerate(zip(self, other)):
            s += obj_diff_list(a, b, _prefix="%s[%i] " % (prefix, i), **sub_kwargs)
        if s:
            return ["%slist diff:" % prefix] + s
        return []

    def _set_diff(a_, b_):
        # assume the values can be sorted
        a_ = sorted(a_)
        b_ = sorted(b_)
        self_diff_ = []
        same_ = []
        for v in a_:
            v_ = equal_map_s2o.get(v, v)
            if v_ in b_:
                same_.append(v)
                b_.remove(v_)
            else:
                self_diff_.append(v)
        other_diff_ = b_
        return self_diff_, same_, other_diff_

    if isinstance(self, set):
        assert isinstance(other, set)
        self_diff, _, other_diff = _set_diff(self, other)
        if len(self_diff) == len(other_diff) == 1:
            # potentially update equal_map
            s = obj_diff_list(list(self_diff)[0], list(other_diff)[0], **sub_kwargs)
            # ignore the potential updated equal_map now for simplicity. we will anyway do a second pass later.
            if len(s) == 1:
                return ["%sset diff value: %s" % (prefix, s[0])]
        s = []
        for key in self_diff:
            s += ["%s  %r not in other" % (prefix, key)]
        for key in other_diff:
            s += ["%s  %r not in self" % (prefix, key)]
        if s:
            return ["%sset diff:" % prefix] + s
        return []

    if isinstance(self, dict):
        assert isinstance(other, dict)
        self_diff, same, other_diff = _set_diff(self.keys(), other.keys())
        if not equal_map_finished and len(self_diff) == len(other_diff) == 1:
            # potentially update equal_map
            obj_diff_list(list(self_diff)[0], list(other_diff)[0], **sub_kwargs)
            # ignore the potential updated equal_map now for simplicity. we will anyway do a second pass later.
        s = []
        for key in self_diff:
            s += ["%s  key %r not in other" % (prefix, key)]
        for key in other_diff:
            s += ["%s  key %r not in self" % (prefix, key)]
        for key in same:
            key_ = equal_map_s2o.get(key, key)
            value_self = self[key]
            value_other = other[key_]
            s += obj_diff_list(value_self, value_other, _prefix="%s[%r] " % (prefix, key), **sub_kwargs)
        if s:
            return ["%sdict diff:" % prefix] + s
        return []

    if isinstance(self, np.ndarray):
        assert isinstance(other, np.ndarray)
        if not np.array_equal(self, other):
            return ["%sself: %r != other: %r" % (prefix, self, other)]
        return []

    if allowed_mapping and self != other and allowed_mapping(self, other):
        if self in equal_map_s2o:
            self = equal_map_s2o[self]
        elif other not in equal_map_o2s:  # don't map multiple times to this
            equal_map_s2o[self] = other
            equal_map_o2s[other] = self

    if self != other:
        return ["%sself: %r != other: %r" % (prefix, self, other)]
    return []


def is_quitting():
    """
    :return: whether we are currently quitting (via :func:`rnn.finalize`)
    :rtype: bool
    """
    import returnn.__main__ as rnn

    if rnn.quit_returnn:  # via rnn.finalize()
        return True
    if getattr(sys, "exited", False):  # set via Debug module when an unexpected SIGINT occurs, or here
        return True
    return False


def interrupt_main():
    """
    Sends :class:`KeyboardInterrupt` to the main thread.

    :return: nothing
    """
    # noinspection PyProtectedMember,PyUnresolvedReferences
    is_main_thread = isinstance(threading.currentThread(), threading._MainThread)
    if is_quitting():  # ignore if we are already quitting
        if is_main_thread:  # strange to get again in main thread
            raise Exception("interrupt_main() from main thread while already quitting")
        # Not main thread. This will just exit the thread.
        sys.exit(1)
    sys.exited = True  # Don't do it twice.
    # noinspection PyProtectedMember,PyUnresolvedReferences
    sys.exited_frame = sys._getframe()
    if is_main_thread:
        raise KeyboardInterrupt
    else:
        thread.interrupt_main()
        sys.exit(1)  # And exit the thread.


def try_run(func, args=(), catch_exc=Exception, default=None):
    """
    :param ((X)->T) func:
    :param tuple args:
    :param type[Exception] catch_exc:
    :param T2 default:
    :return: either ``func()`` or ``default`` if there was some exception
    :rtype: T|T2
    """
    # noinspection PyBroadException
    try:
        return func(*args)
    except catch_exc:
        return default


def class_idx_seq_to_1_of_k(seq, num_classes):
    """
    Basically one_hot.

    :param list[int]|np.ndarray seq:
    :param int num_classes:
    :rtype: np.ndarray
    """
    num_frames = len(seq)
    m = np.zeros((num_frames, num_classes), dtype="float32")
    m[np.arange(num_frames), seq] = 1
    return m


def uniq(seq):
    """
    Like Unix tool uniq. Removes repeated entries.
    See :func:`uniq_generic` for a generic (non-Numpy) version.

    :param numpy.ndarray seq:
    :return: seq
    :rtype: numpy.ndarray
    """
    diffs = np.ones_like(seq)
    diffs[1:] = seq[1:] - seq[:-1]
    idx = diffs.nonzero()
    return seq[idx]


def slice_pad_zeros(x, begin, end, axis=0):
    """
    :param numpy.ndarray x: of shape (..., time, ...)
    :param int begin:
    :param int end:
    :param int axis:
    :return: basically x[begin:end] (with axis==0) but if begin < 0 or end > x.shape[0],
     it will not discard these frames but pad zeros, such that the resulting shape[0] == end - begin.
    :rtype: numpy.ndarray
    """
    assert axis == 0, "not yet fully implemented otherwise"
    pad_left, pad_right = 0, 0
    if begin < 0:
        pad_left = -begin
        begin = 0
    elif begin >= x.shape[axis]:
        return np.zeros((end - begin,) + x.shape[1:], dtype=x.dtype)
    assert end >= begin
    if end > x.shape[axis]:
        pad_right = end - x.shape[axis]
        end = x.shape[axis]
    return np.pad(x[begin:end], [(pad_left, pad_right)] + [(0, 0)] * (x.ndim - 1), mode="constant")


def random_orthogonal(shape, gain=1.0, seed=None):
    """
    Returns a random orthogonal matrix of the given shape.
    Code borrowed and adapted from Keras: https://github.com/fchollet/keras/blob/master/keras/initializers.py
    Reference: Saxe et al., https://arxiv.org/abs/1312.6120
    Related: Unitary Evolution Recurrent Neural Networks, https://arxiv.org/abs/1511.06464

    :param tuple[int] shape:
    :param float gain:
    :param int seed: for Numpy random generator
    :return: random orthogonal matrix
    :rtype: numpy.ndarray
    """
    num_rows = 1
    for dim in shape[:-1]:
        num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_rows, num_cols)
    if seed is not None:
        rnd = np.random.RandomState(seed=seed)
    else:
        rnd = np.random
    a = rnd.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q[: shape[0], : shape[1]]


def parse_orthography_into_symbols(
    orthography, upper_case_special=True, word_based=False, square_brackets_for_specials=True
):
    """
    For Speech.
    Example:
      orthography = "hello [HESITATION] there "
      with word_based == False: returns list("hello ") + ["[HESITATION]"] + list(" there ").
      with word_based == True: returns ["hello", "[HESITATION]", "there"]
    No pre/post-processing such as:
    Spaces are kept as-is. No stripping at begin/end. (E.g. trailing spaces are not removed.)
    No tolower/toupper.
    Doesn't add [BEGIN]/[END] symbols or so.
    Any such operations should be done explicitly in an additional function.
    Anything in []-brackets are meant as special-symbols.
    Also see parse_orthography() which includes some preprocessing.

    :param str orthography: example: "hello [HESITATION] there "
    :param bool upper_case_special: whether the special symbols are always made upper case
    :param bool word_based: whether we split on space and return full words
    :param bool square_brackets_for_specials: handle "[...]"
    :rtype: list[str]
    """
    ret = []
    in_special = 0
    for c in orthography:
        if in_special:
            if c == "[":  # special-special
                in_special += 1
                ret[-1] += "["
            elif c == "]":
                in_special -= 1
                ret[-1] += "]"
            elif upper_case_special:
                ret[-1] += c.upper()
            else:
                ret[-1] += c
        else:  # not in_special
            if square_brackets_for_specials and c == "[":
                in_special = 1
                if ret and not ret[-1]:
                    ret[-1] += c
                else:
                    ret += ["["]
            else:
                if word_based:
                    if c.isspace():
                        ret += [""]
                    else:
                        if not ret:
                            ret += [""]
                        ret[-1] += c
                else:  # not word_based
                    ret += c
    return ret


def parse_orthography(
    orthography, prefix=(), postfix=("[END]",), remove_chars="(){}", collapse_spaces=True, final_strip=True, **kwargs
):
    """
    For Speech. Full processing.
    Example:
      orthography = "hello [HESITATION] there "
      with word_based == False: returns list("hello ") + ["[HESITATION]"] + list(" there") + ["[END]"]
      with word_based == True: returns ["hello", "[HESITATION]", "there", "[END]"]
    Does some preprocessing on orthography and then passes it on to parse_orthography_into_symbols().

    :param str orthography: e.g. "hello [HESITATION] there "
    :param list[str] prefix: will add this prefix
    :param list[str] postfix: will add this postfix
    :param str remove_chars: those chars will just be removed at the beginning
    :param bool collapse_spaces: whether multiple spaces and tabs are collapsed into a single space
    :param bool final_strip: whether we strip left and right
    :param kwargs: passed on to parse_orthography_into_symbols()
    :rtype: list[str]
    """
    for c in remove_chars:
        orthography = orthography.replace(c, "")
    if collapse_spaces:
        orthography = " ".join(orthography.split())
    if final_strip:
        orthography = orthography.strip()
    return list(prefix) + parse_orthography_into_symbols(orthography, **kwargs) + list(postfix)


def json_remove_comments(string, strip_space=True):
    """
    :type string: str
    :param bool strip_space:
    :rtype: str

    via https://github.com/getify/JSON.minify/blob/master/minify_json.py,
    by Gerald Storer, Pradyun S. Gedam, modified by us.
    """
    tokenizer = re.compile('"|(/\\*)|(\\*/)|(//)|\n|\r')
    end_slashes_re = re.compile(r"(\\)*$")

    in_string = False
    in_multi = False
    in_single = False

    new_str = []
    index = 0

    for match in re.finditer(tokenizer, string):

        if not (in_multi or in_single):
            tmp = string[index : match.start()]
            if not in_string and strip_space:
                # replace white space as defined in standard
                tmp = re.sub("[ \t\n\r]+", "", tmp)
            new_str.append(tmp)

        index = match.end()
        val = match.group()

        if val == '"' and not (in_multi or in_single):
            escaped = end_slashes_re.search(string, 0, match.start())

            # start of string or unescaped quote character to end string
            if not in_string or (escaped is None or len(escaped.group()) % 2 == 0):
                in_string = not in_string
            index -= 1  # include " character in next catch
        elif not (in_string or in_multi or in_single):
            if val == "/*":
                in_multi = True
            elif val == "//":
                in_single = True
        elif val == "*/" and in_multi and not (in_string or in_single):
            in_multi = False
        elif val in "\r\n" and not (in_multi or in_string) and in_single:
            in_single = False
        elif not ((in_multi or in_single) or (val in " \r\n\t" and strip_space)):
            new_str.append(val)

    new_str.append(string[index:])
    return "".join(new_str)


def load_json(filename=None, content=None):
    """
    :param str|None filename:
    :param str|None content:
    :rtype: dict[str]
    """
    if content:
        assert not filename
    else:
        content = open(filename).read()
    import json

    content = json_remove_comments(content)
    try:
        json_content = json.loads(content)
    except ValueError as e:
        raise Exception("config looks like JSON but invalid json content, %r" % e)
    return json_content


def collect_class_init_kwargs(cls, only_with_default=False):
    """
    :param type cls: class, where it assumes that kwargs are passed on to base classes
    :param bool only_with_default: if given will only return the kwargs with default values
    :return: set if not with_default, otherwise the dict to the default values
    :rtype: list[str] | dict[str]
    """
    from collections import OrderedDict

    if only_with_default:
        kwargs = OrderedDict()
    else:
        kwargs = []
    for cls_ in inspect.getmro(cls):
        # Check Python function. Could be builtin func or so. Python 2 getargspec does not work in that case.
        if not inspect.ismethod(cls_.__init__) and not inspect.isfunction(cls_.__init__):
            continue
        arg_spec = inspect.getfullargspec(cls_.__init__)
        args = arg_spec.args[1:]  # first arg is self, ignore
        if only_with_default:
            if arg_spec.defaults:
                assert len(arg_spec.defaults) <= len(args)
                args = args[len(args) - len(arg_spec.defaults) :]
                assert len(arg_spec.defaults) == len(args), arg_spec
                for arg, default in zip(args, arg_spec.defaults):
                    kwargs[arg] = default
        else:
            for arg in args:
                if arg not in kwargs:
                    kwargs.append(arg)
    return kwargs


def custom_exec(source, source_filename, user_ns, user_global_ns):
    """
    :param str source:
    :param str source_filename:
    :param dict[str] user_ns:
    :param dict[str] user_global_ns:
    :return: nothing
    """
    if not source.endswith("\n"):
        source += "\n"
    co = compile(source, source_filename, "exec")
    user_global_ns["__package__"] = "returnn"  # important so that imports work
    eval(co, user_global_ns, user_ns)


class FrozenDict(dict):
    """
    Frozen dict.
    """

    def __setitem__(self, key, value):
        raise ValueError("FrozenDict cannot be modified")

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def make_hashable(obj):
    """
    Theano needs hashable objects in some cases, e.g. the properties of Ops.
    This converts all objects as such, i.e. into immutable frozen types.

    :param T|dict|list|tuple obj:
    :rtype: T|FrozenDict|tuple
    """
    if isinstance(obj, dict):
        return FrozenDict([make_hashable(item) for item in obj.items()])
    if isinstance(obj, (list, tuple)):
        return tuple([make_hashable(item) for item in obj])
    if isinstance(obj, (str, unicode, float, int, long)):
        return obj
    if obj is None:
        return obj
    if "tensorflow" in sys.modules:
        import tensorflow as tf

        if isinstance(obj, tf.Tensor):
            return RefIdEq(obj)
    assert False, "don't know how to make hashable: %r (%r)" % (obj, type(obj))


class RefIdEq(Generic[T]):
    """
    Reference to some object (e.g. t.fTensor), but this object is always hashable,
    and uses the `id` of the function for the hash and equality.

    (In case of tf.Tensor, this is for compatibility
     because tf.Tensor.ref() was not available in earlier TF versions.
     However, we also need this for :class:`DictRefKeys`.)
    """

    def __init__(self, obj: T):
        """
        :param obj: for example tf.Tensor
        """
        self.obj = obj

    def __repr__(self):
        return "TensorRef{%r}" % self.obj

    def __eq__(self, other):
        if other is None or not isinstance(other, RefIdEq):
            return False
        return self.obj is other.obj

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return id(self.obj)


def to_bool(v):
    """
    :param int|float|str v: if it is a string, it should represent some integer, or alternatively "true" or "false"
    :rtype: bool
    """
    try:
        return bool(int(v))
    except ValueError:
        pass
    if isinstance(v, (str, unicode)):
        v = v.lower()
        if v in ["true", "yes", "on", "1"]:
            return True
        if v in ["false", "no", "off", "0"]:
            return False
    raise ValueError("to_bool cannot handle %r" % v)


def unicode_to_str(s):
    """
    Tries to decode to utf-8 if input is bytes.
    This function was originally used to distinguish between Python2 and Python3 behavior
    Also see :func:`as_str`.

    :param str|unicode|bytes s:
    :rtype: str
    """
    if isinstance(s, bytes):
        s = s.decode("utf8")
        assert isinstance(s, str)
    assert isinstance(s, str)
    return s


def load_txt_vector(filename):
    """
    Expect line-based text encoding in file.
    We also support Sprint XML format, which has some additional xml header and footer,
    which we will just strip away.

    :param str filename:
    :rtype: list[float]
    """
    return [float(line) for line in open(filename).read().splitlines() if line and not line.startswith("<")]


class CollectionReadCheckCovered:
    """
    Wraps around a dict. It keeps track about all the keys which were read from the dict.
    Via :func:`assert_all_read`, you can check that there are no keys in the dict which were not read.
    The usage is for config dict options, where the user has specified a range of options,
    and where in the code there is usually a default for every non-specified option,
    to check whether all the user-specified options are also used (maybe the user made a typo).
    """

    def __init__(self, collection, truth_value=None):
        """
        :param dict[str] collection:
        :param None|bool truth_value: note: check explicitly for self.truth_value, bool(self) is not the same!
        """
        self.collection = collection
        if truth_value is None:
            truth_value = bool(self.collection)
        self.truth_value = truth_value
        self.got_items = set()

    def __repr__(self):
        return "%s(%r, truth_value=%r)" % (self.__class__.__name__, self.collection, self.truth_value)

    @classmethod
    def from_bool_or_dict(cls, value):
        """
        :param bool|dict[str] value:
        :rtype: CollectionReadCheckCovered
        """
        if isinstance(value, bool):
            return cls(collection={}, truth_value=value)
        if isinstance(value, dict):
            return cls(collection=value)
        raise TypeError("invalid type: %s" % type(value))

    def __getitem__(self, item):
        res = self.collection[item]
        self.got_items.add(item)
        return res

    def get(self, item, default=None):
        """
        :param str item:
        :param T default:
        :rtype: T|typing.Any|None
        """
        try:
            return self[item]
        except KeyError:
            return default

    def __bool__(self):  # Python 3
        return self.truth_value

    __nonzero__ = __bool__  # Python 2

    def __len__(self):
        return len(self.collection)

    def __iter__(self):
        for k in self.collection:
            yield self[k]

    def assert_all_read(self):
        """
        Asserts that all items have been read.
        """
        remaining = set(self.collection).difference(self.got_items)
        assert not remaining, "The keys %r were not read in the collection %r." % (remaining, self.collection)


def which(program):
    """
    Finds `program` in some of the dirs of the PATH env var.

    :param str program: e.g. "python"
    :return: full path, e.g. "/usr/bin/python", or None
    :rtype: str|None
    """
    # noinspection PyShadowingNames
    def is_exe(path):
        """
        :param str path:
        :rtype: str
        """
        return os.path.isfile(path) and os.access(path, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def which_pip():
    """
    :rtype: str
    :return: path to pip for the current Python env
    """
    # Before we look anywhere in PATH, check if there is some pip alongside to the Python executable.
    # This might be more reliable.
    py = sys.executable
    dir_name, basename = py.rsplit("/", 1)
    if basename.startswith("python"):
        postfix = basename[len("python") :]
        pip_path = "%s/pip%s" % (dir_name, postfix)
        if os.path.exists(pip_path):
            return pip_path
    # Generic fallback.
    pip_path = which("pip")
    return pip_path


def pip_install(*pkg_names):
    """
    Install packages via pip for the current Python env.

    :param str pkg_names:
    """
    py = sys.executable
    pip_path = which_pip()
    print("Pip install", *pkg_names)
    in_virtual_env = hasattr(sys, "real_prefix")  # https://stackoverflow.com/questions/1871549/
    cmd = [py, pip_path, "install"]
    if not in_virtual_env:
        cmd += ["--user"]
    cmd += list(pkg_names)
    print("$ %s" % " ".join(cmd))
    subprocess.check_call(cmd, cwd="/")
    _pip_installed_packages.clear()  # force reload


_pip_installed_packages = set()


def pip_check_is_installed(pkg_name):
    """
    :param str pkg_name: without version, e.g. just "tensorflow", or with version, e.g. "tensorflow==1.2.3"
    :rtype: bool
    """
    if not _pip_installed_packages:
        py = sys.executable
        pip_path = which_pip()
        cmd = [py, pip_path, "freeze"]
        for line in sys_exec_out(*cmd).splitlines():
            if line and "==" in line:
                if "==" not in pkg_name:
                    line = line[: line.index("==")]
                _pip_installed_packages.add(line)
    return pkg_name in _pip_installed_packages


def cleanup_env_var_path(env_var, path_prefix):
    """
    :param str env_var: e.g. "LD_LIBRARY_PATH"
    :param str path_prefix:

    Will remove all paths in os.environ[env_var] which are prefixed with path_prefix.
    """
    if env_var not in os.environ:
        return
    ps = os.environ[env_var].split(":")

    def f(p):
        """
        :param str p:
        :rtype: bool
        """
        if p == path_prefix or p.startswith(path_prefix + "/"):
            print("Removing %s from %s." % (p, env_var))
            return False
        return True

    ps = filter(f, ps)
    os.environ[env_var] = ":".join(ps)


def get_login_username():
    """
    :rtype: str
    :return: the username of the current user.
    Use this as a replacement for os.getlogin().
    """
    import os

    if sys.platform == "win32":
        return os.getlogin()
    import pwd

    try:
        return pwd.getpwuid(os.getuid())[0]
    except KeyError:
        # pwd.getpwuid() can throw KeyError: 'getpwuid(): uid not found: 12345'
        # this can happen e.g. in a docker environment with mapped uids unknown to the docker OS
        return str(os.getuid())


def get_temp_dir():
    """
    :rtype: str
    :return: e.g. "/tmp/$USERNAME"
    """
    username = get_login_username()
    for envname in ["TMPDIR", "TEMP", "TMP"]:
        dirname = os.getenv(envname)
        if dirname:
            return "%s/%s" % (dirname, username)
    # /var/tmp should be more persistent than /tmp usually.
    if os.path.exists("/var/tmp"):
        return "/var/tmp/%s" % username
    return "/tmp/%s" % username


def get_cache_dir():
    """
    :return: used to cache non-critical things. by default get_temp_dir. unless you define env RETURNN_CACHE_DIR
    :rtype: str
    """
    if "RETURNN_CACHE_DIR" in os.environ:
        return os.environ["RETURNN_CACHE_DIR"]
    return get_temp_dir()


class LockFile(object):
    """
    Simple lock file.
    """

    def __init__(self, directory, name="lock_file", lock_timeout=1 * 60 * 60):
        """
        :param str directory:
        :param int|float lock_timeout: in seconds
        """
        self.directory = directory
        self.name = name
        self.fd = None
        self.lock_timeout = lock_timeout
        self.lockfile = "%s/%s" % (directory, name)

    def is_old_lockfile(self):
        """
        :return: Whether there is an existing lock file and the existing lock file is old.
        :rtype: bool
        """
        try:
            mtime = os.path.getmtime(self.lockfile)
        except OSError:
            mtime = None
        if mtime and (abs(time.time() - mtime) > self.lock_timeout):
            return True
        return False

    def maybe_remove_old_lockfile(self):
        """
        Removes an existing old lockfile, if there is one.
        """
        if not self.is_old_lockfile():
            return
        print("Removing old lockfile %r (probably crashed proc)." % self.lockfile)
        try:
            os.remove(self.lockfile)
        except OSError as exc:
            print("Remove lockfile exception %r. Ignoring it." % exc)

    def is_locked(self):
        """
        :return: whether there is an active (not old) lockfile
        :rtype: bool
        """
        if self.is_old_lockfile():
            return False
        try:
            return os.path.exists(self.lockfile)
        except OSError:
            return False

    def lock(self):
        """
        Acquires the lock.
        """
        import time
        import errno

        wait_count = 0
        while True:
            # Try to create directory if it does not exist.
            try:
                os.makedirs(self.directory)
            except OSError:
                pass  # Ignore any errors.
            # Now try to create the lock.
            try:
                self.fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return
            except OSError as exc:
                # Possible errors:
                # ENOENT (No such file or directory), e.g. if the directory was deleted.
                # EEXIST (File exists), if the lock already exists.
                if exc.errno not in [errno.ENOENT, errno.EEXIST]:
                    raise  # Other error, so reraise.
            # We did not get the lock.
            # Check if it is a really old one.
            self.maybe_remove_old_lockfile()
            # Wait a bit, and then retry.
            time.sleep(1)
            wait_count += 1
            if wait_count == 10:
                print("Waiting for lock-file: %s" % self.lockfile)

    def unlock(self):
        """
        Releases the lock.
        """
        os.close(self.fd)
        os.remove(self.lockfile)

    def __enter__(self):
        self.lock()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unlock()


def str_is_number(s):
    """
    :param str s: e.g. "1", ".3" or "x"
    :return: whether s can be casted to float or int
    :rtype: bool
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def sorted_values_from_dict(d):
    """
    :param dict[T,V] d:
    :rtype: list[V]
    """
    assert isinstance(d, dict)
    return [v for (k, v) in sorted(d.items())]


def dict_zip(keys, values):
    """
    :param list[T] keys:
    :param list[V] values:
    :rtype: dict[T,V]
    """
    assert len(keys) == len(values)
    return dict(zip(keys, values))


def parse_ld_conf_file(fn):
    """
    Via https://github.com/albertz/system-tools/blob/master/bin/find-lib-in-path.py.

    :param str fn: e.g. "/etc/ld.so.conf"
    :return: list of paths for libs
    :rtype: list[str]
    """
    from glob import glob

    paths = []
    for line in open(fn).read().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("include "):
            for sub_fn in glob(line[len("include ") :]):
                paths.extend(parse_ld_conf_file(sub_fn))
            continue
        paths.append(line)
    return paths


def get_ld_paths():
    """
    To be very correct, see man-page of ld.so.
    And here: https://unix.stackexchange.com/questions/354295/what-is-the-default-value-of-ld-library-path/354296
    Short version, not specific to an executable, in this order:
    - LD_LIBRARY_PATH
    - /etc/ld.so.cache (instead we will parse /etc/ld.so.conf)
    - /lib, /usr/lib (or maybe /lib64, /usr/lib64)
    Via https://github.com/albertz/system-tools/blob/master/bin/find-lib-in-path.py.

    :rtype: list[str]
    :return: list of paths to search for libs (*.so files)
    """
    paths = []
    if "LD_LIBRARY_PATH" in os.environ:
        paths.extend(os.environ["LD_LIBRARY_PATH"].split(":"))
    if os.path.exists("/etc/ld.so.conf"):
        paths.extend(parse_ld_conf_file("/etc/ld.so.conf"))
    paths.extend(["/lib", "/usr/lib", "/lib64", "/usr/lib64"])
    return paths


def find_lib(lib_name):
    """
    :param str lib_name: without postfix/prefix, e.g. "cudart" or "blas"
    :return: returns full path to lib or None
    :rtype: str|None
    """
    if sys.platform == "darwin":
        prefix = "lib"
        postfix = ".dylib"
    elif sys.platform == "win32":
        prefix = ""
        postfix = ".dll"
    else:
        prefix = "lib"
        postfix = ".so"
    for path in get_ld_paths():
        fn = "%s/%s%s%s" % (path, prefix, lib_name, postfix)
        if os.path.exists(fn):
            return fn
    return None


def read_sge_num_procs(job_id=None):
    """
    From the Sun Grid Engine (SGE), reads the num_proc setting for a particular job.
    If job_id is not provided and the JOB_ID env is set, it will use that instead (i.e. it uses the current job).
    This calls qstat to figure out this setting. There are multiple ways this can go wrong,
    so better catch any exception.

    :param int|None job_id:
    :return: num_proc
    :rtype: int|None
    """
    if not job_id:
        if not os.environ.get("SGE_ROOT"):
            return None
        try:
            # qint.py might overwrite JOB_ID but sets SGE_JOB_ID instead.
            job_id = int(os.environ.get("SGE_JOB_ID") or os.environ.get("JOB_ID") or 0)
        except ValueError as exc:
            raise Exception("read_sge_num_procs: %r, invalid JOB_ID: %r" % (exc, os.environ.get("JOB_ID")))
        if not job_id:
            return None
    from subprocess import Popen, PIPE, CalledProcessError

    sge_cmd = ["qstat", "-j", str(job_id)]
    proc = Popen(sge_cmd, stdout=PIPE)
    stdout, _ = proc.communicate()
    if proc.returncode:
        raise CalledProcessError(proc.returncode, sge_cmd, stdout)
    stdout = stdout.decode("utf8")
    ls = [
        line[len("hard resource_list:") :].strip()
        for line in stdout.splitlines()
        if line.startswith("hard resource_list:")
    ]
    assert len(ls) == 1
    opts = dict([opt.split("=", 1) for opt in ls[0].split(",")])  # noqa
    try:
        return int(opts["num_proc"])
    except ValueError as exc:
        raise Exception(
            "read_sge_num_procs: %r, invalid num_proc %r for job id %i.\nline: %r"
            % (exc, opts["num_proc"], job_id, ls[0])
        )


def get_number_available_cpus():
    """
    :return: number of available CPUs, if we can figure it out
    :rtype: int|None
    """
    if hasattr(os, "sched_getaffinity"):  # Python >=3.4
        return len(os.sched_getaffinity(0))
    try:
        # noinspection PyPackageRequirements,PyUnresolvedReferences
        import psutil

        proc = psutil.Process()
        if hasattr(proc, "cpu_affinity"):
            return len(proc.cpu_affinity())
    except ImportError:
        pass
    if hasattr(os, "sysconf") and "SC_NPROCESSORS_ONLN" in os.sysconf_names:
        return os.sysconf("SC_NPROCESSORS_ONLN")
    if hasattr(os, "cpu_count"):  # Python >=3.4
        return os.cpu_count()  # not quite correct; that are all in the system
    return None


def guess_requested_max_num_threads(log_file=None, fallback_num_cpus=True):
    """
    :param io.File log_file:
    :param bool fallback_num_cpus:
    :rtype: int|None
    """
    try:
        sge_num_procs = read_sge_num_procs()
    except Exception as exc:
        if log_file:
            print("Error while getting SGE num_proc: %r" % exc, file=log_file)
    else:
        if sge_num_procs:
            if log_file:
                print("Use num_threads=%i (but min 2) via SGE num_proc." % sge_num_procs, file=log_file)
            return max(sge_num_procs, 2)
    omp_num_threads = int(os.environ.get("OMP_NUM_THREADS") or 0)
    if omp_num_threads:
        # Minimum of 2 threads, should not hurt.
        if log_file:
            print("Use num_threads=%i (but min 2) via OMP_NUM_THREADS." % omp_num_threads, file=log_file)
        return max(omp_num_threads, 2)
    if fallback_num_cpus:
        return get_number_available_cpus()
    return None


TheanoFlags = {
    key: value for (key, value) in [s.split("=", 1) for s in os.environ.get("THEANO_FLAGS", "").split(",") if s]
}


def try_and_ignore_exception(f):
    """
    Calls ``f``, and ignores any exception.

    :param ()->T f:
    :return: whatever ``f`` returns, or None
    :rtype: T|None
    """
    try:
        return f()
    except Exception as exc:
        print("try_and_ignore_exception: %r failed: %s" % (f, exc))
        sys.excepthook(*sys.exc_info())
        return None


def try_get_stack_frame(depth=1):
    """
    :param int depth:
    :rtype: types.FrameType|None
    :return: caller function name. this is just for debugging
    """
    # noinspection PyBroadException
    try:
        # noinspection PyProtectedMember,PyUnresolvedReferences
        frame = sys._getframe(depth + 1)  # one more to count ourselves
        return frame
    except Exception:
        return None


class InfiniteRecursionDetected(Exception):
    """
    Raised when an infinite recursion is detected, by guard_infinite_recursion.
    """


_guard_infinite_recursion_cache = threading.local()


def get_hostname():
    """
    :return: e.g. "cluster-cn-211"
    :rtype: str
    """
    # check_output(["hostname"]).strip().decode("utf8")
    import socket

    return socket.gethostname()


def is_running_on_cluster():
    """
    :return: i6 specific. Whether we run on some of the cluster nodes.
    :rtype: bool
    """
    return get_hostname().startswith("cluster-cn-") or get_hostname().startswith("cn-")


start_time = time.time()


def get_utc_start_time_filename_part():
    """
    :return: string which can be used as part of a filename, which represents the start time of RETURNN in UTC
    :rtype: str
    """
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime(start_time))


def maybe_make_dirs(dirname):
    """
    Creates the directory if it does not yet exist.

    :param str dirname: The path of the directory
    """
    import os

    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except Exception as exc:
            print("maybe_create_folder: exception creating dir:", exc)
            # Maybe a concurrent process, e.g. tf.compat.v1.summary.FileWriter created it in the mean-while,
            # so then it would be ok now if it exists, but fail if it does not exist.
            assert os.path.exists(dirname)


def log_runtime_info_to_dir(path, config):
    """
    This will write multiple logging information into the path.
    It will create returnn.*.log with some meta information,
    as well as copy the used config file.

    :param str path: directory path
    :param returnn.config.Config config:
    """
    import os
    import sys
    import shutil

    try:
        hostname = get_hostname()
        content = [
            "Time: %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
            "Call: %s" % (sys.argv,),
            "Path: %s" % (os.getcwd(),),
            "Hostname: %s" % get_hostname(),
            "PID: %i" % os.getpid(),
            "Returnn: %s" % (describe_returnn_version(),),
            "Config files: %s" % (config.files,),
        ]
        maybe_make_dirs(path)
        log_fn = "%s/returnn.%s.%s.%i.log" % (path, get_utc_start_time_filename_part(), hostname, os.getpid())
        if not os.path.exists(log_fn):
            with open(log_fn, "w") as f:
                f.write("Returnn log file:\n" + "".join(["%s\n" % s for s in content]) + "\n")
        for fn in config.files:
            base_fn = os.path.basename(fn)
            target_fn = "%s/%s" % (path, base_fn)
            if os.path.exists(target_fn):
                continue
            shutil.copy(fn, target_fn)
            comment_prefix = "#"
            with open(target_fn, "a") as f:
                f.write(
                    "\n\n\n"
                    + "".join(
                        ["%s Config-file copied for logging purpose by Returnn.\n" % comment_prefix]
                        + ["%s %s\n" % (comment_prefix, s) for s in content]
                    )
                    + "\n"
                )
    except OSError as exc:
        if "Disk quota" in str(exc):
            print("log_runtime_info_to_dir: Error, cannot write: %s" % exc)
        else:
            raise


# See :func:`maybe_restart_returnn_with_atfork_patch` below for why you might want to use this.
_c_code_patch_atfork = """
#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>
#include <sys/syscall.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// https://stackoverflow.com/questions/46845496/ld-preload-and-linkage
// https://stackoverflow.com/questions/46810597/forkexec-without-atfork-handlers

int pthread_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void)) {
  printf("Ignoring pthread_atfork call!\\n");
  fflush(stdout);
  return 0;
}

int __register_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void)) {
  printf("Ignoring __register_atfork call!\\n");
  fflush(stdout);
  return 0;
}

// Another way to ignore atfork handlers: Override fork.
#ifdef __linux__ // only works on Linux currently
pid_t fork(void) {
  return syscall(SYS_clone, SIGCHLD, 0);
}
#endif

__attribute__((constructor))
void patch_atfork_init() {
  setenv("__RETURNN_ATFORK_PATCHED", "1", 1);
}
"""


def get_patch_atfork_lib():
    """
    :return: path to our patch_atfork lib. see :func:`maybe_restart_returnn_with_atfork_patch`
    :rtype: str
    """
    from .compiling import NativeCodeCompiler

    native = NativeCodeCompiler(base_name="patch_atfork", code_version=2, code=_c_code_patch_atfork, is_cpp=False)
    fn = native.get_lib_filename()
    return fn


def restart_returnn():
    """
    Restarts RETURNN.
    """
    log.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    # https://stackoverflow.com/questions/72335904/simple-way-to-restart-application
    close_all_fds_except({0, 1, 2})
    os.execv(sys.executable, [sys.executable] + sys.argv)
    raise Exception("restart_returnn: execv failed")


def maybe_restart_returnn_with_atfork_patch():
    """
    What we want: subprocess.Popen to always work.
    Problem: It uses fork+exec internally in subprocess_fork_exec, via _posixsubprocess.fork_exec.
    That is a problem because fork can trigger any atfork handlers registered via pthread_atfork,
    and those can crash/deadlock in some cases.

    https://github.com/tensorflow/tensorflow/issues/13802
    https://github.com/xianyi/OpenBLAS/issues/240
    https://trac.sagemath.org/ticket/22021
    https://bugs.python.org/issue31814
    https://stackoverflow.com/questions/46845496/ld-preload-and-linkage
    https://stackoverflow.com/questions/46810597/forkexec-without-atfork-handlers

    The solution here: Just override pthread_atfork, via LD_PRELOAD.
    Note that in some cases, this is not enough (see the SO discussion),
    so we also overwrite fork itself.
    See also tests/test_fork_exec.py for a demo.
    """
    if os.environ.get("__RETURNN_ATFORK_PATCHED") == "1":
        print("Running with patched atfork.")
        return
    if os.environ.get("__RETURNN_TRY_ATFORK_PATCHED") == "1":
        print("Patching atfork did not work! Will continue anyway.")
        return
    lib = get_patch_atfork_lib()
    env = os.environ.copy()
    env["DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"] = lib
    env["__RETURNN_TRY_ATFORK_PATCHED"] = "1"
    print("Restarting Returnn with atfork patch...", sys.executable, sys.argv)
    sys.stdout.flush()
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)
    print("execvpe did not work?")


def close_all_fds_except(except_fds):
    """
    Calls os.closerange except for the given fds.
    Code adopted and extended from multiprocessing.util.close_all_fds_except.

    :param typing.Collection[int] except_fds: usually at least {0,1,2}
    """
    # noinspection PyBroadException
    try:
        max_fd = os.sysconf("SC_OPEN_MAX")
    except Exception:
        max_fd = 256

    except_fds = sorted(list(except_fds) + [-1, max_fd])
    assert except_fds[0] == -1 and except_fds[-1] == max_fd, "fd invalid"

    for i in range(len(except_fds) - 1):
        if except_fds[i] + 1 < except_fds[i + 1]:
            os.closerange(except_fds[i] + 1, except_fds[i + 1])


class Stats:
    """
    Collects mean and variance, running average.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self, format_str=None):
        """
        :param None|((float|numpy.ndarray)->str) format_str:
        """
        self.format_str = format_str or str
        self.mean = 0.0
        self.mean_sq = 0.0
        self.var = 0.0
        self.min = None
        self.max = None
        self.total_data_len = 0
        self.num_seqs = 0

    def __str__(self):
        if self.num_seqs > 0:
            if self.num_seqs == self.total_data_len:
                extra_str = "avg_data_len=1"
            else:
                extra_str = "total_data_len=%i, avg_data_len=%f" % (
                    self.total_data_len,
                    float(self.total_data_len) / self.num_seqs,
                )
            return "Stats(mean=%s, std_dev=%s, min=%s, max=%s, num_seqs=%i, %s)" % (
                self.format_str(self.get_mean()),
                self.format_str(self.get_std_dev()),
                self.format_str(self.min),
                self.format_str(self.max),
                self.num_seqs,
                extra_str,
            )
        return "Stats(num_seqs=0)"

    def collect(self, data):
        """
        :param numpy.ndarray|list[int]|list[float] data: shape (time, dim) or (time,)
        """
        import numpy

        if isinstance(data, (list, tuple)):
            data = numpy.array(data)
        assert isinstance(data, numpy.ndarray)
        assert data.ndim >= 1
        if data.shape[0] == 0:
            return
        self.num_seqs += 1
        data_min = numpy.min(data, axis=0)
        data_max = numpy.max(data, axis=0)
        if self.min is None:
            self.min = data_min
            self.max = data_max
        else:
            self.min = numpy.minimum(self.min, data_min)
            self.max = numpy.maximum(self.max, data_max)
        new_total_data_len = self.total_data_len + data.shape[0]
        mean_diff = numpy.mean(data, axis=0) - self.mean
        m_a = self.var * self.total_data_len
        m_b = numpy.var(data, axis=0) * data.shape[0]
        m2 = m_a + m_b + mean_diff**2 * self.total_data_len * data.shape[0] / new_total_data_len
        self.var = m2 / new_total_data_len
        data_sum = numpy.sum(data, axis=0)
        delta = data_sum - self.mean * data.shape[0]
        self.mean += delta / new_total_data_len
        delta_sq = numpy.sum(data * data, axis=0) - self.mean_sq * data.shape[0]
        self.mean_sq += delta_sq / new_total_data_len
        self.total_data_len = new_total_data_len

    def get_mean(self):
        """
        :return: mean, shape (dim,)
        :rtype: numpy.ndarray
        """
        assert self.num_seqs > 0
        return self.mean

    def get_std_dev(self):
        """
        :return: std dev, shape (dim,)
        :rtype: numpy.ndarray
        """
        import numpy

        assert self.num_seqs > 0
        return numpy.sqrt(self.var)
        # return numpy.sqrt(self.mean_sq - self.mean * self.mean)

    def dump(self, output_file_prefix=None, stream=None, stream_prefix=""):
        """
        :param str|None output_file_prefix: if given, will numpy.savetxt mean|std_dev to disk
        :param str stream_prefix:
        :param io.TextIOBase stream: sys.stdout by default
        """
        if stream is None:
            stream = sys.stdout
        import numpy

        print("%sStats:" % stream_prefix, file=stream)
        if self.num_seqs != self.total_data_len:
            print(
                "  %i seqs, %i total frames, %f average frames"
                % (self.num_seqs, self.total_data_len, self.total_data_len / float(self.num_seqs)),
                file=stream,
            )
        else:
            print("  %i seqs" % (self.num_seqs,), file=stream)
        print("  Mean: %s" % (self.format_str(self.get_mean()),), file=stream)
        print("  Std dev: %s" % (self.format_str(self.get_std_dev()),), file=stream)
        print("  Min/max: %s / %s" % (self.format_str(self.min), self.format_str(self.max)), file=stream)
        # print("Std dev (naive): %s" % numpy.sqrt(self.mean_sq - self.mean * self.mean), file=stream)
        if output_file_prefix:
            print("  Write mean/std-dev to %s.(mean|std_dev).txt." % (output_file_prefix,), file=stream)
            numpy.savetxt("%s.mean.txt" % output_file_prefix, self.get_mean())
            numpy.savetxt("%s.std_dev.txt" % output_file_prefix, self.get_std_dev())


def is_namedtuple(cls):
    """
    :param T cls: tuple, list or namedtuple type
    :return: whether cls is a namedtuple type
    :rtype: bool
    """
    return issubclass(cls, tuple) and cls is not tuple


# cache for the i6 cache manager
_cf_lock = Lock()
_cf_cache = {}
_cf_msg_printed = False


# generic tempdir cache
_tempdir_cache = {}


def cf(filename):
    """
    Cache manager. i6 specific.

    :return: filename
    :rtype: str
    """
    import os
    from subprocess import check_output

    global _cf_msg_printed
    global _cf_lock
    global _cf_cache
    global _tempdir_cache

    _cf_lock.acquire()

    # first try caching via i6 cf
    if filename in _cf_cache:
        _cf_lock.release()
        return _cf_cache[filename]
    try:
        cached_fn = check_output(["cf", filename]).strip().decode("utf8")
        assert os.path.exists(cached_fn)
        _cf_cache[filename] = cached_fn
        _cf_lock.release()
        return cached_fn
    except (CalledProcessError, FileNotFoundError):
        if not _cf_msg_printed:
            print("i6 cache manager: Error occurred, using internal cache manager", file=log.v3)
            _cf_msg_printed = True

    # otherwise do generic caching
    tmp_root = get_temp_dir()
    if not os.path.exists(tmp_root):
        os.mkdir(tmp_root)

    real_filename = os.path.realpath(filename)
    assert real_filename.startswith("/")
    if filename in _tempdir_cache:
        _cf_lock.release()
        return _tempdir_cache[filename]

    real_filename = os.path.realpath(filename)
    temp_file = os.path.join(tmp_root, real_filename[1:])  # join without root slash
    if os.path.exists(temp_file):
        if filecmp.cmp(real_filename, temp_file) is True:
            _cf_lock.release()
            return temp_file

    print(f"internal cache manager, copy:\n{real_filename} to \n{temp_file}", file=log.v3)
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    shutil.copy(real_filename, temp_file)
    _tempdir_cache[filename] = temp_file

    _cf_lock.release()
    return temp_file


def binary_search_any(cmp, low, high):
    """
    Binary search for a custom compare function.

    :param (int)->int cmp: e.g. cmp(idx) == compare(array[idx], key)
    :param int low: inclusive
    :param int high: exclusive
    :rtype: int|None
    """
    while low < high:
        mid = (low + high) // 2
        r = cmp(mid)
        if r < 0:
            low = mid + 1
        elif r > 0:
            high = mid
        else:
            return mid
    return low
