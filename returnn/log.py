"""
Provides the main class for logging, :class:`Log`, and some helpers.
"""

from __future__ import annotations

import logging
import os
import sys
import io
import threading
from threading import RLock
import contextlib
import string
import typing

PY3 = sys.version_info[0] >= 3


class Stream:
    """
    Simple stream wrapper, which provides :func:`write` and :func:`flush`.
    """

    # noinspection PyShadowingNames
    def __init__(self, log, lvl):
        """
        :type log: logging.Logger
        :type lvl: int
        """
        self.buf = io.StringIO()
        self.log = log
        self.lvl = lvl
        self.lock = RLock()

    def write(self, msg):
        """
        :param str msg:
        """
        with self.lock:
            if msg == "\n":
                self.flush()
            else:
                self.buf.write(msg)

    def flush(self):
        """
        Flush, i.e. writes to the log.
        """
        with self.lock:
            self.buf.flush()
            self.log.log(self.lvl, self.buf.getvalue())
            self.buf.truncate(0)
            # truncate does not change the current position.
            # In Python 2.7, it incorrectly does. See: https://bugs.python.org/issue30250
            self.buf.seek(0)


class Log:
    """
    The main logging class.
    """

    def __init__(self):
        self.initialized = False
        self.filename = None  # type: typing.Optional[str]
        self.v = None  # type: typing.Optional[typing.List[logging.Logger]]
        self.verbose = None  # type: typing.Optional[typing.List[bool]]
        self.v1 = None  # type: typing.Optional[Stream]
        self.v2 = None  # type: typing.Optional[Stream]
        self.v3 = None  # type: typing.Optional[Stream]
        self.v4 = None  # type: typing.Optional[Stream]
        self.v5 = None  # type: typing.Optional[Stream]
        self._printed_warning_history = set()  # type: typing.Set[str]

    def initialize(self, logs=None, verbosity=None, formatter=None, propagate=False):
        """
        This resets and configures the "returnn" logger.

        :param list[str|logging.Handler] logs: "stdout", "|<pipe-cmd>", "<filename>"|"<filename>$date<ext>".
          "stdout" is always added when propagate=False.
        :param list[int] verbosity: levels 0-5 for the log handlers
        :param list[str] formatter: 'default', 'timed', 'raw' or 'verbose', for the log handlers
        :param bool propagate:
        """
        if formatter is None:
            formatter = []
        if verbosity is None:
            verbosity = []
        if logs is None:
            logs = []
        self.initialized = True
        fmt = {
            "default": logging.Formatter("%(message)s"),
            "timed": logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d,%H:%M:%S.%MS"),
            "raw": logging.Formatter("%(message)s"),
            "verbose": logging.Formatter("%(levelname)s - %(asctime)s %(message)s", datefmt="%Y-%m-%d,%H:%M:%S.%MS"),
        }
        logger = logging.getLogger("returnn")
        # Note on propagation:
        # This is not so clear. By default, the root logger anyway does nothing.
        # However, if you mix RETURNN with other code, which might setup the root logger
        # (e.g. via logging.basicConfig(...)), then there is some root logger,
        # and maybe we should also use it.
        # But at this point here, we might not know about this
        # -- maybe the user would call logging.basicConfig(...) at some later point.
        # In any case, if there is a root logger and we would propagate,
        # we should not add "stdout" here,
        # although that might depend on the root logger level and handlers.
        # For now, the default is to just disable propagation, to keep that separated
        # and avoid any such problems.
        logger.propagate = propagate
        # Reset handler list, in case we have initialized some earlier (e.g. multiple log.initialize() calls).
        logger.handlers = []
        self.v = [logger] * 6  # no need for separate loggers, we do all via log levels
        if "stdout" not in logs and not propagate:
            logs.append("stdout")
        if len(formatter) == 1:
            # if only one format provided, use it for all logs
            formatter = [formatter[0]] * len(logs)
        # Define own level names. In reverse order, such that the name by default still has the default behavior.
        logging.addLevelName(logging.DEBUG + 2, "DEBUG")
        logging.addLevelName(logging.DEBUG + 1, "DEBUG")
        logging.addLevelName(logging.DEBUG + 0, "DEBUG")
        logging.addLevelName(logging.INFO + 1, "INFO")
        logging.addLevelName(logging.INFO + 0, "INFO")
        _VerbosityToLogLevel = {
            0: logging.ERROR,
            1: logging.INFO + 1,
            2: logging.INFO,
            3: logging.DEBUG + 2,
            4: logging.DEBUG + 1,
            5: logging.DEBUG,
        }
        self.verbose = [False] * 6
        for i in range(len(logs)):
            t = logs[i]
            v = 3
            if i < len(verbosity):
                v = verbosity[i]
            elif len(verbosity) == 1:
                v = verbosity[0]
            assert v <= 5, "invalid verbosity: " + str(v)
            for j in range(v + 1):
                self.verbose[j] = True
            f = fmt["default"] if i >= len(formatter) or formatter[i] not in fmt else fmt[formatter[i]]
            if isinstance(t, logging.Handler):
                handler = t
            elif t == "stdout":
                handler = StdoutHandler()
            elif t.startswith("|"):  # pipe-format
                proc_cmd = t[1:].strip()
                from subprocess import Popen, PIPE

                proc = Popen(proc_cmd, shell=True, stdin=PIPE)
                handler = logging.StreamHandler(proc.stdin)
            elif os.path.isdir(os.path.dirname(t)):
                if "$" in t:
                    from returnn.util.basic import get_utc_start_time_filename_part

                    t = string.Template(t).substitute(date=get_utc_start_time_filename_part())
                self.filename = t
                handler = logging.FileHandler(t)
            else:
                assert False, "invalid log target %r" % t
            handler.setLevel(_VerbosityToLogLevel[v])
            handler.setFormatter(f)
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        self.v1 = Stream(self.v[1], _VerbosityToLogLevel[1])
        self.v2 = Stream(self.v[2], _VerbosityToLogLevel[2])
        self.v3 = Stream(self.v[3], _VerbosityToLogLevel[3])
        self.v4 = Stream(self.v[4], _VerbosityToLogLevel[4])
        self.v5 = Stream(self.v[5], _VerbosityToLogLevel[5])

    def init_by_config(self, config):
        """
        :param returnn.config.Config config:
        """
        logs = config.list("log", [])
        log_verbosity = config.int_list("log_verbosity", [])
        log_format = config.list("log_format", [])
        if config.is_true("use_horovod"):
            raise NotImplementedError("Horovod is currently not supported")
        self.initialize(logs=logs, verbosity=log_verbosity, formatter=log_format)

    def print_warning(self, text, prefix_text="WARNING:", extra_text=None):
        """
        Write a warning to log.v2. Does not write repeated warnings.

        :param str text:
        :param str prefix_text:
        :param str|None extra_text:
        """
        if text in self._printed_warning_history:
            return
        self._printed_warning_history.add(text)
        print(prefix_text, text, file=log.v2)
        if extra_text:
            print(extra_text, file=log.v2)

    def print_deprecation_warning(self, text, behavior_version=None):
        """
        Write a deprecation warning to log.v2. Does not write repeated warnings.

        :param str text:
        :param int|None behavior_version: if this deprecation is already covered by a behavior_version check
        """
        if behavior_version:
            behavior_text = "This will be disallowed with behavior_version %d." % behavior_version
        else:
            behavior_text = "This might be disallowed with a future behavior_version."
        self.print_warning(text, prefix_text="DEPRECATION WARNING:", extra_text=behavior_text)

    def flush(self):
        """
        Flush all streams.
        """
        for stream in [self.v1, self.v2, self.v3, self.v4, self.v5]:
            if stream:
                stream.flush()


log = Log()


class StdoutHandler(logging.StreamHandler):
    """
    This class is like a StreamHandler using sys.stdout, but always uses
    whatever sys.stdout is currently set to rather than the value of
    sys.stdout at handler construction time.

    Copied and adopted from logging._StderrHandler.
    """

    @property
    def stream(self):
        """
        stream
        """
        return sys.stdout

    @stream.setter
    def stream(self, stream):
        pass  # ignore
