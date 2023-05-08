"""
Provides :class:`Config` and some related helpers.
"""

from __future__ import annotations

import contextlib
import sys
import typing
import os


class Config:
    """
    Reads in a python-based config file, and provides access to the key/value items.
    """

    def __init__(self, items=None):
        """
        :param dict[str]|None items: optional initial typed_dict
        """
        self.dict = {}  # type: typing.Dict[str, typing.List[str]]
        self.typed_dict = {}  # :type: typing.Dict[str]  # could be loaded via JSON or so
        self.network_topology_json = None  # type: typing.Optional[str]
        self.files = []
        if items is not None:
            self.typed_dict.update(items)

    def load_file(self, f):
        """
        Reads the configuration parameters from a file and adds them to the inner set of parameters.

        :param string|io.TextIOBase|io.StringIO f:
        """
        if isinstance(f, str):
            assert os.path.isfile(f), "config file not found: %r" % f
            self.files.append(f)
            filename = f
            dirname = os.path.dirname(filename) or "."
            content = open(filename).read()
        else:
            # assume stream-like
            filename = "<config string>"
            dirname = None
            content = f.read()
        content = content.strip()
        if content.startswith("#!") or filename.endswith(".py"):  # assume Python
            if dirname and os.path.exists(f"{dirname}/__init__.py"):
                # It looks like a Python module inside a Python package.
                # Import it as a module.
                import importlib

                basedir = os.path.abspath(dirname)
                while os.path.exists(f"{basedir}/__init__.py"):
                    basedir = os.path.dirname(basedir)
                if basedir not in sys.path:
                    sys.path.insert(0, basedir)
                modname = os.path.relpath(dirname, basedir).replace("/", ".") + "." + os.path.basename(filename)[:-3]
                mod = importlib.import_module(modname)
                self.update(vars(mod))

            else:
                # Directly execute the Python code.
                from returnn.util.basic import custom_exec

                # Operate inplace on ourselves.
                # Also, we want that it's available as the globals() dict, so that defined functions behave well
                # (they would loose the local context otherwise).
                user_ns = self.typed_dict
                # Always overwrite:
                user_ns.update({"config": self, "__file__": filename, "__name__": "__returnn_config__"})
                custom_exec(content, filename, user_ns, user_ns)
            return
        else:
            raise ValueError("Invalid config type, maybe you forgot '#!' in the beginning of your config file?")

    def parse_cmd_args(self, args):
        """
        :param list[str]|tuple[str] args:
        """
        from optparse import OptionParser

        parser = OptionParser()
        parser.add_option("--config", dest="load_config", help="[STRING] load config")
        (options, args) = parser.parse_args(list(args))
        options = vars(options)
        for opt in options.keys():
            if options[opt] is not None:
                if opt == "load_config":
                    self.load_file(options[opt])
                else:
                    self.add_line(opt, options[opt])
        assert len(args) % 2 == 0, "expect (++key, value) config tuples in remaining args: %r" % args
        for i in range(0, len(args), 2):
            key, value = args[i : i + 2]
            assert key[0:2] == "++", "expect key prefixed with '++' in (%r, %r)" % (key, value)
            if value[:2] == "+-":
                value = value[1:]  # otherwise we never could specify things like "++threshold -0.1"
            self.add_line(key=key[2:], value=value)

    def add_line(self, key, value):
        """
        Adds one specific configuration (key,value) pair to the inner set of parameters
        :type key: str
        :type value: str
        """
        if key in self.typed_dict:
            # This is a special case. We overwrite a config value which was typed before.
            # E.g. this could have been loaded via a Python config file.
            # We want to keep the entry in self.typed_dict because there might be functions/lambdas inside
            # the config which require the global variable to be available.
            # See :func:`test_rnn_init_config_py_global_var`.
            value_type = type(self.typed_dict[key])
            if value_type == str:
                pass  # keep as-is
            else:
                try:
                    value = eval(value)
                except SyntaxError:
                    from returnn.log import log

                    print(
                        "WARNING: can't evaluate config param %r to previous type: %s. Keeping as string."
                        % (value, value_type),
                        file=log.v1,
                    )
            self.typed_dict[key] = value
            return
        if value.find(",") > 0:
            value = value.split(",")
        else:
            value = [value]
        if key == "include":
            for f in value:
                self.load_file(f)
        else:
            self.dict[key] = value

    def has(self, key):
        """
        Returns whether the given key is present in the inner set of parameters
        :type key: string
        :rtype: boolean
        :returns True if and only if the given key is in the inner set of parameters
        """
        if key in self.typed_dict:
            return True
        return key in self.dict

    def is_typed(self, key):
        """
        :type key: string
        :rtype: boolean
        :returns True if and only if the value of the given key has a specified data type
        """
        return key in self.typed_dict

    def is_true(self, key, default=False):
        """
        :param str key:
        :param bool default:
        :return: bool(value) if it is set or default
        :rtype: bool
        """
        if self.is_typed(key):
            return bool(self.typed_dict[key])
        return self.bool(key, default=default)

    def is_of_type(self, key, types):
        """
        :param str key:
        :param type|tuple[type] types: for isinstance() check
        :return: whether is_typed(key) is True and isinstance(value, types) is True
        :rtype: bool
        """
        if key in self.typed_dict:
            return isinstance(self.typed_dict[key], types)
        return False

    def get_of_type(self, key, types, default=None):
        """
        :param str key:
        :param type|list[type]|T types: for isinstance() check
        :param T|None default:
        :return: if is_of_type(key, types) is True, returns the value, otherwise default
        :rtype: T
        """
        if self.is_of_type(key, types):
            return self.typed_dict[key]
        return default

    def set(self, key, value):
        """
        :type key: str
        :type value: list[str] | str | int | float | bool | dict | None
        """
        self.typed_dict[key] = value

    def update(self, dikt):
        """
        :type dikt: dict
        """
        for key, value in dikt.items():
            self.set(key, value)

    def value(self, key, default, index=None, list_join_str=","):
        """
        :type key: str
        :type default: T
        :type index: int | None
        :param str list_join_str:
        :rtype: str | T
        """
        if key in self.typed_dict:
            ls = self.typed_dict[key]
            if index is None:
                if isinstance(ls, (list, tuple)):
                    return list_join_str.join([str(v) for v in ls])
                elif ls is None:
                    return default
                else:
                    return str(ls)
            else:
                return str(ls[index])
        if key in self.dict:
            ls = self.dict[key]
            if index is None:
                return list_join_str.join(ls)
            else:
                return ls[index]
        return default

    def typed_value(self, key, default=None, index=None):
        """
        :type key: str
        :type default: T
        :type index: int | None
        :rtype: T | typing.Any
        """
        value = self.typed_dict.get(key, default)
        if index is not None:
            assert isinstance(index, int)
            if isinstance(value, (list, tuple)):
                value = value[index]
            else:
                assert index == 0
        return value

    def opt_typed_value(self, key, default=None):
        """
        :param str key:
        :param T|None default:
        :rtype: T|object|str|None
        """
        if key in self.typed_dict:
            return self.typed_dict[key]
        return self.value(key, default)

    def int(self, key, default, index=0):
        """
        Parses the value of the given key as integer, returning default if not existent
        :type key: str
        :type default: T
        :type index: int
        :rtype: int | T
        """
        if key in self.typed_dict:
            value = self.typed_value(key, default=default, index=index)
            if value is not None:
                assert isinstance(value, int)
            return value
        if key in self.dict:
            return int(self.value(key, default, index))
        return default

    def bool(self, key, default, index=0):
        """
        Parses the value of the given key as boolean, returning default if not existent
        :type key: str
        :type default: T
        :type index: bool
        :rtype: bool | T
        """
        if key in self.typed_dict:
            value = self.typed_value(key, default=default, index=index)
            if isinstance(value, int):
                value = bool(value)
            if value is not None:
                assert isinstance(value, bool)
            return value
        if key not in self.dict:
            return default
        v = str(self.value(key, None, index))
        if not v:
            return default
        from returnn.util.basic import to_bool

        return to_bool(v)

    def bool_or_other(self, key, default, index=0):
        """
        :param str key:
        :param T default:
        :param int index:
        :return: if we have typed value, just as-is. otherwise try to convert to bool. or default if not there.
        :rtype: bool|T|object
        """
        if key in self.typed_dict:
            return self.typed_value(key, default=default, index=index)
        if key not in self.dict:
            return default
        v = str(self.value(key, None, index))
        if not v:
            return default
        from returnn.util.basic import to_bool

        try:
            return to_bool(v)
        except ValueError:
            return v

    def float(self, key, default, index=0):
        """
        Parses the value of the given key as float, returning default if not existent
        :type key: str
        :type default: T
        :type index: int
        :rtype: float | T
        """
        if key in self.typed_dict:
            value = self.typed_value(key, default=default, index=index)
        else:
            value = self.value(key, default, index)
        if value is not None:
            if isinstance(value, str):
                # Special case for float as str. We automatically cast this case.
                # This is also to handle special values such as "inf".
                value = float(value)
            assert isinstance(value, (int, float))
        return value

    def list(self, key, default=None):
        """
        :type key: str
        :type default: T
        :rtype: list[str] | T
        """
        if default is None:
            default = []
        if key in self.typed_dict:
            value = self.typed_value(key, default=default)
            if value is None:
                return default
            if not isinstance(value, (tuple, list)):
                value = [value]
            return list(value)
        if key not in self.dict:
            return default
        return self.dict[key]

    def int_list(self, key, default=None):
        """
        :type key: str
        :type default: T
        :rtype: list[int] | T
        """
        if default is None:
            default = []
        if key in self.typed_dict:
            value = self.typed_value(key, default=default)
            if value is None:
                return default
            if not isinstance(value, (tuple, list)):
                value = [value]
            for x in value:
                assert isinstance(x, int)
            return list(value)
        return [int(x) for x in self.list(key, default)]

    def float_list(self, key, default=None):
        """
        :type key: str
        :type default: T
        :rtype: list[float] | T
        """
        if default is None:
            default = []
        if key in self.typed_dict:
            value = self.typed_value(key, default=default)
            if value is None:
                return default
            if not isinstance(value, (tuple, list)):
                value = [value]
            for x in value:
                assert isinstance(x, (float, int))
            return list(value)
        return [float(x) for x in self.list(key, default)]

    def int_pair(self, key, default=None):
        """
        :param str key:
        :param (int,int)|None default:
        :rtype: (int,int)
        """
        if default is None:
            default = (0, 0)
        if not self.has(key):
            return default
        if key in self.typed_dict:
            value = self.typed_value(key, default=default)
            if not isinstance(value, (tuple, list)):
                value = (value, value)
            assert len(value) == 2
            for x in value:
                assert isinstance(x, int)
            return tuple(value)
        value = self.value(key, "")
        if ":" in value:
            return int(value.split(":")[0]), int(value.split(":")[1])
        else:
            return int(value), int(value)


_global_config = None  # type: typing.Optional[Config]


@contextlib.contextmanager
def global_config_ctx(config: Config):
    """
    sets the config as global config in this context,
    and recovers the original global config afterwards
    """
    global _global_config
    prev_global_config = _global_config
    try:
        set_global_config(config)
        yield
    finally:
        _global_config = prev_global_config


def set_global_config(config):
    """
    Will define the global config, returned by :func:`get_global_config`

    :param Config config:
    """
    global _global_config
    _global_config = config


def get_global_config(raise_exception=True, auto_create=False):
    """
    :param bool raise_exception: if no global config is found, raise an exception, otherwise return None
    :param bool auto_create: if no global config is found, it creates one and returns it
    :rtype: Config|None
    """
    if _global_config:
        return _global_config
    # We are the main process.
    import sys

    main_mod = sys.modules["__main__"]  # should be rnn.py
    if hasattr(main_mod, "config") and isinstance(main_mod.config, Config):
        return main_mod.config
    # Maybe __main__ is not rnn.py, or config not yet loaded.
    # Anyway, try directly. (E.g. for SprintInterface.)
    import returnn.__main__ as rnn

    if isinstance(rnn.config, Config):
        return rnn.config
    if auto_create:
        config = Config()
        set_global_config(config)
        return config
    if raise_exception:
        raise Exception("No global config found.")
    return None
