"""
Main task definitions of RETURNN.

Also keeps track of some global variables (TODO: this can maybe be solved differently)

This is the main entry point, providing :func:`main`.
See :func:`init_config` for some arguments, or just run ``./rnn.py --help``.
"""

from __future__ import annotations


import os
import sys
import time
import typing
from returnn.log import log
from returnn.config import Config
from returnn.datasets import Dataset, init_dataset
from returnn.datasets.hdf import HDFDataset
from returnn.util import debug as debug_util
from returnn.util import basic as util
from returnn.engine.base import EngineBase

# These imports are not directly used here, but make them available, as other code imports them from here.
# noinspection PyUnresolvedReferences
from returnn.util.debug import init_ipython_kernel, init_better_exchook, init_faulthandler

# noinspection PyUnresolvedReferences
from returnn.util.basic import describe_returnn_version

config = None  # type: typing.Optional[Config]
engine = None  # type: typing.Optional[EngineBase]
train_data = None  # type: typing.Optional[Dataset]
dev_data = None  # type: typing.Optional[Dataset]
eval_data = None  # type: typing.Optional[Dataset]
quit_returnn = False


def init_config(config_filename=None, command_line_options=(), default_config=None, extra_updates=None):
    """
    :param str|None config_filename:
    :param list[str]|tuple[str] command_line_options: e.g. ``sys.argv[1:]``
    :param dict[str]|None default_config:
    :param dict[str]|None extra_updates:

    Initializes the global config.
    There are multiple sources which are used to init the config:

      * ``configFilename``, and maybe first item of ``commandLineOptions`` interpret as config filename
      * other options via ``commandLineOptions``
      * ``extra_updates``

    Note about the order/priority of these:

      * ``extra_updates``
      * options from ``commandLineOptions``
      * ``configFilename``
      * config filename from ``commandLineOptions[0]``
      * ``extra_updates``
      * options from ``commandLineOptions``

    ``extra_updates`` and ``commandLineOptions`` are used twice so that they are available
    when the config is loaded, which thus has access to them, and can e.g. use them via Python code.
    However, the purpose is that they overwrite any option from the config;
    that is why we apply them again in the end.

    ``commandLineOptions`` is applied after ``extra_updates`` so that the user has still the possibility
    to overwrite anything set by ``extra_updates``.
    """
    global config
    config = Config()

    config_filenames_by_cmd_line = []
    if command_line_options:
        # Assume that the first argument prefixed with "+" or "-" and all following is not a config file.
        i = 0
        for arg in command_line_options:
            if arg[:1] in "-+":
                break
            config_filenames_by_cmd_line.append(arg)
            i += 1
        command_line_options = command_line_options[i:]

    if default_config:
        config.update(default_config)
    if extra_updates:
        config.update(extra_updates)
    if command_line_options:
        config.parse_cmd_args(command_line_options)
    if config_filename:
        config.load_file(config_filename)
    for fn in config_filenames_by_cmd_line:
        config.load_file(fn)
    if extra_updates:
        config.update(extra_updates)
    if command_line_options:
        config.parse_cmd_args(command_line_options)

    # I really don't know where to put this otherwise:
    if config.bool("EnableAutoNumpySharedMemPickling", False):
        import returnn.util.task_system

        returnn.util.task_system.SharedMemNumpyConfig["enabled"] = True


def init_log():
    """
    Initializes the global :class:`Log`.
    """
    log.init_by_config(config)


# noinspection PyShadowingNames
def load_data(config, files_config_key, **kwargs):
    """
    :param Config config:
    :param str files_config_key: such as "train" or "dev"
    :param kwargs: passed on to init_dataset() or init_dataset_via_str()
    :rtype: (Dataset,int)
    :returns the dataset, and the cache byte size left over if we cache the whole dataset.
    """
    if not config.bool_or_other(files_config_key, None):
        return None, 0
    kwargs = kwargs.copy()
    kwargs.setdefault("name", files_config_key)
    if config.is_typed(files_config_key) and isinstance(config.typed_value(files_config_key), dict):
        config_opts = config.typed_value(files_config_key)
        assert isinstance(config_opts, dict)
        kwargs.update(config_opts)
        data = init_dataset(kwargs)
    elif config.is_typed(files_config_key) and callable(config.typed_value(files_config_key)):
        data = init_dataset(config.typed_value(files_config_key), default_kwargs=kwargs)
    else:
        raise ValueError(
            "A dataset has to be defined as dict or callable, but found %s" % type(config.typed_value(files_config_key))
        )
    cache_leftover = 0
    if isinstance(data, HDFDataset):
        cache_leftover = data.definite_cache_leftover
    return data, cache_leftover


def init_data():
    """
    Initializes the globals train,dev,eval of type Dataset.
    """
    global train_data, dev_data, eval_data
    dev_data, extra_cache_bytes_dev = load_data(config, "dev")
    eval_data, extra_cache_bytes_eval = load_data(config, "eval")
    train_data, extra_train = load_data(config, "train")


def print_task_properties():
    """
    print information about used data
    """
    if train_data:
        print("Train data:", file=log.v2)
        print("  input:", train_data.num_inputs, file=log.v2)
        print("  output:", train_data.num_outputs, file=log.v2)
        print(" ", train_data.len_info() or "no info", file=log.v2)
    if dev_data:
        print("Dev data:", file=log.v2)
        print(" ", dev_data.len_info() or "no info", file=log.v2)
    if eval_data:
        print("Eval data:", file=log.v2)
        print(" ", eval_data.len_info() or "no info", file=log.v2)


def init_engine():
    """
    Initializes global ``engine``, for example :class:`returnn.tf.engine.Engine`.
    """
    global engine, config
    from .torch.engine import Engine

    assert config is not None, "Engine can not be initialized without a config defined"
    if config.is_typed("CustomEngine") and config.is_of_type("CustomEngine", typing.Callable):
        CustomEngine = config.typed_value("CustomEngine")
        print("Using custom engine from config", file=log.v5)
        engine = CustomEngine(config=config)
        assert isinstance(engine, EngineBase)
    else:
        engine = Engine(config=config)


def returnn_greeting(config_filename=None, command_line_options=None):
    """
    Prints some RETURNN greeting to the log.

    :param str|None config_filename:
    :param list[str]|None command_line_options:
    """
    print(
        "RETURNN starting up, version %s, date/time %s, pid %i, cwd %s, Python %s"
        % (
            util.describe_returnn_version(),
            time.strftime("%Y-%m-%d-%H-%M-%S (UTC%z)"),
            os.getpid(),
            os.getcwd(),
            sys.executable,
        ),
        file=log.v3,
    )
    if config_filename:
        print("RETURNN config: %s" % config_filename, file=log.v4)
        if os.path.islink(config_filename):
            print("RETURNN config is symlink to: %s" % os.readlink(config_filename), file=log.v4)
    if command_line_options is not None:
        print("RETURNN command line options: %s" % (command_line_options,), file=log.v4)
    import socket

    print("Hostname:", socket.gethostname(), file=log.v4)


def init(config_filename=None, command_line_options=(), config_updates=None, extra_greeting=None):
    """
    :param str|None config_filename:
    :param tuple[str]|list[str]|None command_line_options: e.g. sys.argv[1:]
    :param dict[str]|None config_updates: see :func:`init_config`
    :param str|None extra_greeting:
    """
    debug_util.init_better_exchook()
    init_config(
        config_filename=config_filename, command_line_options=command_line_options, extra_updates=config_updates
    )
    if config.bool("patch_atfork", False):
        from returnn.util.basic import maybe_restart_returnn_with_atfork_patch

        maybe_restart_returnn_with_atfork_patch()
    init_log()
    if extra_greeting:
        print(extra_greeting, file=log.v1)
    returnn_greeting(config_filename=config_filename, command_line_options=command_line_options)
    debug_util.init_faulthandler()
    init_engine()
    if config.bool("ipython", False):
        debug_util.init_ipython_kernel()
    if need_data():
        init_data()
    print_task_properties()
    init_engine()


def finalize():
    """
    Cleanup at the end, currently doing nothing
    """
    print("Quitting", file=getattr(log, "v4", sys.stderr))
    global quit_returnn
    quit_returnn = True
    sys.exited = True


def need_data():
    """
    :return: whether we need to init the data (call :func:`init_data`) for the current task (:func:`execute_main_task`)
    :rtype: bool
    """
    if config.has("need_data") and not config.bool("need_data", True):
        return False
    task = config.value("task", "train")
    if task in ["nop", "cleanup_old_models"]:
        return False
    return True


def execute_main_task():
    """
    Executes the main task (via config ``task`` option).
    """
    from returnn.util.basic import hms_fraction

    start_time = time.time()
    task = config.value("task", "train")
    if config.is_true("dry_run"):
        print("Dry run, will not save anything.", file=log.v1)
    if task == "train":
        assert (
            train_data and train_data.have_seqs()
        ), "no train files specified, check 'train' option: %s" % config.value("train", None)
        engine.init_train(train_data, dev_data, eval_data)
        engine.train()
    elif task == "eval":
        raise NotImplementedError("eval task is currently not implemented")
        # if config.value("load", None):
        #     # this would directly load whatever model is specified
        #     print("Evaluate model", config.value("load", None), file=log.v2)
        #     lr_control_update_scores = False
        # else:
        #     # Assume the configured model with some given epoch.
        #     epoch = config.int("epoch", -1)
        #     load_epoch = config.int("load_epoch", -1)
        #     if epoch >= 0:
        #         assert (load_epoch < 0) or (load_epoch == epoch), "epoch and load_epoch have to match"
        #         engine.epoch = epoch
        #         config.set("load_epoch", engine.epoch)
        #     else:
        #         assert load_epoch >= 0, "specify epoch or load_epoch"
        #         engine.epoch = load_epoch
        #     print("Evaluate epoch", engine.epoch, file=log.v2)
        #     lr_control_update_scores = True
        # engine.init_train_from_config(config, train_data, dev_data, eval_data)
        # engine.eval_model(
        #     output_file=config.value("eval_output_file", None),
        #     output_per_seq_file=config.value("eval_output_file_per_seq", None),
        #     loss_name=config.value("loss_name", None),
        #     output_per_seq_format=config.list("output_per_seq_format", ["score"]),
        #     output_per_seq_file_format=config.value("output_per_seq_file_format", "txt"),
        #     lr_control_update_scores=lr_control_update_scores,
        # )
    elif task in ["forward"]:
        assert eval_data is not None, "no eval data provided"
        if config.has("epoch"):
            config.set("load_epoch", config.int("epoch", 0))
        engine.init_forward(eval_data=eval_data)
        engine.forward()
    elif task == "search":
        raise NotImplementedError("search task is currently not implemented")
        # engine.use_search_flag = True
        # engine.use_eval_flag = config.bool("search_do_eval", True)
        # engine.init_network_from_config(config)
        # if config.value("search_data", "eval") in ["train", "dev", "eval"]:
        #     data = {"train": train_data, "dev": dev_data, "eval": eval_data}[config.value("search_data", "eval")]
        #     assert data, "set search_data"
        # else:
        #     data = init_dataset(config.opt_typed_value("search_data"))
        # engine.search(
        #     data,
        #     do_eval=config.bool("search_do_eval", True),
        #     output_layer_names=config.typed_value("search_output_layer", "output"),
        #     output_file=config.value("search_output_file", ""),
        #     output_file_format=config.value("search_output_file_format", "txt"),
        # )
    elif task == "compute_priors":
        raise NotImplementedError("compute_priors task is currently not implemented")
        # assert train_data is not None, "train data for priors should be provided"
        # engine.init_network_from_config(config)
        # engine.compute_priors(dataset=train_data, config=config)
    elif task == "cleanup_old_models":
        engine.cleanup_old_models(ask_for_confirmation=True)
    elif task.startswith("config:"):
        action = config.typed_dict[task[len("config:") :]]
        print("Task: %r" % action, file=log.v1)
        assert callable(action)
        action()
    elif task.startswith("optional-config:"):
        action = config.typed_dict.get(task[len("optional-config:") :], None)
        if action is None:
            print("No task found for %r, so just quitting." % task, file=log.v1)
        else:
            print("Task: %r" % action, file=log.v1)
            assert callable(action)
            action()
    elif task == "nop":
        print("Task: No-operation", file=log.v1)
    elif task == "initialize_model":
        engine.init_train(train_data, dev_data, eval_data)
        engine.save_model()
    else:
        raise Exception("unknown task: %r" % (task,))

    print(("elapsed: %s" % hms_fraction(time.time() - start_time)), file=log.v3)


def main(argv=None):
    """
    Main entry point of RETURNN.

    :param list[str]|None argv: ``sys.argv`` by default
    """
    if argv is None:
        argv = sys.argv
    return_code = 0
    try:
        assert len(argv) >= 2, "usage: %s <config>" % argv[0]
        init(command_line_options=argv[1:])
        execute_main_task()
    except KeyboardInterrupt:
        return_code = 1
        print("KeyboardInterrupt", file=getattr(log, "v3", sys.stderr))
        if getattr(log, "verbose", [False] * 6)[5]:
            sys.excepthook(*sys.exc_info())
    finalize()
    if return_code:
        sys.exit(return_code)


if __name__ == "__main__":
    main(sys.argv)
