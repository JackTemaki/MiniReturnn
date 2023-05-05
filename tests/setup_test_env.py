"""
Setups the environment for tests.
In the test code, you would have this::

    import _setup_test_env  # noqa

Also see :mod:`_setup_returnn_env`.
See :func:`setup` below for details.
"""


from __future__ import annotations


def setup():
    """
    Calls necessary setups.
    """
    import logging
    import os
    import sys

    # Enable all logging, up to debug level.
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    # Get us some further useful debug messages (in some cases, e.g. CUDA).
    # For example: https://github.com/tensorflow/tensorflow/issues/24496
    # os.environ["CUDNN_LOGINFO_DBG"] = "1"
    # os.environ["CUDNN_LOGDEST_DBG"] = "stdout"
    # The following might fix (workaround): Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
    # (https://github.com/tensorflow/tensorflow/issues/24496).
    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    os.environ.setdefault("RETURNN_TEST", "1")

    import returnn.util.basic as util

    util.init_thread_join_hack()

    import better_exchook

    if sys.excepthook != sys.__excepthook__:
        prev_sys_excepthook = sys.excepthook

        def _chained_excepthook(exctype, value, traceback):
            better_exchook.better_exchook(exctype, value, traceback)
            prev_sys_excepthook(exctype, value, traceback)

        sys.excepthook = _chained_excepthook
    else:
        sys.excepthook = better_exchook.better_exchook
    better_exchook.replace_traceback_format_tb()

    from returnn.log import log

    # No propagate, use stdout directly.
    log.initialize(verbosity=[5], propagate=False)

    import returnn.util.debug as debug

    debug.install_lib_sig_segfault()

    try:
        # noinspection PyUnresolvedReferences
        import faulthandler

        # Enable after libSigSegfault, so that we have both,
        # because faulthandler will also call the original sig handler.
        faulthandler.enable()
    except ImportError:
        print("no faulthandler")



setup()
