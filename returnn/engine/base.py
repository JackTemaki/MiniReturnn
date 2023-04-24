"""
Provides :class:`EngineBase`.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from returnn.config import Config, get_global_config
from returnn.learning_rate_control import load_learning_rate_control_from_config, LearningRateControl
from returnn.log import log
from returnn.util import basic as util


class EngineBase(object):
    """
    Base class for a backend engine, such as :class:`TFEngine.Engine`.
    """

    FILE_POSTFIX = None

    def __init__(self, config: Optional[Config] = None):
        """
        :param config:
        """
        if config is None:
            config = get_global_config(auto_create=True)
        self.config = config
        self.epoch = 0
        self.model_filename = None  # type: Optional[str]
        self.learning_rate = 0.0  # set in init_train_epoch
        self.learning_rate_control = None  # type: Optional[LearningRateControl]

    def init_train_from_config(self, config: Optional[Config] = None):
        """
        Initialize all engine parts needed for training

        :param config:
        """
        self.learning_rate_control = load_learning_rate_control_from_config(config)
        self.learning_rate = self.learning_rate_control.default_learning_rate

    @classmethod
    def config_get_final_epoch(cls, config):
        """
        :param returnn.config.Config config:
        :rtype: int
        """
        num_epochs = config.int("num_epochs", 5)
        if config.has("load_epoch"):
            num_epochs = max(num_epochs, config.int("load_epoch", 0))
        return num_epochs

    @classmethod
    def get_existing_models(cls, config):
        """
        :param returnn.config.Config config:
        :return: dict epoch -> model filename
        :rtype: dict[int,str]
        """
        model_filename = config.value("model", "")
        if not model_filename:
            return {}
        # Automatically search the filesystem for existing models.
        file_list = {}
        for epoch in range(1, cls.config_get_final_epoch(config) + 1):
            for is_pretrain in [False, True]:
                fn = cls.epoch_model_filename(model_filename, epoch)
                if os.path.exists(fn):
                    file_list[epoch] = fn
                    break
                if os.path.exists(fn + cls.get_file_postfix()):
                    file_list[epoch] = fn
                    break
        return file_list

    @classmethod
    def get_epoch_model(cls, config):
        """
        :type config: returnn.config.Config
        :returns (epoch, modelFilename)
        :rtype: (int|None, str|None)
        """
        start_epoch_mode = config.value("start_epoch", "auto")
        if start_epoch_mode == "auto":
            start_epoch = None
        else:
            start_epoch = int(start_epoch_mode)
            assert start_epoch >= 1

        load_model_epoch_filename = util.get_checkpoint_filepattern(config.value("load", ""))
        if load_model_epoch_filename:
            assert os.path.exists(
                load_model_epoch_filename + cls.get_file_postfix(),
            ), "load option %r, file %r does not exist" % (
                config.value("load", ""),
                load_model_epoch_filename + cls.get_file_postfix(),
            )

        import_model_train_epoch1 = util.get_checkpoint_filepattern(config.value("import_model_train_epoch1", ""))
        if import_model_train_epoch1:
            assert os.path.exists(import_model_train_epoch1 + cls.get_file_postfix())

        existing_models = cls.get_existing_models(config)
        load_epoch = config.int("load_epoch", -1)
        if load_model_epoch_filename:
            if load_epoch <= 0:
                load_epoch = util.model_epoch_from_filename(load_model_epoch_filename)
        else:
            if load_epoch > 0:  # ignore if load_epoch == 0
                assert load_epoch in existing_models
                load_model_epoch_filename = existing_models[load_epoch]
                assert util.model_epoch_from_filename(load_model_epoch_filename) == load_epoch

        # Only use this when we don't train.
        # For training, we first consider existing models
        # before we take the 'load' into account when in auto epoch mode.
        # In all other cases, we use the model specified by 'load'.
        if load_model_epoch_filename and (config.value("task", "train") != "train" or start_epoch is not None):
            if config.value("task", "train") == "train" and start_epoch is not None:
                # Ignore the epoch. To keep it consistent with the case below.
                epoch = None
            else:
                epoch = load_epoch
            epoch_model = (epoch, load_model_epoch_filename)

        # In case of training, always first consider existing models.
        # This is because we reran RETURNN training, we usually don't want to train from scratch
        # but resume where we stopped last time.
        elif existing_models:
            epoch_model = sorted(existing_models.items())[-1]
            if load_model_epoch_filename:
                print("note: there is a 'load' which we ignore because of existing model", file=log.v4)

        elif config.value("task", "train") == "train" and import_model_train_epoch1 and start_epoch in [None, 1]:
            epoch_model = (0, import_model_train_epoch1)

        # Now, consider this also in the case when we train, as an initial model import.
        elif load_model_epoch_filename:
            # Don't use the model epoch as the start epoch in training.
            # We use this as an import for training.
            epoch_model = (load_epoch, load_model_epoch_filename)

        else:
            epoch_model = (None, None)

        if start_epoch == 1:
            if epoch_model[0]:  # existing model
                print("warning: there is an existing model: %s" % (epoch_model,), file=log.v4)
                epoch_model = (None, None)
        elif (start_epoch or 0) > 1:
            if epoch_model[0]:
                if epoch_model[0] != start_epoch - 1:
                    print("warning: start_epoch %i but there is %s" % (start_epoch, epoch_model), file=log.v4)
                epoch_model = start_epoch - 1, existing_models[start_epoch - 1]

        return epoch_model

    @classmethod
    def get_train_start_epoch(cls, config: Config) -> int:
        """
        :param config: returnn.config.Config
        """
        last_epoch, _ = cls.get_epoch_model(config)
        if last_epoch is None:
            start_epoch = 1
        else:
            # Start with next epoch.
            start_epoch = last_epoch + 1
        return start_epoch

    @classmethod
    def epoch_model_filename(cls, model_filename, epoch):
        """
        :type model_filename: str
        :type epoch: int
        :rtype: str
        """
        if sys.platform == "win32" and model_filename.startswith("/tmp/"):
            import tempfile

            model_filename = tempfile.gettempdir() + model_filename[len("/tmp") :]
        return model_filename + ".%03d" % epoch

    def get_epoch_model_filename(self, epoch=None):
        """
        :param int|None epoch:
        :return: filename, excluding TF specific postfix
        :rtype: str
        """
        if not epoch:
            epoch = self.epoch
        return self.epoch_model_filename(self.model_filename, epoch)

    def get_epoch_str(self):
        """
        :return: e.g. "epoch 3", or "pretrain epoch 5"
        :rtype: str
        """
        return "epoch %s" % self.epoch

    @classmethod
    def get_file_postfix(cls):
        if cls.FILE_POSTFIX is None:
            raise NotImplementedError("Missing FILE_POSTFIX in Engine")
        return cls.FILE_POSTFIX
