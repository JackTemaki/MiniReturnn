"""
Provides :class:`EngineBase`.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

from returnn.config import Config, get_global_config
from returnn.datasets import Dataset, init_dataset
from returnn.learning_rate_control import load_learning_rate_control_from_config, LearningRateControl
from returnn.log import log
from returnn.util import basic as util


class EngineBase(object):
    """
    Base class for a backend engine.

    The purpose of the engine is to provide all necessary functions to train and execute neural models,
    being initialized by a config objects and provided datasets.

    Manages the following components:
     - The config
     - The Returnn datasets for train and eval
     - The learning rate control system
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

        self.train_dataset = None  # type: Optional[Dataset]
        self.eval_datasets = {}  # type: Dict[str, Dataset]
        self.forward_dataset = None  # type: Optional[Dataset]

        self.learning_rate = 0.0  # set in init_train_epoch
        self.learning_rate_control = None  # type: Optional[LearningRateControl]

    def init_train(
        self,
        train_data: Optional[Dataset] = None,
        dev_data: Optional[Dataset] = None,
    ):
        """
        Initialize all engine parts needed for training

        :param train_data:
        :param dev_data:
        """
        self.train_dataset = train_data or init_dataset(
            self.config.typed_value("train", {}), default_kwargs={"name": "train"}
        )

        self.eval_datasets.clear()
        self.eval_datasets["dev"] = dev_data or init_dataset(
            self.config.typed_value("dev", {}), default_kwargs={"name": "dev"}
        )
        if self.config.has("eval_datasets"):
            for dataset_name, dataset_opts in self.config.typed_value("eval_datasets", {}).items():
                self.eval_datasets[dataset_name] = init_dataset(dataset_opts, default_kwargs={"name": dataset_name})

        self.learning_rate_control = load_learning_rate_control_from_config(self.config)
        self.learning_rate = self.learning_rate_control.default_learning_rate

    def init_forward(self, forward_data: Optional[Dataset] = None):
        """
        Initialize all engine parts needed for training

        :param forward_data:
        """
        self.forward_dataset = forward_data or init_dataset(
            self.config.typed_value("forward", {}), default_kwargs={"name": "forward"}
        )

    def train(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

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
        load_model_epoch_filename = util.get_checkpoint_filepattern(config.value("load", ""))
        if load_model_epoch_filename:
            assert os.path.exists(
                load_model_epoch_filename + cls.get_file_postfix(),
            ), "load option %r, file %r does not exist" % (
                config.value("load", ""),
                load_model_epoch_filename + cls.get_file_postfix(),
            )
            # If "load" is given and we are not training, always load explicitly
            if config.value("task", "train") != "train":
                return None, load_model_epoch_filename

        if config.value("task", "train") == "train":
            start_epoch_mode = config.value("start_epoch", "auto")
            if start_epoch_mode == "auto":
                start_epoch = None
            else:
                start_epoch = int(start_epoch_mode)
                assert start_epoch >= 1

            existing_models = cls.get_existing_models(config)
            load_epoch = config.int("load_epoch", -1)

            import_model_train_epoch1 = util.get_checkpoint_filepattern(config.value("import_model_train_epoch1", ""))
            if import_model_train_epoch1:
                assert os.path.exists(import_model_train_epoch1 + cls.get_file_postfix())

            # We are in training so we first want to consider existing models, prioritizing a given start_epoch otherwise using the latest
            # If no models exist, we check if a given parameter initialization was defined in 'import_model_train_epoch1'
            # The last case is a given model in 'load', the epoch will then be read from the checkpoint.
            if existing_models:
                epoch_model = sorted(existing_models.items())[-1]
                print(f"Using existing model {epoch_model[1]}", file=log.v4)
                if load_model_epoch_filename:
                    print("note: there is a 'load' which we ignore because of existing model", file=log.v4)
                if start_epoch == 1:
                    if epoch_model[0]:  # existing model
                        print(
                            "warning: there is an existing model: %s. Model will be ignored and new model will be initialized!"
                            % (epoch_model,),
                            file=log.v4,
                        )
                        epoch_model = (None, None)
                elif (start_epoch or 0) > 1:
                    if epoch_model[0]:
                        if epoch_model[0] != start_epoch - 1:
                            print("warning: start_epoch %i but there is %s" % (start_epoch, epoch_model), file=log.v4)
                        epoch_model = start_epoch - 1, existing_models[start_epoch - 1]

            elif import_model_train_epoch1 and start_epoch in [None, 1]:
                print(f"Using import model {import_model_train_epoch1}", file=log.v4)
                epoch_model = (0, import_model_train_epoch1)

            elif load_model_epoch_filename:
                print(f"Using load model {load_model_epoch_filename} with start_epoch {start_epoch}", file=log.v4)

                if (start_epoch or 0) > 1:
                    load_epoch = start_epoch
                else:
                    load_epoch = None

                epoch_model = (load_epoch, load_model_epoch_filename)

            else:
                print(f"Using fresh model", file=log.v4)
                epoch_model = (None, None)

        else:
            assert load_model_epoch_filename, "No model given but task is not training!"

            if config.int("load_epoch", -1) > -1:
                print(
                    "warning: 'load_epoch' is used together with 'load'. 'load_epoch' will be ignored and epoch from checkpoint in 'load' will be used instead.",
                    file=log.v4,
                )

            # Epoch number is read from the checkpoint
            epoch_model = (None, load_model_epoch_filename)

        return epoch_model

    def cleanup_old_models(self, ask_for_confirmation=False):
        """
        :param bool ask_for_confirmation: if True, will ask the user interactively to confirm
        """
        from returnn.util.basic import CollectionReadCheckCovered, human_bytes_size, confirm
        from itertools import count

        opts = CollectionReadCheckCovered(self.config.get_of_type("cleanup_old_models", dict, {}))
        existing_models = self.get_existing_models(config=self.config)
        if self.learning_rate_control is not None:
            lr_control = self.learning_rate_control
        else:
            lr_control = load_learning_rate_control_from_config(self.config)
        epochs = sorted(existing_models.keys())
        if not epochs:
            print("Cannot cleanup models, no models found.", file=log.v2)
            return
        keep_last_n = opts.get("keep_last_n", 2)
        keep_best_n = opts.get("keep_best_n", 4)
        assert keep_last_n >= 1 and keep_best_n >= 0
        if max(keep_last_n, keep_best_n) >= len(epochs):
            print(
                (
                    "Only %i epochs stored so far and keeping last %i epochs and best %i epochs,"
                    " thus not cleaning up any epochs yet."
                )
                % (len(epochs), keep_last_n, keep_best_n),
                file=log.v2,
            )
            return
        keep_epochs = set()  # type: typing.Set[int]
        keep_epochs.update(opts.get("keep", set()))
        keep_epochs.update(epochs[-keep_last_n:])
        score_keys = set()  # e.g. "dev_error", "dev_score", etc.
        # Collect all possible score keys. Note that we could have different ones for different epochs.
        for data in lr_control.epoch_data.values():
            score_keys.update(data.error.keys())
        assert score_keys
        score_keys = sorted(score_keys)
        score_values = {key: [] for key in score_keys}
        for epoch in epochs:
            epoch_scores = lr_control.epoch_data[epoch].error
            for key in epoch_scores.keys():
                score_values[key].append(epoch_scores[key])
        for key in list(score_keys):
            scores = score_values[key]
            if min(scores) == max(scores):
                print(
                    "Ignoring score key %r because all epochs have the same value %r." % (key, scores[0]), file=log.v3
                )
                score_keys.remove(key)
                score_values.pop(key)
        # Actually, terminology is a bit confusing. We call it "score" here (and elsewhere), but it's a loss,
        # so the maximum value is the worst possible value.
        worst_score_values = {key: max(scores) for (key, scores) in score_values.items()}
        for key in score_keys:
            scores = sorted(
                [(lr_control.epoch_data[epoch].error.get(key, worst_score_values[key]), epoch) for epoch in epochs]
            )
            scores = scores[:keep_best_n]
            keep_epochs.update([v[1] for v in scores])
        keep_epochs.intersection_update(epochs)
        if len(keep_epochs) == len(epochs):
            print("%i epochs stored so far and keeping all." % len(epochs), file=log.v2)
            return
        remove_epochs = sorted(set(epochs).difference(keep_epochs))
        assert remove_epochs
        if len(epochs) > 6:
            epoch_summary = "[%s, ..., %s]" % (", ".join(map(str, epochs[:3])), ", ".join(map(str, epochs[-3:])))
        else:
            epoch_summary = str(epochs)
        print(
            "We have stored models for epochs %s and keep epochs %s." % (epoch_summary, sorted(keep_epochs)),
            file=log.v3,
        )
        print("We will delete the models of epochs %s." % (remove_epochs,), file=log.v3)
        opts.assert_all_read()
        if self.config.bool("dry_run", False):
            print("Dry-run, will not delete models.", file=log.v2)
            return
        if ask_for_confirmation:
            confirm("Delete those models?", exit_on_false=True)
        count_bytes = 0
        for epoch in remove_epochs:
            count_bytes += self.delete_model(existing_models[epoch])
        print("Deleted %s." % human_bytes_size(count_bytes), file=log.v2)

    @staticmethod
    def delete_model(filename):
        """
        :param str filename:
        :return: accumulated file-size in bytes of deleted files
        :rtype: int
        """
        raise NotImplementedError

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

    def _is_dataset_evaluated(self, name: str) -> bool:
        """
        Check via self.learning_rate_control.

        :param name: e.g. "dev"
        :return: whether there is an entry for the score in the learning rate file
        """
        assert self.learning_rate_control.filename  # otherwise we would not have stored it
        error_dict = self.learning_rate_control.get_epoch_error_dict(self.epoch)
        if not error_dict:
            return False
        return any([k.startswith(name) for k in error_dict.keys()])
