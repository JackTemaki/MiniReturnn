"""
Main engine for PyTorch
"""

from __future__ import annotations
from typing import Optional, Callable, Dict, Tuple

import os
import time
import torch
import torch.utils.data.datapipes as dp
from torch import Tensor
from torchdata.dataloader2 import DataLoader2
from random import random

from returnn.config import Config
from returnn.log import log
from returnn.engine.base import EngineBase
from returnn.datasets.basic import init_dataset, Dataset
from returnn.util import NumbersDict
from .updater import Updater
from .data import pipeline as data_pipeline
from .data import returnn_dataset_wrapper
from .context import get_run_ctx, init_train_step_run_ctx, init_forward_step_run_ctx, RunCtx, Loss


class Engine(EngineBase):
    """
    PyTorch engine
    """

    FILE_POSTFIX = ".pt"

    def __init__(self, config: Config):
        """
        :param config:
        """
        super(Engine, self).__init__(config=config)
        self.model_filename = self.config.value("model", None)
        self._mp_manager = torch.multiprocessing.Manager()
        self._epoch_mp_shared = self._mp_manager.Value("i", 0)
        self.train_dataset = None  # type: Optional[Dataset]
        self.eval_datasets = {}
        self._train_dataloader = None  # type: Optional[DataLoader2]
        self._eval_dataloaders = {}  # type: Dict[str, DataLoader2]

        self._start_epoch = None  # type: Optional[int]
        self._final_epoch = None  # type: Optional[int]
        self._model = None  # type: Optional[torch.nn.Module]
        self._train_step_func = None  # type: Optional[Callable]
        self._save_model_epoch_interval = 1
        self._updater = None  # type: Optional[Updater]

        self._device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        print(f"Using device {self._device}", file=log.v3)

    def init_train_from_config(
        self,
        config: Optional[Config] = None,
        train_data: Optional[Dataset] = None,
        dev_data: Optional[Dataset] = None,
        eval_data: Optional[Dataset] = None,
    ):
        """
        :param config:
        :param train_data:
        :param dev_data:
        :param eval_data:
        """
        assert config is self.config
        super().init_train_from_config(config=config)
        self.train_dataset = train_data
        self.eval_datasets.clear()
        if dev_data:
            self.eval_datasets["dev"] = dev_data
        if eval_data:
            self.eval_datasets["eval"] = eval_data
        if config.has("eval_datasets"):
            for dataset_name, dataset_opts in config.typed_value("eval_datasets", {}).items():
                self.eval_datasets[dataset_name] = init_dataset(dataset_opts, default_kwargs={"name": dataset_name})

        self._train_dataloader = self._create_data_loader(train_data) if train_data else None
        for dataset_name, dataset in self.eval_datasets.items():
            self._eval_dataloaders[dataset_name] = self._create_data_loader(dataset)

        self._start_epoch = self.get_train_start_epoch(self.config)
        self._final_epoch = self.config_get_final_epoch(self.config)

        self._load_model(epoch=self._start_epoch, step=0)
        self._save_model_epoch_interval = config.int("save_interval", 1)

        self._updater = Updater(self.config, self._model, self.learning_rate)
        self._updater.create_optimizer()
        if self._start_epoch > 1:
            self._load_optimizer(self._start_epoch)

        self._train_step_func = self.config.typed_value("train_step")
        assert self._train_step_func, "train_step not defined"

    def train(self):
        """
        Main training loop.
        """

        print("Starting training at epoch {}.".format(self._start_epoch), file=log.v3)
        assert self._model, "Model not initialized, call init_train_from_config()."

        self.epoch = self._start_epoch
        self._epoch_mp_shared.value = self.epoch
        while self.epoch <= self._final_epoch:
            self.init_train_epoch()
            self.train_epoch()

            self.epoch += 1
            self._epoch_mp_shared.value = self.epoch

        print("Finished training at epoch {}.".format(self.epoch), file=log.v3)

    def init_train_epoch(self):
        """
        init train (sub)epoch. LR etc
        """
        self.learning_rate = self.learning_rate_control.get_learning_rate_for_epoch(self.epoch)

        # Update learning rate
        self._updater.set_learning_rate(self.learning_rate)

    def train_epoch(self):
        """
        train one (sub)epoch
        """
        print("start", self.get_epoch_str(), "with learning rate", self.learning_rate, "...", file=log.v4)
        epoch_start_time = time.time()

        self._model.train()
        init_train_step_run_ctx(device=self._device)

        accumulated_losses_dict = NumbersDict()
        accumulated_inv_norm_dict = NumbersDict()
        step_idx = 0
        for data in self._train_dataloader:
            step_time_start = time.time()

            run_ctx = get_run_ctx()
            run_ctx.init_step()

            total_loss, ctx_losses_dict = self.run_train_step(data, run_ctx)

            losses_dict = NumbersDict({
                "train_loss_" + name: float(loss.loss.detach().cpu().numpy()) for name, loss in ctx_losses_dict.items()
            })
            inv_norm_dict = NumbersDict({
                "train_loss_" + name:
                # in case we have no inv norm factor we use 1 to normalize via the step count
                float(loss.inv_norm_factor.detach().cpu().numpy()) if loss.inv_norm_factor is not None else 1
                for name, loss in ctx_losses_dict.items()
            })
            accumulated_losses_dict += losses_dict
            accumulated_inv_norm_dict += inv_norm_dict
            self.print_step_info(
                f"train epoch {self.epoch}", step_idx, step_start_time=step_time_start,
                total_loss=float(total_loss.detach().cpu().numpy()),
                loss_dict=losses_dict / inv_norm_dict
            )
            step_idx += 1

        print("Trained %i steps, took %.3fs" % (step_idx, time.time() - epoch_start_time))

        accumulated_losses_dict = accumulated_losses_dict / accumulated_inv_norm_dict
        self.learning_rate_control.set_epoch_error(self.epoch, dict(accumulated_losses_dict))
        self.learning_rate_control.save()

        if self.epoch % self._save_model_epoch_interval == 0 or self.epoch == self._final_epoch:
            self._save_model()
            self._save_optimizer()

        self.eval_model()
        self.cleanup_old_models(ask_for_confirmation=False)

    def eval_model(self):
        """
        Runs model on all eval datasets and calculates the loss.
        """
        self._model.eval()
        init_train_step_run_ctx(device=self._device)

        for dataset_name, dataset in self.eval_datasets.items():
            dataset_start_time = time.time()
            print(f"Evaluating dataset {dataset_name!r}'", file=log.v3)

            data_loader = self._eval_dataloaders[dataset_name]

            accumulated_loss = 0.0
            accumulated_losses_dict = NumbersDict()
            accumulated_inv_norm_dict = NumbersDict()
            step_idx = 0

            with torch.no_grad():
                for data in data_loader:
                    step_time_start = time.time()
                    run_ctx = get_run_ctx()
                    run_ctx.init_step()

                    total_loss, ctx_losses_dict = self.run_eval_step(data, run_ctx)

                    losses_dict = NumbersDict({
                        "train_loss_" + name: float(loss.loss.detach().cpu().numpy()) for name, loss in
                        ctx_losses_dict.items()
                    })
                    inv_norm_dict = NumbersDict({
                        "train_loss_" + name:
                        # in case we have no inv norm factor we use 1 to normalize via the step count
                            float(
                                loss.inv_norm_factor.detach().cpu().numpy()) if loss.inv_norm_factor is not None else 1
                        for name, loss in ctx_losses_dict.items()
                    })
                    accumulated_losses_dict += losses_dict
                    accumulated_inv_norm_dict += inv_norm_dict
                    self.print_step_info(
                        f"eval {dataset_name} epoch {self.epoch}", step_idx, step_start_time=step_time_start,
                        total_loss=float(total_loss.detach().cpu().numpy()),
                        loss_dict=losses_dict / inv_norm_dict
                    )
                    accumulated_loss += total_loss
                    accumulated_losses_dict += NumbersDict(losses_dict)
                    step_idx += 1

            assert step_idx > 0, "No data in dataset '{}'.".format(dataset_name)
            accumulated_losses_dict = accumulated_losses_dict / accumulated_inv_norm_dict

            self.learning_rate_control.set_epoch_error(self.epoch, dict(accumulated_losses_dict))

            print(
                "Total loss for '{}': {:.6}, took: %.3f".format(
                    dataset_name, accumulated_loss, time.time() - dataset_start_time
                ),
                file=log.v3,
            )

        self.learning_rate_control.save()

    def _create_data_loader(self, dataset: Dataset) -> DataLoader2:
        """
        :param dataset: RETURNN dataset
        :return: PyTorch data loader created from given RETURNN dataset
        """
        # Make sure that _dataset_reset does not keep a ref to `self`,
        # otherwise it would trigger to pickle `self` and all its members.
        dataset_reset = returnn_dataset_wrapper.ReturnnDatasetResetMpSharedEpochCallback(
            dataset=dataset, epoch_mp_shared=self._epoch_mp_shared
        )

        wrapped_dataset = returnn_dataset_wrapper.ReturnnDatasetIterDataPipe(dataset, reset_callback=dataset_reset)

        chunking = self.config.typed_value("chunking", None)
        if chunking:
            wrapped_dataset = data_pipeline.ChunkingIterDataPipe(wrapped_dataset, chunking)

        batch_size = self.config.typed_value("batch_size", 1)
        max_seqs = self.config.int("max_seqs", -1)
        batches_dataset = data_pipeline.BatchingIterDataPipe(wrapped_dataset, batch_size=batch_size, max_seqs=max_seqs)
        batches_dataset = dp.iter.Collator(batches_dataset, collate_fn=data_pipeline.collate_batch)

        try:
            return DataLoader2(batches_dataset)
        except TypeError as exc:
            try:
                # noinspection PyPackageRequirements
                import dill
            except ImportError:
                raise ModuleNotFoundError("Possible type error in DataLoader2 due to missing module 'dill'") from exc

    def run_train_step(self, data: dict[str, torch.Tensor], run_ctx: RunCtx) -> Tuple[Tensor, Dict[str, Loss]]:
        """
        :param data: model inputs for the step
        :param run_ctx: the current run ctx object
        :return: total loss (weighted sum) calculated for the step, and individual losses as a name -> value mapping
        """
        assert isinstance(data, dict) and data
        # move all data to the target device as default
        # note that in some cases, e.g. for using rnn.pack_padded_sequence you need to have
        # length tensors on CPU
        data = {k: v.to(self._device) for (k, v) in data.items()}

        sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
        self._train_step_func(model=self._model, data=data, run_ctx=run_ctx, **sentinel_kw)

        losses_dict = run_ctx.losses
        total_loss = run_ctx.total_loss()

        self._updater.get_optimizer().zero_grad()
        total_loss.backward()
        self._updater.get_optimizer().step()

        return total_loss, losses_dict

    def run_eval_step(self, data: dict[str, torch.Tensor], run_ctx: RunCtx) -> Tuple[Tensor, Dict[str, Loss]]:
        """
        :param data: model inputs for the step
        :param run_ctx: the current run ctx object
        :return: total loss (weighted sum) calculated for the step, and individual losses as a name -> value mapping
        """
        assert isinstance(data, dict) and data
        # move all data to the target device as default
        # note that in some cases, e.g. for using rnn.pack_padded_sequence you need to have
        # length tensors on CPU
        data = {k: v.to(self._device) for (k, v) in data.items()}

        sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
        # currently we only support the _train_step_func,
        # this can be an optional _eval_step_func later on
        self._train_step_func(model=self._model, data=data, run_ctx=run_ctx, **sentinel_kw)

        losses_dict = run_ctx.losses
        total_loss = run_ctx.total_loss()

        return total_loss, losses_dict

    def _load_model(self, *, epoch: int, step: int):
        """
        Sets self._model to a torch.nn.Module.

        :param epoch:
        :param step:
        """
        # See :mod:`rf.rand` docstring for an explanation of this logic.
        random_seed = self.config.int("random_seed", 42)
        random_seed = (epoch * 193939 + step * 19937 + random_seed * 27644437 + 479001599) % (2**31)
        # rf.set_random_seed(random_seed)

        get_model_func = self.config.typed_value("get_model")
        assert get_model_func, "get_model not defined"
        sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
        self._model = get_model_func(**sentinel_kw)
        assert isinstance(self._model, torch.nn.Module)

        if epoch > 1:
            filename = self.get_epoch_model_filename(epoch=epoch - 1) + ".pt"
            print("Load model %s" % (filename,), file=log.v4)
            model_state = torch.load(filename, map_location=torch.device(self._device))
            self._model.load_state_dict(model_state)
        preload_from_files = self.config.typed_value("preload_from_files", {})
        if preload_from_files:
            # see `preload_from_files` in tf engine and `returnn.tf.network.CustomCheckpointLoader`
            is_training = self.config.value("task", "train") == "train"
            is_first_train_epoch = epoch == 1 and (
                is_training or self.config.value("task", "train") == "initialize_model"
            )
            # We use the reversed sorted order here to achieve consistent behavior with the TF engine.
            # There, the keys are used in sorted order but if a variable is loaded,
            # it will not be considered anymore afterwards.
            # So the first occurrence is used.
            # Here, we overwrite variables even if they have been loaded before.
            # In order to get consistent behavior, we use the reversed order.
            for preload_key, opts in reversed(sorted(preload_from_files.items())):
                assert isinstance(opts, dict) and "filename" in opts
                if opts.get("init_for_train", False):
                    if not is_first_train_epoch:
                        continue
                else:  # default: init for recog
                    if is_training:
                        continue
                print(f"Pre-load weights for key '{preload_key}' from {opts['filename']}", file=log.v3)
                preload_model_state = torch.load(opts["filename"])
                if opts.get("checkpoint_key", None) is not None:
                    # This can be used if an external checkpoint saves a checkpoint a different structure that just the
                    # model state dict. E.g., if a checkpoint is created using
                    # `torch.save({"model": model.state_dict(), "optimizer": optimizer.state)_dict(), ...})`
                    # we can set checkpoint_key = "model" to load the model.
                    # Currently, this only supports single level dicts, but it could be extended if needed.
                    preload_model_state = preload_model_state[opts["checkpoint_key"]]
                if opts.get("prefix", ""):
                    # Only params with this prefix should be loaded.
                    # They are expected to be in the checkpoint without this prefix.
                    # By adding the prefix to all params,
                    # we make sure that only params matching this condition are loaded.
                    # This is in line with the behavior of the TF engine.
                    preload_model_state = {opts["prefix"] + key: value for key, value in preload_model_state.items()}
                ignore_params = opts.get("ignore_params", [])
                ignore_params_prefixes = opts.get("ignore_params_prefixes", [])
                for key in list(preload_model_state.keys()):
                    if key in ignore_params or any(
                        [key.startswith(ignore_key) for ignore_key in ignore_params_prefixes]
                    ):
                        print(f"Ignoring variable {key}", file=log.v3)
                        preload_model_state.pop(key)
                for new_name, name_in_checkpoint in opts.get("var_name_mapping", {}).items():
                    preload_model_state[new_name] = preload_model_state.pop(name_in_checkpoint)
                missing_keys, _ = self._model.load_state_dict(preload_model_state, strict=False)
                if not opts.get("ignore_missing", False):
                    prefix_keys = [key for key in self._model.state_dict() if key.startswith(opts.get("prefix", ""))]
                    missing_prefix_keys = set(prefix_keys).intersection(set(missing_keys))
                    assert not missing_prefix_keys, f"Missing keys and ignore_missing=False: {missing_prefix_keys}"
                print(f"Missing keys: {missing_keys}", file=log.v4)

        self._model.to(self._device)

    def _save_model(self):
        """
        Saves the state of self._model to file.
        """
        filename = self.get_epoch_model_filename() + ".pt"
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        print("Save model under %s" % (filename,), file=log.v4)
        torch.save(self._model.state_dict(), filename)

    def _load_optimizer(self, epoch):
        """
        Loads a torch.optim.Optimizer from disk and uses it as the optimizer.
        This function is a wrapper to Updater.load_optimizer().

        :param int epoch: Epoch from which to load the optimizer state.
        """
        filename = self.get_epoch_model_filename(epoch=epoch - 1) + ".opt.pt"
        self._updater.load_optimizer(filename, device=self._device)

    def _save_optimizer(self):
        """
        Saves the optimizer state to a file.
        This function is a wrapper to Updater.save_optimizer().
        """
        filename = self.get_epoch_model_filename() + ".opt.pt"
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self._updater.save_optimizer(filename)

        # keep only the last two optimizer states (two in case one file gets corrupted)
        clean_epoch = self.epoch - 2
        if clean_epoch > 0:
            filename = self.get_epoch_model_filename(epoch=clean_epoch) + ".opt.pt"
            if os.path.isfile(filename):
                os.unlink(filename)

    @staticmethod
    def delete_model(filename):
        """
        :param str filename:
        :return: accumulated file-size in bytes of deleted files
        :rtype: int
        """
        count_bytes = 0
        fn = filename + ".pt"
        assert os.path.exists(fn)
        count_bytes += os.stat(fn).st_size
        os.remove(fn)
        assert count_bytes > 0
        return count_bytes

    @staticmethod
    def print_step_info(report_prefix: str, step: int, step_start_time: float, total_loss: float, loss_dict: NumbersDict):
        """

        :param report_prefix:
        :param step:
        :param step_start_time:
        :param total_loss:
        :param loss_dict:
        """
        if log.verbose[5]:
            info = [report_prefix, "step %i" % step, "took: %.3f" % (time.time() - step_start_time)]
            info += ["total (grad) loss: %f" % total_loss]
            info += ["%s: %.5f" % (k, v) for k, v in sorted(loss_dict.items())]
            print(", ".join(filter(None, info)), file=log.v5)
