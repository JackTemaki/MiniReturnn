"""
Main engine for PyTorch
"""

from __future__ import annotations
from functools import partial
from typing import Optional, Callable, Dict, Tuple
from contextlib import nullcontext

import gc
import os
import time
import torch
import torch.utils.data.datapipes as dp
from torch import autocast, Tensor
from torch.utils.data import DataLoader
from random import random
import math

from returnn.config import Config
from returnn.log import log
from returnn.engine.base import EngineBase
from returnn.datasets.basic import Dataset
from returnn.util import NumbersDict
from .updater import Updater
from .data import pipeline as data_pipeline
from .data import returnn_dataset_wrapper
from .context import get_run_ctx, init_load_run_ctx, init_train_step_run_ctx, init_forward_step_run_ctx, RunCtx, Loss
from .util import diagnose_gpu


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
        self._train_dataloader = None  # type: Optional[DataLoader]
        self._forward_dataloader = None  # type: Optional[DataLoader]
        self._eval_dataloaders = {}  # type: Dict[str, DataLoader]

        self._start_epoch = None  # type: Optional[int]
        self._final_epoch = None  # type: Optional[int]
        self._model = None  # type: Optional[torch.nn.Module]
        self._train_step = 0
        self._train_step_func = None  # type: Optional[Callable]
        self._forward_step_func = None  # type: Optional[Callable]
        self._save_model_epoch_interval = 1
        self._updater = None  # type: Optional[Updater]

        device = config.typed_value("device")
        if device == "gpu":
            print("Use of deprecated keyword 'gpu' to set device. Using 'cuda' instead.", file=log.v1)
            device = "cuda"
        self._device = device
        if self._device == "cuda":
            diagnose_gpu.print_available_devices(file=log.v2)
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                diagnose_gpu.print_using_cuda_device_report(self._device, file=log.v2)
            else:
                reasons = diagnose_gpu.diagnose_no_gpu()
                raise Exception("No GPU device found, but config requested 'gpu' device.\n" + "\n".join(reasons))

        self._amp_dtype = None  # type: Optional[torch.dtype]

    def init_train(
        self,
        train_data: Optional[Dataset] = None,
        dev_data: Optional[Dataset] = None,
    ):
        """
        :param train_data: Used when initializing from existing Datasets
        :param dev_data: Used when initializing from existing Datasets
        """
        super().init_train(train_data=train_data, dev_data=dev_data)

        self._train_dataloader = self._create_data_loader(self.train_dataset)
        for dataset_name, dataset in self.eval_datasets.items():
            self._eval_dataloaders[dataset_name] = self._create_data_loader(dataset)

        self._start_epoch, filename = self.get_epoch_model(self.config)

        if self._start_epoch is not None:
            self._start_epoch += 1
        else:
            self._start_epoch = 1

        self._final_epoch = self.config_get_final_epoch(self.config)

        self._load_model(epoch=self._start_epoch, filename=filename)

        self._save_model_epoch_interval = self.config.int("save_interval", 1)

        self._updater = Updater(
            config=self.config, network=self._model, device=self._device, initial_learning_rate=self.learning_rate
        )
        self._updater.create_optimizer()

        if self._start_epoch > 1:
            self._load_optimizer(self._start_epoch, filename=filename + ".opt.pt")

        self._train_step_func = self.config.typed_value("train_step")
        assert self._train_step_func, "train_step not defined"

        amp_options = self.config.typed_value("torch_amp_options")
        if amp_options is not None:
            assert isinstance(amp_options, dict)
            amp_dtype_str = amp_options.get("dtype")
            assert amp_dtype_str in ["float16", "bfloat16"]
            self._amp_dtype = getattr(torch, amp_dtype_str)
            self._updater.create_grad_scaler()

    def init_forward(
        self,
        forward_data: Optional[Dataset] = None,
    ):
        """
        :param forward_data:
        """
        super().init_forward(forward_data=forward_data)
        self._forward_dataloader = self._create_data_loader(self.forward_dataset)

        _, filename = self.get_epoch_model(self.config)

        # for now assume we only do forward within one epoch setting
        self._final_epoch = self._start_epoch

        self._load_model(epoch=None, filename=filename)

        self._forward_step_func = self.config.typed_value("forward_step")
        assert self._forward_step_func, "forward_step not defined"

    def train(self):
        """
        Main training loop.
        """

        print("Starting training at epoch {}.".format(self._start_epoch), file=log.v3)
        assert self._model is not None, "Model not initialized, call init_train_from_config()."
        self._check_epoch_missing_eval()

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
        init_train_step_run_ctx(device=self._device, engine=self, epoch=self.epoch)

        accumulated_losses_dict = NumbersDict()
        accumulated_inv_norm_dict = NumbersDict()
        step_idx = 0
        for data in self._train_dataloader:
            step_time_start = time.time()

            run_ctx = get_run_ctx()
            run_ctx.init_step(self._train_step)

            total_loss, ctx_losses_dict = self.run_train_step(data, run_ctx, step_idx)

            losses_dict = NumbersDict(
                {name: float(loss.loss.detach().cpu().numpy()) for name, loss in ctx_losses_dict.items()}
            )
            inv_norm_dict = NumbersDict(
                {
                    # in case we have no inv norm factor we use 1 to normalize via the step count
                    name: float(loss.inv_norm_factor.detach().cpu().numpy()) if loss.inv_norm_factor is not None else 1
                    for name, loss in ctx_losses_dict.items()
                }
            )

            if self.config.bool("stop_on_nonfinite_train_score", True):
                self._check_nonfinite_train_score(losses_dict, accumulated_losses_dict)

            accumulated_losses_dict += losses_dict
            accumulated_inv_norm_dict += inv_norm_dict
            self.print_step_info(
                f"train epoch {self.epoch}",
                step_idx,
                step_start_time=step_time_start,
                total_loss=float(total_loss.detach().cpu().numpy()),
                loss_dict=losses_dict / inv_norm_dict,
            )
            step_idx += 1
            self._train_step += 1

        print("Trained %i steps, took %.3fs" % (step_idx, time.time() - epoch_start_time), file=log.v2)

        accumulated_losses_dict = accumulated_losses_dict / accumulated_inv_norm_dict
        self.learning_rate_control.set_epoch_error(self.epoch, dict(accumulated_losses_dict))
        self.learning_rate_control.save()

        if self.epoch % self._save_model_epoch_interval == 0 or self.epoch == self._final_epoch:
            self.save_model()
            self._save_optimizer()

        self.eval_model()
        if self.config.bool_or_other("cleanup_old_models", None):
            self.cleanup_old_models(ask_for_confirmation=False)

    def eval_model(self, *, skip_already_evaluated: bool = False):
        """
        Runs model on all eval datasets and calculates the loss.
        """
        self._model.eval()
        init_train_step_run_ctx(device=self._device, engine=self, epoch=self.epoch)

        for dataset_name, dataset in self.eval_datasets.items():
            if skip_already_evaluated and self._is_dataset_evaluated(name=dataset_name):
                continue
            dataset_start_time = time.time()
            print(f"Evaluating dataset {dataset_name!r}'", file=log.v3)

            data_loader = self._eval_dataloaders[dataset_name]

            accumulated_losses_dict = NumbersDict()
            accumulated_inv_norm_dict = NumbersDict()
            step_idx = 0

            with torch.no_grad():
                for data in data_loader:
                    step_time_start = time.time()
                    run_ctx = get_run_ctx()
                    run_ctx.init_step(self._train_step)

                    total_loss, ctx_losses_dict = self.run_eval_step(data, run_ctx)

                    losses_dict = NumbersDict(
                        {name: float(loss.loss.detach().cpu().numpy()) for name, loss in ctx_losses_dict.items()}
                    )
                    inv_norm_dict = NumbersDict(
                        {
                            # in case we have no inv norm factor we use 1 to normalize via the step count
                            name: (
                                float(loss.inv_norm_factor.detach().cpu().numpy())
                                if loss.inv_norm_factor is not None
                                else 1
                            )
                            for name, loss in ctx_losses_dict.items()
                        }
                    )
                    accumulated_losses_dict += losses_dict
                    accumulated_inv_norm_dict += inv_norm_dict
                    self.print_step_info(
                        f"eval {dataset_name} epoch {self.epoch}",
                        step_idx,
                        step_start_time=step_time_start,
                        total_loss=float(total_loss.detach().cpu().numpy()),
                        loss_dict=losses_dict / inv_norm_dict,
                    )
                    step_idx += 1

            assert step_idx > 0, "No data in dataset '{}'.".format(dataset_name)
            accumulated_losses_dict = accumulated_losses_dict / accumulated_inv_norm_dict

            self.learning_rate_control.set_epoch_error(
                self.epoch, {f"{dataset_name}_loss_{k}": v for k, v in accumulated_losses_dict.items()}
            )
            self.learning_rate_control.save()

            print(
                "Finished evaluating {} in {:.3}s".format(dataset_name, time.time() - dataset_start_time),
                file=log.v3,
            )

    def forward(self):
        """
        Runs the model
        """
        self._model.eval()
        self.epoch = self._start_epoch
        init_forward_step_run_ctx(device=self._device, engine=self, epoch=self.epoch)

        if forward_init_hook := self.config.typed_value("forward_init_hook", None):
            assert callable(forward_init_hook)
            forward_init_hook(get_run_ctx(), **{"__random_arg_%i" % int(random() * 100): None})

        dataset_start_time = time.time()
        print(f"Start Forwarding", file=log.v3)

        data_loader = self._forward_dataloader
        step_idx = 0

        with torch.no_grad():
            for data in data_loader:
                step_time_start = time.time()
                run_ctx = get_run_ctx()
                run_ctx.init_step(self._train_step)

                self.run_forward_step(data, run_ctx)

                self.print_step_info(
                    f"forward epoch {self.epoch}",
                    step_idx,
                    step_start_time=step_time_start,
                )
                step_idx += 1

        assert step_idx > 0, "No data in forward dataset"

        if forward_finish_hook := self.config.typed_value("forward_finish_hook", None):
            assert callable(forward_finish_hook)
            forward_finish_hook(get_run_ctx(), **{"__random_arg_%i" % int(random() * 100): None})
        print(
            "Finished forwarding in {:.3}s".format(time.time() - dataset_start_time),
            file=log.v3,
        )

    def _create_data_loader(self, dataset: Dataset) -> DataLoader:
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
        min_seq_length = self.config.typed_value("min_seq_length", None)  # type: Union[int,Dict[str,int],NumbersDict]
        max_seq_length = self.config.typed_value("max_seq_length", None)  # type: Union[int,Dict[str,int],NumbersDict]
        if (min_seq_length is not None) or (max_seq_length is not None):
            wrapped_dataset = data_pipeline.LenFilterDataPipe(
                wrapped_dataset, min_seq_length=min_seq_length, max_seq_length=max_seq_length
            )
        chunking_options = self.config.typed_value("chunking_options", None)
        if chunking_options:
            wrapped_dataset = data_pipeline.ChunkingIterDataPipe(wrapped_dataset, chunking_options)

        batch_size = self.config.typed_value("batch_size", 1)
        max_seqs = self.config.int("max_seqs", -1)
        batch_drop_last = self.config.bool("batch_drop_last", False)
        batches_dataset = data_pipeline.BatchingIterDataPipe(
            wrapped_dataset, batch_size=batch_size, max_seqs=max_seqs, drop_last=batch_drop_last
        )
        batches_dataset = dp.iter.Collator(
            batches_dataset, collate_fn=partial(data_pipeline.collate_batch, device=self._device)
        )

        num_workers = self.config.int("num_workers_per_gpu", 1)
        if num_workers > 1:
            assert dataset.supports_sharding(), "Dataset does not support sharding but 'num_workers_per_gpu' is > 0"

        return DataLoader(
            dataset=batches_dataset,
            batch_size=None,
            num_workers=num_workers,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    def run_train_step(
        self, data: dict[str, torch.Tensor], run_ctx: RunCtx, step_idx: int
    ) -> Tuple[Tensor, Dict[str, Loss]]:
        """
        :param data: model inputs for the step
        :param run_ctx: the current run ctx object
        :param step_idx: the current step index passed to the updater
        :return: total loss (weighted sum) calculated for the step, and individual losses as a name -> value mapping
        """
        assert isinstance(data, dict) and data
        data = self.move_data_to_device(data)
        sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
        with autocast(device_type=self._device, dtype=self._amp_dtype) if self._amp_dtype else nullcontext():
            self._train_step_func(model=self._model, data=data, run_ctx=run_ctx, **sentinel_kw)

        losses_dict = run_ctx.losses
        total_loss = run_ctx.total_loss()

        self._updater.step(total_loss, step_idx)

        return total_loss, losses_dict

    def run_eval_step(self, data: dict[str, torch.Tensor], run_ctx: RunCtx) -> Tuple[Tensor, Dict[str, Loss]]:
        """
        :param data: model inputs for the step
        :param run_ctx: the current run ctx object
        :return: total loss (weighted sum) calculated for the step, and individual losses as a name -> value mapping
        """
        assert isinstance(data, dict) and data
        data = self.move_data_to_device(data)
        sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
        # currently we only support the _train_step_func,
        # this can be an optional _eval_step_func later on
        with autocast(device_type=self._device, dtype=self._amp_dtype) if self._amp_dtype else nullcontext():
            self._train_step_func(model=self._model, data=data, run_ctx=run_ctx, **sentinel_kw)

        losses_dict = run_ctx.losses
        total_loss = run_ctx.total_loss()

        return total_loss, losses_dict

    def run_forward_step(self, data: dict[str, torch.Tensor], run_ctx: RunCtx):
        """
        :param data: model inputs for the step
        :param run_ctx: the current run ctx object
        """
        assert isinstance(data, dict) and data
        data = self.move_data_to_device(data)
        sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
        # currently we only support the _train_step_func,
        # this can be an optional _eval_step_func later on
        with autocast(device_type=self._device, dtype=self._amp_dtype) if self._amp_dtype else nullcontext():
            self._forward_step_func(model=self._model, data=data, run_ctx=run_ctx, **sentinel_kw)

    def _load_model(self, *, epoch: Optional[int], filename: Optional[str] = None):
        """
        Sets self._model to a torch.nn.Module.

        In case of running on CPU we move all objects to the CPU,
        otherwise we keep the original assignment.

        :param epoch: e.g. via BaseEngine.get_epoch_model()
        :param filename: Filename of checkpoint that needs to be loaded to initiliaze the parameters, e.g. coming from BaseEngine.get_epoch_model()
        """
        checkpoint_state = None
        if filename is not None:
            print("Load model %s" % (filename,), file=log.v4)
            load_filename = filename + self.get_file_postfix()
            if not os.path.exists(load_filename) and os.path.exists(filename):
                load_filename = filename
            checkpoint_state = torch.load(
                load_filename,
                map_location=self._device,
                weights_only=False,
            )

            if epoch is None:
                step = checkpoint_state["step"]
                epoch = checkpoint_state["epoch"]
            else:
                step = 0

        else:
            step = 0
            # TODO: the epoch handling might still be inconsistent
            epoch = 1
        self._train_step = step

        random_seed = self.config.int("random_seed", 42)
        random_seed = (epoch * 193939 + step * 19937 + random_seed * 27644437 + 479001599) % (2**31)
        torch.manual_seed(random_seed)

        init_load_run_ctx(device=self._device, engine=self, epoch=epoch)

        get_model_func = self.config.typed_value("get_model")
        assert get_model_func, "get_model not defined"
        sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
        self._model = get_model_func(epoch=epoch, step=step, **sentinel_kw)
        assert isinstance(self._model, torch.nn.Module)
        print("Model:", self._model, file=log.v4)
        params = sum([parameter.data.size().numel() for parameter in self._model.parameters()])
        print(f"Total number of parameters: {params}", file=log.v4)

        if checkpoint_state is not None:
            missing_keys, unexpected_keys = self._model.load_state_dict(checkpoint_state["model"], strict=False)
            if missing_keys and not self.config.bool("load_ignore_missing_vars", False):
                raise Exception(
                    "\n".join(
                        [
                            f"While loading model {filename}:",
                            "Unexpected key(s) in state_dict: " + ", ".join(map(repr, unexpected_keys)),
                            "Missing key(s) in state_dict: " + ", ".join(map(repr, missing_keys)),
                            "Any missing key is an error.",
                        ]
                    )
                )
            if unexpected_keys:
                print(
                    f"Note: While loading {filename}, unexpected key(s) in state_dict: "
                    + ", ".join(map(repr, unexpected_keys)),
                    file=log.v4,
                )

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
                preload_model_state = torch.load(opts["filename"], weights_only=False)
                if opts.get("checkpoint_key", "model") is not None:
                    # This can be used if an external checkpoint saves a checkpoint a different structure that just the
                    # model state dict. E.g., if a checkpoint is created using
                    # `torch.save({"model": model.state_dict(), "optimizer": optimizer.state)_dict(), ...})`
                    # we can set checkpoint_key = "model" to load the model.
                    # Currently, this only supports single level dicts, but it could be extended if needed.
                    preload_model_state = preload_model_state[opts.get("checkpoint_key", "model")]
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
                if missing_keys and not opts.get("ignore_missing", False):
                    prefix_keys = [key for key in self._model.state_dict() if key.startswith(opts.get("prefix", ""))]
                    missing_prefix_keys = set(prefix_keys).intersection(set(missing_keys))
                    assert not missing_prefix_keys, f"Missing keys and ignore_missing=False: {missing_prefix_keys}"
                print(f"Missing keys: {missing_keys}", file=log.v4)

        # https://github.com/rwth-i6/returnn/issues/1345
        del checkpoint_state
        gc.collect()
        self._model.to(self._device)

    def save_model(self):
        """
        Saves the state of self._model to file.
        """
        filename = self.get_epoch_model_filename() + ".pt"
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        print("Save model under %s" % (filename,), file=log.v4)
        torch.save({"model": self._model.state_dict(), "epoch": self.epoch, "step": self._train_step}, filename)

    def _load_optimizer(self, epoch, filename=None):
        """
        Loads a torch.optim.Optimizer from disk and uses it as the optimizer.
        This function is a wrapper to Updater.load_optimizer().

        :param int epoch: Epoch from which to load the optimizer state.
        :param filename (Optional): Filename of optimizer state
        """
        if filename is None:
            filename = self.get_epoch_model_filename(epoch=epoch - 1) + ".opt.pt"

        if os.path.isfile(filename):
            self._updater.load_optimizer(filename)
        elif self.config.bool("allow_missing_optimizer_checkpoint", False):
            print(
                "Warning: No optimizer state for the given checkpoint could be loaded. Continuing training with a fresh optimizer...",
                file=log.v4,
            )
        else:
            raise Exception(f"Optimizer file {filename} not found and allow_missing_optimizer_checkpoint is False")

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

    def _check_nonfinite_train_score(self, scores: Dict, accumulated_scores: Dict = None):
        """
        Checks if the score dictionary contains any nan or inf values
        :param scores: Dict containing score values
        :param accumulated_scores (optional): Dict containing accumulated score values. Defaults to None
        :raises Exception if nan or inf value is found.
        """
        if any((math.isnan(v) or math.isinf(v)) for v in scores.values()):
            print("Model seems broken, got inf or nan loss.", file=log.v1)
            if accumulated_scores is not None:
                print(f"Accumulated losses: {accumulated_scores}", file=log.v1)
            print(f"Step losses: {scores}", file=log.v1)
            raise Exception(f"Inf/nan loss in epoch {self.epoch}, step {self._train_step}.")

    def move_data_to_device(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Place Tensors on the desired devices.

        Currently moves all tensors to the target device, e.g. GPU.

        :param data: the data dict with tensors to be placed onto the desired devices
        :return: data placed on the desired device
        """
        return {k: v.to(self._device) if isinstance(v, Tensor) else v for k, v in data.items()}

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

    def print_step_info(
        self,
        report_prefix: str,
        step: int,
        step_start_time: float,
        total_loss: Optional[float] = None,
        loss_dict: Optional[NumbersDict] = None,
    ):
        """

        :param report_prefix:
        :param step:
        :param step_start_time:
        :param total_loss:
        :param loss_dict:
        """
        if log.verbose[5]:

            info = [report_prefix, "step:%i" % step, "step time: %.3f" % (time.time() - step_start_time)]
            if hasattr(self, "time_since_last_step"):
                info += ["total time: %.3f" % (time.time() - self.time_since_last_step)]
            if self._device == "cuda":
                t = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                r = torch.cuda.max_memory_reserved(0) / (1024**3)
                a = torch.cuda.max_memory_allocated(0) / (1024**3)
                info += ["GMEM: %.1f/%.1f/%.1f" % (a, r, t)]
            if total_loss is not None:
                info += ["total (grad) loss: %f" % total_loss]
            if loss_dict is not None:
                info += [self.format_loss_dict(loss_dict)]
            info = ", ".join(filter(None, info))
            print(info, file=log.v5)
            self.time_since_last_step = time.time()

    @staticmethod
    def format_loss_dict(loss_dict: NumbersDict):
        """
        :param loss_dict: the loss dict from train/eval
        """
        return ", ".join(["%s: %.5f" % (k, v) for k, v in sorted(loss_dict.items())])

    def _check_epoch_missing_eval(self):
        """
        Checks if there are outstanding tasks (eval_model) for the epoch,
        and executes them.
        """
        if not self.epoch:
            return
        if self.learning_rate_control.filename:
            for name, dataset in self.eval_datasets.items():
                if not self._is_dataset_evaluated(name=name):
                    # This can happen when we have a previous model but did not test it yet.
                    print(f"Last epoch model not yet evaluated on {name}. Doing that now.", file=log.v3)
                    self.eval_model(skip_already_evaluated=True)
                    break
