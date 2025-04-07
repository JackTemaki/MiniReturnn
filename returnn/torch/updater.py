"""
This module covers the optimizer (SGD, Adam, etc) logic,
and model param update logic in general.
"""

from __future__ import annotations

import gc
import torch
from torch.cuda import amp
from typing import Any, Dict, List, Optional, Union

from returnn.log import log
from returnn.config import Config

_OptimizerClassesDictInitialized = False
_OptimizerClassesDict: Dict[str, type(torch.optim.Optimizer)] = {}


def _init_optimizer_classes_dict():
    """
    Initializes a global dictionary with all optimizers available in PyTorch.
    """
    global _OptimizerClassesDictInitialized
    if _OptimizerClassesDictInitialized:
        return
    _OptimizerClassesDictInitialized = True
    for name, cls in list(vars(torch.optim).items()):
        assert isinstance(name, str)
        # Check if cls is a valid subclass of torch.optim.Optimizer
        if not isinstance(cls, type) or not issubclass(cls, torch.optim.Optimizer):
            continue
        assert name not in _OptimizerClassesDict
        _OptimizerClassesDict[name.lower()] = cls


def get_optimizer_class(class_name: Union[str, type(torch.optim.Optimizer)]) -> type(torch.optim.Optimizer):
    """
    :param class_name: Optimizer data, e.g. "adam", torch.optim.Adam...
    :return: Optimizer class
    """
    _init_optimizer_classes_dict()
    if isinstance(class_name, type):
        assert issubclass(class_name, torch.optim.Optimizer)
    elif callable(class_name):
        raise ValueError("Callable 'class' in 'optimizer' is no longer supported with version 0.4")
    else:
        assert isinstance(class_name, str)
        assert (
            class_name.lower() in _OptimizerClassesDict
        ), "%s not found in the available torch optimizers list: %s." % (
            class_name.lower(),
            ", ".join("'%s'" % key for key in _OptimizerClassesDict),
        )
        class_name = _OptimizerClassesDict[class_name.lower()]

    return class_name


def _get_class_init_kwargs(optim_class: type[torch.optim.Optimizer]) -> List[str]:
    """
    Obtains the keyword arguments of the class provided as parameter that the user can add to their optimizer.

    :param optim_class: Optimizer class.
    :return: Keyword arguments of the provided class.
    """
    from returnn.util.basic import collect_class_init_kwargs

    optim_class_init_kwargs = collect_class_init_kwargs(optim_class)
    # We already provide params by default, remove it so that the user doesn't add it to the optimizer dict.
    optim_class_init_kwargs.remove("params")

    return optim_class_init_kwargs


class Updater(object):
    """
    Wraps a torch.optim.Optimizer, and extends it by some further functionality.
    """

    def __init__(self, *, config: Config, network: torch.nn.Module, device: str, initial_learning_rate: float = 1.0):
        """
        :param config: config defining the training conditions.
        :param network: PyTorch Module defining the network.
        :param device: The currently used device
        :param initial_learning_rate:
        """
        self.config = config
        self.learning_rate = initial_learning_rate
        self.network = network
        self._device = device
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self._grad_scaler: Optional[amp.GradScaler] = None

        self._grad_clip = self.config.float("gradient_clip", None)
        self._grad_clip_norm = self.config.float("gradient_clip_norm", None)
        self._grad_clip_norm_invalid_gradient_threshold = self.config.int("gradient_clip_norm_invalid_gradient_threshold", 3)
        self._num_invalid_gradients = 0

        self._accum_grad_multiple_step = config.int("accum_grad_multiple_step", 1)

    def create_grad_scaler(self):
        """
        Creates an AMP gradient scaler
        """
        self._grad_scaler = amp.GradScaler()

    def set_learning_rate(self, value: float):
        """
        Updates the learning rate of the optimizer at each (sub)epoch.

        :param value: New learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = value

    def create_optimizer(self):
        """
        Creates an optimizer and stores it in self.optimizer.
        """
        optimizer_opts = self.config.typed_value("optimizer", None)
        if optimizer_opts is None:
            raise ValueError("config field 'optimizer' needs to be set explicitely for the Torch backend")
        self.optimizer = self._create_optimizer(optimizer_opts)

    def load_optimizer(self, filename: str):
        """
        Loads a torch.optim.Optimizer from disk and stores it in self.optimizer.

        :param filename: File from which to load the optimizer state.
        """
        print("Load optimizer %s" % filename, file=log.v4)
        optimizer_state = torch.load(filename, map_location=self._device)
        self.optimizer.load_state_dict(optimizer_state)
        # https://github.com/rwth-i6/returnn/issues/1345
        del optimizer_state
        gc.collect()

    def save_optimizer(self, filename: str):
        """
        Saves the state of self.optimizer to a file.

        :param filename: File in which to save the optimizer state.
        """
        print("Save optimizer under %s" % filename, file=log.v4)
        torch.save(self.optimizer.state_dict(), filename)

    def get_optimizer(self) -> torch.optim.Optimizer:
        """
        unused within RETURNN, can be used via the config
        """
        return self.optimizer

    def step(self, loss: torch.Tensor, step_idx: int):
        """
        Executes backpropagation of the loss, handles AMP if necessary and calls the updater step.

        For details, explanations and the background for this code see:
        https://pytorch.org/docs/stable/notes/amp_examples.html

        :param loss: The total loss of the current step
        :param step_idx: index of the current step, used e.g. for gradient accumulation
        """

        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step_idx + 1) % self._accum_grad_multiple_step == 0:
            if self._grad_scaler is not None:
                # unscale gradient if using AMP
                self._grad_scaler.unscale_(self.optimizer)

            # modify the gradient only after accumulation and unscaling
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_value_(self.network.parameters(), self._grad_clip)

            has_invalid_gradient = False
            if self._grad_clip_norm is not None:
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self._grad_clip_norm)
                has_invalid_gradient = torch.isnan(norm) or torch.isinf(norm)
                if has_invalid_gradient:
                    self._num_invalid_gradients += 1
                    if self._num_invalid_gradients >= self._grad_clip_norm_invalid_gradient_threshold:
                        raise RuntimeError("Got %i invalid gradients in succession, abort training" % self._num_invalid_gradients)

            # perform the actual gradient update on the parameters
            if self._grad_scaler is not None:
                if not has_invalid_gradient:
                    self._grad_scaler.step(self.optimizer)

                # even if we have an invalid gradient and drop the step we need to call update
                self._grad_scaler.update()
            elif not has_invalid_gradient:
                self.optimizer.step()

            # remove gradients at the end of gradient accumulation
            self.optimizer.zero_grad()

    def _create_optimizer(self, optimizer_opts: Union[Dict[str, Any], torch.optim.Optimizer]) -> torch.optim.Optimizer:
        """
        Returns a valid optimizer considering the dictionary given by the user in the config.

        :param optimizer_opts: Optimizer configuration specified by the user.
            If it's a dict, it must contain "class" with the optimizer name or callable.
        :return: A valid optimizer.
        """
        lr = self.learning_rate

        # If the parameter is already a valid optimizer, return it without further processing
        if isinstance(optimizer_opts, torch.optim.Optimizer):
            return optimizer_opts
        elif callable(optimizer_opts):
            raise ValueError("Callable 'optimizer' is no longer supported with version 0.4")
        else:
            if not isinstance(optimizer_opts, dict):
                raise ValueError("'optimizer' must of type dict or torch.optim.Optimizer instance.")
            if "class" not in optimizer_opts:
                raise ValueError("'class' field of 'optimizer' dict was not set (use e.g. 'SGD', 'Adam', ...)")

        # Resolve the optimizer class
        optim_class_name = optimizer_opts.pop("class")
        optim_class = get_optimizer_class(optim_class_name)

        # Resolve the optimizer arguments
        opt_kwargs = optimizer_opts.copy()
        optim_class_init_kwargs = _get_class_init_kwargs(optim_class)
        # epsilon is named eps in torch.
        # If the user specified it as epsilon, parse it as eps for the optimizer
        if "eps" in optim_class_init_kwargs and "epsilon" in opt_kwargs:
            opt_kwargs["eps"] = opt_kwargs.pop("epsilon")
        if "learning_rate" in optimizer_opts:
            raise ValueError("'learning_rate' should be set outside of the 'optimizer' dict.")
        lr = lr * optimizer_opts.get("learning_rate_multiplier", 1.0)
        opt_kwargs["lr"] = lr

        param_groups = self._get_optimizer_param_groups(optim_class, opt_kwargs)
        optimizer = optim_class(param_groups, **opt_kwargs)
        print("Optimizer: %s" % optimizer, file=log.v1)
        assert isinstance(optimizer, torch.optim.Optimizer)

        return optimizer

    # noinspection PyUnusedLocal
    def _get_optimizer_param_groups(
        self, optim_class: type(torch.optim.Optimizer), optimizer_opts: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        TODO: the current behavior is to not exclude any parameter at all, unclear if this is desired

        :param optim_class: Optimizer class, currently unused
        :param optimizer_opts: Optimizer configuration specified by the user, currently unused
        :return: List of configurations for the different sets of parameters.
        """
        network_params = self.network.parameters()
        return [{"params": network_params}]
