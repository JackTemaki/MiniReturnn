"""
Run context

We can either be in param-init stage,
or in the main training loop,
or forwarding loop.
"""

from __future__ import annotations
from typing import Optional, Union, Dict, TYPE_CHECKING
from dataclasses import dataclass

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .engine import Engine

__all__ = ["RunCtx", "Loss", "get_run_ctx", "init_train_step_run_ctx", "init_forward_step_run_ctx"]


_run_ctx = None  # type: Optional[RunCtx]


def reset_run_ctx():
    """
    If we get out of a train step or forward step.
    """
    global _run_ctx
    _run_ctx = None


def init_train_step_run_ctx(device: str, engine: Engine):
    """
    Call this at the beginning of a new train step.
    """
    global _run_ctx
    _run_ctx = RunCtx(stage="train_step", device=device, engine=engine)


def init_forward_step_run_ctx(device: str, engine: Engine):
    """
    Call this at the beginning of a new forward step.
    """
    global _run_ctx
    _run_ctx = RunCtx(stage="forward_step", device=device, engine=engine)


def get_run_ctx() -> RunCtx:
    """
    :return: current run context, see :class:`RunCtx`
    """
    global _run_ctx, _init_run_ctx
    if _run_ctx is None:
        if _init_run_ctx is None:
            _init_run_ctx = RunCtx(stage="init")
        return _init_run_ctx
    return _run_ctx


class RunCtx:
    """
    We can either be in param-init stage,
    or in the main training (or eval) loop,
    or forwarding loop (doing recog, beam search, dumping whatever, ...).

    In training/eval, we expect that some loss is being defined via mark_as_loss().
    """

    def __init__(self, *, device: str, stage: str, engine: Engine):
        """
        :param device:
        :param stage:
            - "init"
            - "train_step", also for eval, for mark_as_loss and get_total_loss
            - "forward_step", for mark_as_output
        :param engine: reference to the engine
        """
        self.device = device
        self.stage = stage
        self._engine = engine
        self.losses: Dict[str, Loss] = {}

    @property
    def engine(self):
        return self._engine

    def init_step(self):
        """ """
        self.losses = {}

    def mark_as_loss(
        self, loss: Tensor, name: str, *, scale: float = 1.0, inv_norm_factor: Optional[Tensor] = None
    ) -> None:
        """
        Mark the given loss tensor as a loss.

        :param loss: scalar loss
        :param name: name of the loss. this name is used for reporting by RETURNN, and also for LR scheduling.
        :param scale: scale the loss by this factor for the training optimizer
        :param inv_norm_factor: scalar norm factor, e.g. number of frames in the batch
        """
        assert self.stage == "train_step"
        assert name not in self.losses
        self.losses[name] = Loss(
            loss=loss,
            name=name,
            scale=scale,
            inv_norm_factor=inv_norm_factor,
        )

    def total_loss(self) -> Union[Tensor]:
        """
        :return: total loss, as it is used for backpropagation
        """
        assert self.stage == "train_step"
        assert self.losses, "call RunCtx.mark_as_loss(...)"
        loss = torch.zeros((), device=self.device)
        for name, loss_obj in self.losses.items():
            if loss_obj.scale == 0.0:
                continue
            loss += loss_obj.loss * loss_obj.scale / (loss_obj.inv_norm_factor or 1)
        return loss


@dataclass
class Loss:
    """
    Loss via :func:`RunCtx.mark_as_loss`.

    We collect all relevant information here.
    """

    loss: Tensor
    name: str

    scale: float = 1.0
    inv_norm_factor: Optional[Tensor] = None
