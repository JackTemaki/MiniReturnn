"""
mainly a wrapper from the upstream RETURNN to tool interface to a custom export function defined in the config.
Contains no logic except config init and model logic, expects the export function to do the actual export.
"""
import torch
from typing import Callable, Optional, Dict
import argparse

import _setup_returnn_env  # noqa
from returnn.config import Config
from returnn.torch.context import init_load_run_ctx

def main():
    """
    Main entry point, exactly follows the interface of upstream RETURNN
    """
    parser = argparse.ArgumentParser(description="Converts a RF/PT module to ONNX.")
    parser.add_argument(
        "config",
        type=str,
        help="Filename to config file. Must have `get_model()` and `forward_step()`. Can optionally have `export()`.",
    )
    parser.add_argument("checkpoint", type=str, help="Checkpoint to RF module, considering the backend.")
    parser.add_argument("out_onnx_filename", type=str, help="Filename of the final ONNX model.")
    parser.add_argument("--verbosity", default=4, type=int, help="does nothing, only for interface consistency")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' (default) or 'gpu'.")
    args = parser.parse_args()

    config = Config()
    config.load_file(args.config)

    model_state = torch.load(args.checkpoint, map_location=torch.device(args.device))
    if isinstance(model_state, dict):
        epoch = model_state["epoch"]
        step = model_state["step"]
        model_state = model_state["model"]
    else:
        epoch = 1
        step = 0

    init_load_run_ctx(device=args.device, engine=None, epoch=epoch)

    get_model_func = config.typed_value("get_model")
    assert get_model_func, "get_model not defined"
    model = get_model_func(epoch=epoch, step=step, onnx_export=True)
    assert isinstance(model, torch.nn.Module)

    model.load_state_dict(model_state)

    export_func = config.typed_value("export")

    export_func(
        model=model,
        args=None,
        f=args.out_onnx_filename,
    )


if __name__ == "__main__":
    main()
