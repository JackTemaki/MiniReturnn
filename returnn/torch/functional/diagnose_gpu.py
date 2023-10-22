"""
Diagnostic functions for GPU information, failings, memory usage, etc.
"""

from __future__ import annotations
from typing import List
import os
import sys
import subprocess
import torch
import better_exchook


def diagnose_no_gpu() -> List[str]:
    """
    Diagnose why we have no GPU.
    Print to stdout, but also prepare summary strings.

    :return: summary strings
    """
    # Currently we assume Nvidia CUDA here, but once we support other backends (e.g. ROCm),
    # first check which backend is most reasonable here.

    res = []
    res += [f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}"]
    res += [f"CUDA available: {torch.cuda.is_available()}"]
    res += [f"Visible CUDA devices: {torch.cuda.device_count()}"]

    try:
        torch.cuda.init()
    except Exception as exc:
        print("torch.cuda.init() failed:", exc)
        better_exchook(*sys.exc_info(), debugshell=False)
        res.append(f"torch.cuda.init() failed: {type(exc).__name__} {exc}")

    try:
        subprocess.check_call(["nvidia-smi"])
    except Exception as exc:
        print("nvidia-smi failed:", exc)
        better_exchook(*sys.exc_info(), debugshell=False)
        res.append(f"nvidia-smi failed")

    return res
