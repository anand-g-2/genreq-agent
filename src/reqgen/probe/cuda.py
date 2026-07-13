"""CUDA toolkit / driver compatibility detection."""

from __future__ import annotations

import re
import subprocess

from reqgen.probe.compat_table import max_cuda_for_driver
from reqgen.probe.schema import CUDAInfo


def _get_nvcc_version() -> str | None:
    """Return the CUDA toolkit version from `nvcc --version`, or None if not installed."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    match = re.search(r"release (\d+\.\d+)", result.stdout)
    return match.group(1) if match else None


def _get_torch_cuda_version() -> str | None:
    """Return the CUDA version PyTorch was built against, if torch is importable."""
    try:
        import torch
    except ImportError:
        return None

    return torch.version.cuda  # None if this is a CPU-only build


def detect_cuda(driver_version: str | None) -> tuple[CUDAInfo, list[str]]:
    """
    Gather CUDA toolkit + PyTorch CUDA build info, and cross-reference the
    installed driver against the local driver->max-CUDA compatibility table.

    Args:
        driver_version: the NVIDIA driver version string from gpu.detect_gpus(),
            or None if no GPU/driver was found.
    """
    warnings: list[str] = []

    nvcc_version = _get_nvcc_version()
    if nvcc_version is None:
        warnings.append(
            "nvcc not found on PATH — CUDA toolkit may not be installed, or only the "
            "driver-bundled runtime is present (common in Docker/cloud GPU setups)."
        )

    torch_cuda_version = _get_torch_cuda_version()

    max_supported = None
    if driver_version:
        max_supported = max_cuda_for_driver(driver_version)
        if max_supported is None:
            warnings.append(
                f"Driver version {driver_version} not found in the local compat table — "
                "it may be newer than this table covers. Verify against NVIDIA's official "
                "CUDA compatibility docs before trusting resolved requirements."
            )

    return (
        CUDAInfo(
            nvcc_version=nvcc_version,
            torch_cuda_version=torch_cuda_version,
            max_supported_cuda=max_supported,
        ),
        warnings,
    )