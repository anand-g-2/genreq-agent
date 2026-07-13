"""System probing: GPU, driver, and CUDA toolkit detection (Phase 1)."""

from __future__ import annotations

from reqgen.probe.cuda import detect_cuda
from reqgen.probe.gpu import detect_gpus
from reqgen.probe.schema import CUDAInfo, GPUInfo, SystemProfile

__all__ = ["build_system_profile", "SystemProfile", "GPUInfo", "CUDAInfo"]


def build_system_profile() -> SystemProfile:
    """Run all Phase 1 probes and assemble a single SystemProfile."""
    gpus, gpu_warnings = detect_gpus()

    driver_version = gpus[0].driver_version if gpus else None
    cuda_info, cuda_warnings = detect_cuda(driver_version)

    return SystemProfile(
        has_nvidia_gpu=bool(gpus),
        gpus=gpus,
        cuda=cuda_info,
        probe_warnings=[*gpu_warnings, *cuda_warnings],
    )