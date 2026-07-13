"""Pydantic models for Phase 1 system probing results."""
 
from __future__ import annotations
 
from typing import Optional
 
from pydantic import BaseModel, Field
 
 
class GPUInfo(BaseModel):
    """Details for a single detected NVIDIA GPU."""
 
    index: int
    name: str
    driver_version: str
    vram_total_mb: int
    vram_free_mb: int
    compute_capability: Optional[str] = Field(
        default=None,
        description="e.g. '8.6' for an RTX 3080. None if NVML couldn't report it.",
    )
 
 
class CUDAInfo(BaseModel):
    """CUDA toolkit / build information gathered from various sources."""
 
    nvcc_version: Optional[str] = Field(
        default=None,
        description="CUDA toolkit version reported by `nvcc --version`, if installed.",
    )
    torch_cuda_version: Optional[str] = Field(
        default=None,
        description="CUDA version PyTorch was built against, if torch is importable.",
    )
    max_supported_cuda: Optional[str] = Field(
        default=None,
        description="Max CUDA version supported by the installed driver, per the local compat table.",
    )
 
 
class SystemProfile(BaseModel):
    """Complete Phase 1 output: everything downstream phases need about this machine."""
 
    has_nvidia_gpu: bool
    gpus: list[GPUInfo] = Field(default_factory=list)
    cuda: CUDAInfo = Field(default_factory=CUDAInfo)
    probe_warnings: list[str] = Field(
          default_factory=list,
        description="Non-fatal issues encountered during probing (missing tools, unknown driver, etc).",
    )
 