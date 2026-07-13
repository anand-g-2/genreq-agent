#!/usr/bin/env python3
"""
Standalone CLI harness for Phase 1: run the system probe and print results.

Usage:
    python scripts/run_probe.py
    python scripts/run_probe.py --json     # raw JSON output only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly without `pip install -e .` first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reqgen.probe import build_system_profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe this system's GPU/CUDA setup.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON only, no formatting.")
    args = parser.parse_args()

    profile = build_system_profile()

    if args.json:
        print(profile.model_dump_json(indent=2))
        return

    print("=== System Probe Results ===\n")

    if not profile.has_nvidia_gpu:
        print("No NVIDIA GPU detected.")
    else:
        for gpu in profile.gpus:
            print(f"GPU {gpu.index}: {gpu.name}")
            print(f"  Driver version:      {gpu.driver_version}")
            print(f"  VRAM:                {gpu.vram_free_mb} MB free / {gpu.vram_total_mb} MB total")
            print(f"  Compute capability:  {gpu.compute_capability or 'unknown'}")
            print()

    print("CUDA:")
    print(f"  nvcc (toolkit) version:   {profile.cuda.nvcc_version or 'not found'}")
    print(
        f"  torch built for CUDA:     "
        f"{profile.cuda.torch_cuda_version or 'torch not installed / CPU build'}"
    )
    print(f"  max CUDA per driver:      {profile.cuda.max_supported_cuda or 'unknown'}")

    if profile.probe_warnings:
        print("\nWarnings:")
        for w in profile.probe_warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()