"""Tests for Phase 1 system probing.

Note: GPU-detection tests here focus on logic that doesn't require actual
NVIDIA hardware (compat table lookups, schema validation, graceful
degradation when no GPU is present). Full GPU-present behavior should be
spot-checked manually via `python scripts/run_probe.py` on a GPU machine.
"""

from __future__ import annotations

from reqgen.probe import build_system_profile
from reqgen.probe.compat_table import max_cuda_for_driver
from reqgen.probe.schema import CUDAInfo, GPUInfo, SystemProfile


class TestCompatTable:
    def test_known_driver_returns_expected_cuda(self):
        assert max_cuda_for_driver("535.104.05") == "12.2"

    def test_driver_above_max_known_falls_back_to_highest(self):
        # A driver newer than anything in the table should still resolve to
        # the highest known entry, not None — that's the whole point of the
        # >= threshold design.
        assert max_cuda_for_driver("999.00.00") == "12.8"

    def test_driver_below_table_range_returns_none(self):
        assert max_cuda_for_driver("300.00.00") is None

    def test_malformed_driver_string_returns_none(self):
        assert max_cuda_for_driver("not-a-version") is None
        assert max_cuda_for_driver("") is None


class TestSchema:
    def test_system_profile_defaults(self):
        profile = SystemProfile(has_nvidia_gpu=False)
        assert profile.gpus == []
        assert profile.probe_warnings == []
        assert isinstance(profile.cuda, CUDAInfo)

    def test_gpu_info_requires_core_fields(self):
        gpu = GPUInfo(
            index=0,
            name="NVIDIA GeForce RTX 3080",
            driver_version="535.104.05",
            vram_total_mb=10240,
            vram_free_mb=9800,
        )
        assert gpu.compute_capability is None  # optional field, not provided


class TestBuildSystemProfile:
    def test_runs_without_crashing_on_no_gpu_machine(self):
        # This sandbox/test environment has no NVIDIA GPU — the real value of
        # this test is confirming the whole probe pipeline degrades to
        # warnings instead of raising.
        profile = build_system_profile()
        assert isinstance(profile, SystemProfile)
        assert isinstance(profile.probe_warnings, list)