"""
Static NVIDIA driver -> max supported CUDA version table.

Source: NVIDIA's official CUDA compatibility documentation
(https://docs.nvidia.com/deploy/cuda-compatibility/). This table only tracks
major driver branches and is NOT guaranteed to stay current — NVIDIA ships
new CUDA major/minor versions periodically, and this file won't update itself.

Treat `max_cuda_for_driver()` output as a strong hint for the LLM research
and validation stages in later phases, not as ground truth. If you hit a
driver version not covered here, check the docs link above and add a row.

TODO (Phase 2): replace/augment this static table with a live-scraped
version from NVIDIA's docs, falling back to this table if scraping fails.
"""

from __future__ import annotations

# Sorted descending by minimum driver major version.
# Each entry: (min_driver_major, max_cuda_version_supported)
_DRIVER_TO_MAX_CUDA: list[tuple[int, str]] = [
    (570, "12.8"),
    (560, "12.6"),
    (555, "12.5"),
    (550, "12.4"),
    (545, "12.3"),
    (535, "12.2"),
    (530, "12.1"),
    (525, "12.0"),
    (520, "11.8"),
    (515, "11.7"),
    (510, "11.6"),
    (495, "11.5"),
    (470, "11.4"),
    (460, "11.2"),
    (450, "11.0"),
]


def max_cuda_for_driver(driver_version: str) -> str | None:
    """
    Look up the max CUDA version supported by a given NVIDIA driver version string.

    Example:
        max_cuda_for_driver("535.104.05") -> "12.2"

    Returns None if the driver major version isn't recognized — either older
    than this table goes back, or newer than this table has been updated for.
    """
    try:
        major = int(driver_version.strip().split(".")[0])
    except (ValueError, IndexError, AttributeError):
        return None

    for min_major, max_cuda in _DRIVER_TO_MAX_CUDA:
        if major >= min_major:
            return max_cuda

    return None