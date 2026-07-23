"""Runs installs inside a throwaway venv (via `uv pip install`) and captures raw output."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from reqgen.validator.schema import ValidationResult

def install_requirements(
    venv_python: Path,
    requirements: list[str],
    extra_index_urls: list[str] | None = None,
    dry_run: bool = True,
    timeout: float = 600.0,
) -> ValidationResult:
    """
    Attempt to install a candidate requirements list into a throwaway venv
    using `uv pip install`, and capture the raw result for parsing.

    Args:
        venv_python: path to the venv's python executable
        requirements: requirement strings, e.g. ["torch==2.1.0", "boxmot>=10.0.0"].
        extra_index_urls: additional package index URLs — needed for CUDA-specific
            wheels, e.g. "https://download.pytorch.org/whl/cu121" for torch built
            against CUDA 12.1.
        dry_run: if True, uses `uv pip install --dry-run`, which resolves the full
            dependency graph WITHOUT downloading or installing anything. This is
            fast and catches version conflicts, but it does NOT catch runtime
            failures — e.g. a wheel that resolves and installs cleanly but
            segfaults on `import torch` because of a CUDA/driver mismatch.
            Set dry_run=False for that deeper guarantee, at the cost of a much
            slower validation pass.
        timeout: seconds to allow the install/resolution to run before giving up.
    """
    cmd = ["uv", "pip", "install", "--python", str(venv_python)]

    if dry_run:
        cmd.append("--dry-run")

    for url in extra_index_urls or []:
        cmd += ["--extra-index-url", url]

    cmd += list(requirements)

    start = time.monotonic()
    timed_out = False
    stdout, stderr = "", ""
    return_code = -1

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return_code = result.returncode
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\n[reqgen] Install timed out after {timeout}s."

    duration = time.monotonic() - start

    return ValidationResult(
        success=(return_code == 0 and not timed_out),
        requirements=list(requirements),
        stdout=stdout,
        stderr=stderr,
        return_code=return_code,
        duration_seconds=round(duration, 2),
    )
