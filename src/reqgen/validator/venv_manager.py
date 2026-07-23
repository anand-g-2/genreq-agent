"""Throwaway virtual environment management for dependency validation."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

class VenvCreationError(RuntimeError):
    """Raised when `uv venv` fails to create a throwaway environment."""

@contextmanager
def temp_venv(python_version: str | None = None, timeout: float = 60.0):
    """
    Create a throwaway virtual environment via `uv venv`, yield its path, and
    guarantee cleanup afterward regardless of what happens inside the block.

    Usage:
        with temp_venv() as venv_dir:
            python_path = venv_python_path(venv_dir)
            result = install_requirements(python_path, ["torch==2.1.0"])

    Args:
        python_version: optional Python version to request (e.g. "3.10"),
            passed to `uv venv --python <version>`. If omitted, uv uses
            whatever Python it finds on PATH.
        timeout: seconds to wait for `uv venv` to finish before giving up.
    """
    # Create a temporary directory for the virtual environment in the system's 
    # temp folder with a predefined prefix for easier identification
    venv_dir = Path(tempfile.mkdtemp(prefix="reqgen_validator_"))

    cmd = ["uv", "venv", str(venv_dir)]
    if python_version:
        cmd += ["--python", python_version]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        shutil.rmtree(venv_dir, ignore_errors=True)
        raise VenvCreationError(
            "`uv` is not installed or not on PATH. Install it: "
            "https://docs.astral.sh/uv/getting-started/installation/"
        ) from e
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(venv_dir, ignore_errors=True)
        raise VenvCreationError(f"`uv venv` timed out after {timeout}s.") from e

    if result.returncode != 0:
        shutil.rmtree(venv_dir, ignore_errors=True)
        raise VenvCreationError(f"`uv venv` failed:\n{result.stderr}")

    try:
        yield venv_dir
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)

def venv_python_path(venv_dir: Path) -> Path:
    """Return the path to the venv's python executable, handling OS layout differences."""
    candidate_unix = venv_dir / "bin" / "python"
    candidate_windows = venv_dir / "Scripts" / "python.exe"
    return candidate_unix if candidate_unix.exists() else candidate_windows
