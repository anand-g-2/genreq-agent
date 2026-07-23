"""Validator: throwaway venv creation, install attempts, error parsing (Phase 3)."""

from __future__ import annotations

from reqgen.validator.installer import install_requirements
from reqgen.validator.parser import parse_install_errors
from reqgen.validator.schema import ConflictInfo, ValidationResult
from reqgen.validator.venv_manager import VenvCreationError, temp_venv, venv_python_path

__all__ = [
    "validate_requirements",
    "ValidationResult",
    "ConflictInfo",
    "VenvCreationError",
    "temp_venv",
    "venv_python_path",
    "install_requirements",
    "parse_install_errors",
]


def validate_requirements(
    requirements: list[str],
    extra_index_urls: list[str] | None = None,
    python_version: str | None = None,
    dry_run: bool = True,
) -> ValidationResult:
    """
    End-to-end Phase 3 entry point: create a throwaway venv, attempt to
    install/resolve the given requirements, parse any errors, and guarantee
    the venv is cleaned up regardless of outcome.
    """
    with temp_venv(python_version=python_version) as venv_dir:
        python_path = venv_python_path(venv_dir)
        result = install_requirements(
            venv_python=python_path,
            requirements=requirements,
            extra_index_urls=extra_index_urls,
            dry_run=dry_run,
        )

    if not result.success:
        result.conflicts = parse_install_errors(result.stderr)

    return result