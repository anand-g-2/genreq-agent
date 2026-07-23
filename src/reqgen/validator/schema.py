"""Pydantic models for Phase 3 validation results."""

from __future__ import annotations
from pydantic import BaseModel, Field

class ConflictInfo(BaseModel):
    """Represents a package installation conflict.

    Unlike simple "A conflicts with B" errors, real dependency problems often 
    involve multiple packages and version constraints (e.g., "package X needs 
    version >=1.0 of package Y, but you have version 0.5"). 

    This structure stores ALL packages involved in the conflict, not just two.
    """
    reason: str = Field(
        description="One of: 'no_solution', 'no_matching_distribution', 'unrecognized_error'."
    )
    packages_involved: list[str] = Field(default_factory=list)
    detail: str = Field(description="The extracted conflict message, whitespace-normalized.")


class ValidationResult(BaseModel):
    """Result of attempting to install/resolve a candidate requirements list."""
    success: bool
    requirements: list[str]
    stdout: str
    stderr: str
    return_code: int
    duration_seconds: float
    conflicts: list[ConflictInfo] = Field(default_factory=list)
