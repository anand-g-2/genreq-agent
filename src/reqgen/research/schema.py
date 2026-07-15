"""Pydantic models for Phase 2 research results."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

class PackageMetadata(BaseModel):
    """Metadata for a single package, as reported by PyPI."""
    name: str
    summary: Optional[str] = None
    latest_version: Optional[str] = None
    available_versions: list[str] = Field(default_factory=list)
    declared_dependencies: list[str] = Field(
        default_factory=list,
        description="Raw requires_dist strings from PyPI for the latest release ",
    )
    project_urls: dict[str, str] = Field(default_factory=dict)
    source: str = "pypi"
    fetch_warnings: list[str] = Field(default_factory=list)

class CompatNote(BaseModel):
    """A single line of text pulled from a README/release notes that looks
    like it's stating a version compatibility constraint."""
    package: str
    source_url: str
    note: str

class SearchResult(BaseModel):
    """A single result from the web search fallback tool."""
    title: str
    url: str
    snippet: str    