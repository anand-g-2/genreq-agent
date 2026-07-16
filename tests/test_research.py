"""Tests for Phase 2 research tools.

Network calls are mocked via monkeypatching httpx.get/post rather than hitting
real PyPI/GitHub/Tavily — keeps the suite fast and reliable in CI regardless
of rate limits or API keys.
"""

from __future__ import annotations

import httpx
import pytest

from reqgen.research.github_client import extract_compat_notes
from reqgen.research.pypi_client import get_pypi_metadata
from reqgen.research.schema import CompatNote
from reqgen.research.web_search import web_search_compat


class _FakeResponse:
    def __init__(self, status_code: int, json_data: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data

class TestPypiClient:
    """Tests for get_pypi_metadata, which is the main entry point for Phase 2 research."""
    def test_successful_fetch_parses_dependencies(self, monkeypatch):
        """Simulate a successful PyPI JSON API response and check parsing."""
        fake_data = {
            "info": {
                "summary": "Fake package",
                "version": "1.2.3",
                "requires_dist": ["torch>=1.8.0", "numpy<2.0"],
                "project_urls": {"Homepage": "https://example.com"},
            },
            "releases": {
                "1.0.0": [{"yanked": False}],
                "1.2.3": [{"yanked": False}],
                "0.9.0": [{"yanked": True}],  # should be excluded
            },
        }
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(200, fake_data))

        result = get_pypi_metadata("fake-package")

        assert result.latest_version == "1.2.3"
        assert "torch>=1.8.0" in result.declared_dependencies
        assert "1.0.0" in result.available_versions
        assert "0.9.0" not in result.available_versions  # fully yanked, excluded
        assert result.fetch_warnings == []

    def test_404_produces_warning_not_exception(self, monkeypatch):
        """Simulate a 404 response and check that we get a warning instead of an exception."""
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(404))

        result = get_pypi_metadata("nonexistent-package")

        assert result.available_versions == []
        assert len(result.fetch_warnings) == 1
        assert "not found" in result.fetch_warnings[0].lower()

    def test_network_error_produces_warning_not_exception(self, monkeypatch):
        """Simulate a network error and check that we get a warning instead of an exception."""
        def raise_error(*args, **kwargs):
            raise httpx.RequestError("connection failed")

        monkeypatch.setattr(httpx, "get", raise_error)

        result = get_pypi_metadata("some-package")

        assert result.fetch_warnings
        assert "network error" in result.fetch_warnings[0].lower()


class TestGithubCompatExtraction:
    """Tests for extract_compat_notes, which is the main entry point for Phase 2 GitHub research."""
    def test_extracts_lines_with_compat_keywords(self):
        """Simulate a README with some lines that look like version compatibility notes 
        and check that they are extracted."""
        text = (
            "# My Package\n"
            "This is a great tool.\n"
            "Requires torch>=2.1.0 and CUDA 12.1 or newer.\n"
            "Also needs pillow (any version).\n"
            "tensorrt support is experimental.\n"
        )
        notes = extract_compat_notes("mypackage", text, "https://github.com/x/y")

        assert len(notes) == 2  # the torch/CUDA line and the tensorrt line
        assert all(isinstance(n, CompatNote) for n in notes)
        assert any("torch" in n.note.lower() for n in notes)

    def test_no_matches_returns_empty_list(self):
        """Simulate a README with no version compatibility info and 
        check that an empty list is returned."""
        text = "Just a regular readme with no version info at all."
        notes = extract_compat_notes("mypackage", text, "https://github.com/x/y")
        assert notes == []


class TestWebSearch:
    """Tests for web_search_compat, which is the main entry point for Phase 2 web search fallback."""
    def test_missing_api_key_returns_warning_not_exception(self, monkeypatch):
        """If TAVILY_API_KEY isn't set, the function should return a warning instead of raising."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        results, warnings = web_search_compat("torch cuda 12.2 compatibility")

        assert results == []
        assert len(warnings) == 1
        assert "TAVILY_API_KEY" in warnings[0]