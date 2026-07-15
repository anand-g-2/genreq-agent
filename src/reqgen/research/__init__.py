"""Research tools: PyPI metadata, GitHub compat notes, web search fallback (Phase 2)."""

from __future__ import annotations

from reqgen.research.github_client import extract_compat_notes, get_latest_release_notes, get_readme_text
from reqgen.research.pypi_client import get_pypi_metadata
from reqgen.research.schema import CompatNote, PackageMetadata, SearchResult
from reqgen.research.web_search import web_search_compat

__all__ = [
    "get_pypi_metadata",
    "get_readme_text",
    "get_latest_release_notes",
    "extract_compat_notes",
    "web_search_compat",
    "PackageMetadata",
    "CompatNote",
    "SearchResult",
    "research_package",
]


def research_package(
    package_name: str, github_owner: str | None = None, github_repo: str | None = None
) -> tuple[PackageMetadata, list[CompatNote], list[str]]:
    """
    Convenience function combining PyPI metadata + GitHub README/release compat
    notes for a single package. This is the shape an agent node in Phase 4 will
    call directly.

    Args:
        package_name: PyPI project name.
        github_owner, github_repo: if known, enables README/release scanning.
            If omitted, only PyPI metadata is returned (still useful on its own).

    Returns:
        (metadata, compat_notes, warnings)
    """
    warnings: list[str] = []

    metadata = get_pypi_metadata(package_name)
    warnings.extend(metadata.fetch_warnings)

    compat_notes: list[CompatNote] = []

    if github_owner and github_repo:
        readme_text, readme_warnings = get_readme_text(github_owner, github_repo)
        warnings.extend(readme_warnings)
        if readme_text:
            readme_url = f"https://github.com/{github_owner}/{github_repo}#readme"
            compat_notes.extend(extract_compat_notes(package_name, readme_text, readme_url))

        release_text, release_warnings = get_latest_release_notes(github_owner, github_repo)
        warnings.extend(release_warnings)
        if release_text:
            release_url = f"https://github.com/{github_owner}/{github_repo}/releases/latest"
            compat_notes.extend(extract_compat_notes(package_name, release_text, release_url))

    return metadata, compat_notes, warnings