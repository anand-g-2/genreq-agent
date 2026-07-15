"""PyPI JSON API client for package metadata."""

from __future__ import annotations
import httpx
from reqgen.research.schema import PackageMetadata

PYPI_BASE_URL = "https://pypi.org/pypi"

def get_pypi_metadata(package_name: str, timeout: float = 10.0) -> PackageMetadata:
    """
    Fetch package metadata from the PyPI JSON API.

    Args:
        package_name: the exact PyPI package name (e.g. "ultralytics", "boxmot").
        timeout: request timeout in seconds.

    Returns:
        PackageMetadata. If the fetch fails for any reason, returns a
        PackageMetadata with empty fields and a warning in `fetch_warnings`
        rather than raising — callers (agent nodes) should check
        `fetch_warnings` rather than wrapping every call in try/except.
    """
    url = f"{PYPI_BASE_URL}/{package_name}/json"

    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
    except httpx.RequestError as e:
        return PackageMetadata(
            name=package_name,
            fetch_warnings=[f"Network error fetching {url}: {e}"],
        )

    if response.status_code == 404:
        return PackageMetadata(
            name=package_name,
            fetch_warnings=[f"Package '{package_name}' not found on PyPI (404)."],
        )

    if response.status_code != 200:
        return PackageMetadata(
            name=package_name,
            fetch_warnings=[f"PyPI returned status {response.status_code} for '{package_name}'."],
        )

    data = response.json()
    info = data.get("info", {})
    releases = data.get("releases", {})

    warnings: list[str] = []

    # Only keep versions that have at least one non-yanked file uploaded.
    available_versions = []
    for version, files in releases.items():
        if not files:
            continue
        if all(f.get("yanked", False) for f in files):
            continue
        available_versions.append(version)

    # Loose sort
    available_versions.sort()

    if not available_versions:
        warnings.append(f"No installable (non-yanked) releases found for '{package_name}'.")

    return PackageMetadata(
        name=package_name,
        summary=info.get("summary"),
        latest_version=info.get("version"),
        available_versions=available_versions,
        declared_dependencies=list(info.get("requires_dist") or []),
        project_urls=dict(info.get("project_urls") or {}),
        source="pypi",
        fetch_warnings=warnings,
    )

def get_pypi_release_metadata(package_name: str, version: str, timeout: float = 10.0) -> dict:
    """
    Fetch metadata for a *specific* version, not just the latest.

    Useful when the LLM proposes an older pin (e.g. because the latest
    version doesn't support the detected CUDA version) and you need that
    version's actual declared dependencies, not the latest release's.

    Raises httpx.HTTPStatusError on failure — unlike get_pypi_metadata
    """
    url = f"{PYPI_BASE_URL}/{package_name}/{version}/json"
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.json()