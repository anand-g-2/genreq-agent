"""GitHub client for fetching README/release notes to find compatibility hints.

Many CV/ML packages (ultralytics, boxmot, insightface) document real-world
version pins in their README or GitHub releases that never make it into
PyPI's `requires_dist` metadata. This module fetches that text; the
keyword-based `extract_compat_notes` is a simple pre-filter to shrink what
gets passed to the LLM, not a substitute for the LLM reading the matches.
"""

from __future__ import annotations

import base64
import os
import httpx

from reqgen.research.schema import CompatNote

GITHUB_API_BASE = "https://api.github.com"

def _auth_headers() -> dict[str, str]:
    """Attach a GITHUB_TOKEN if present, to avoid the 60 req/hr unauthenticated limit."""
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def get_readme_text(owner: str, repo: str, timeout: float = 10.0) -> tuple[str | None, list[str]]:
    """
    Fetch and decode a repo's README via the GitHub API.

    Returns (readme_text, warnings). readme_text is None if the fetch failed
    for any reason (repo not found, rate limited, network error).
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme"

    try:
        response = httpx.get(url, headers=_auth_headers(), timeout=timeout)
    except httpx.RequestError as e:
        return None, [f"Network error fetching README for {owner}/{repo}: {e}"]

    if response.status_code == 403:
        return None, [
            f"GitHub API rate-limited fetching {owner}/{repo} README. "
            "Set a GITHUB_TOKEN env var to raise the limit from 60 to 5000 req/hr."
        ]

    if response.status_code == 404:
        return None, [f"Repo or README not found: {owner}/{repo}."]

    if response.status_code != 200:
        return None, [f"GitHub returned {response.status_code} fetching {owner}/{repo} README."]

    data = response.json()
    content_b64 = data.get("content", "")
    try:
        readme_text = base64.b64decode(content_b64).decode("utf-8", errors="replace")
    except Exception as e:
        return None, [f"Failed to decode README content for {owner}/{repo}: {e}"]

    return readme_text, []


def get_latest_release_notes(owner: str, repo: str, timeout: float = 10.0) -> tuple[str | None, list[str]]:
    """Fetch the body text of the latest GitHub release, if the repo publishes any."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/releases/latest"

    try:
        response = httpx.get(url, headers=_auth_headers(), timeout=timeout)
    except httpx.RequestError as e:
        return None, [f"Network error fetching releases for {owner}/{repo}: {e}"]

    if response.status_code == 404:
        return None, [f"No releases found for {owner}/{repo} (repo may only use tags, not releases)."]

    if response.status_code != 200:
        return None, [f"GitHub returned {response.status_code} fetching {owner}/{repo} releases."]

    data = response.json()
    return data.get("body"), []


# Narrow, CV/ML-specific keywords — kept deliberately tight to avoid
# flooding the LLM context with irrelevant matched lines.
_COMPAT_KEYWORDS = (
    "cuda",
    "torch==",
    "torch>=",
    "torch<",
    "torchvision",
    "onnxruntime-gpu",
    "tensorrt",
    "python_requires",
    "requires python",
    "compatible with",
)


def extract_compat_notes(repo: str, text: str, source_url: str) -> list[CompatNote]:
    """
    Scan README/release notes for lines that look like they're
    stating a version compatibility constraint.
    """
    notes: list[CompatNote] = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(keyword in lowered for keyword in _COMPAT_KEYWORDS):
            stripped = line.strip()
            if stripped:
                notes.append(CompatNote(package=repo, source_url=source_url, note=stripped))
    return notes