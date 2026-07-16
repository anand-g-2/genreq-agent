"""Web search tool — fallback for compatibility info not found via PyPI/GitHub.

Uses the Tavily Search API (https://tavily.com), which is commonly paired with
LangChain/LangGraph agents and has a free tier. Requires a TAVILY_API_KEY env var.

This tool exists for gaps where known torch+CUDA+onnxruntime-gpu version gotchas 
that live in GitHub issues, forums, or blog posts rather than in any package's 
official metadata.
"""

from __future__ import annotations

import os
import httpx

from reqgen.research.schema import SearchResult

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

def web_search_compat(
    query: str, max_results: int = 5, timeout: float = 15.0
) -> tuple[list[SearchResult], list[str]]:
    """
    Run a web search for compatibility info not covered by PyPI/GitHub.

    Returns (results, warnings). Returns empty results + a warning if
    TAVILY_API_KEY isn't set, rather than raising — this tool is meant to be
    optional.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return [], [
            "TAVILY_API_KEY not set — web search fallback is disabled. "
            "Get a free key at https://tavily.com and set it as an env var to enable."
        ]

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }

    try:
        response = httpx.post(TAVILY_SEARCH_URL, json=payload, timeout=timeout)
    except httpx.RequestError as e:
        return [], [f"Network error during web search for '{query}': {e}"]

    if response.status_code != 200:
        return [], [f"Tavily search returned status {response.status_code} for '{query}'."]

    data = response.json()
    results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
        )
        for item in data.get("results", [])
    ]

    return results, []