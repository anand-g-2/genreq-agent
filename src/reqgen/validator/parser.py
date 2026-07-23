"""Parses raw uv/pip install error output into structured conflict information.

Built and tuned against uv's actual observed error format —
notably, uv wraps multi-line conflict explanations WITHOUT sentence-ending
punctuation before the line break, e.g.:

    Because boto3==1.34.100 depends on botocore>=1.34.100,<1.35.0
    and you require boto3==1.34.100, we can conclude that you require
    botocore>=1.34.100,<1.35.0.
    And because you require botocore==1.20.0, we can conclude that your
    requirements are unsatisfiable.

A naive line-by-line regex parser misses this entirely. This module
normalizes whitespace across the whole block before pattern-matching.

This is deliberately conservative: real resolver conflicts are prose 
(error messages are written in human language), and regex parsing of prose 
is inherently fragile. The goal is to extract *enough* structure 
(which packages are implicated, what kind of failure) to give the
LLM repair prompt better signal than raw text alone — not to build a
complete formal parser. The full normalized text is always preserved in
`detail`, so nothing is lost even when categorization is imperfect.
"""

from __future__ import annotations

import re
from reqgen.validator.schema import ConflictInfo

# regex patterns
# ([A-Za-z][A-Za-z0-9_.\-]*) - start with a letter, then letters, digits, underscore, dot, or hyphen
# Example: boto3, botocore, torch, boxmot
# \s* - optional whitespace; example: boto3==1.0 or boto3 == 1.0
# (?:==|>=|<=|!=|~=|<|>) - version operator (==, >=, <=, !=, ~=, <, >)
# [A-Za-z0-9_.\*]+ - version string; example: 1.34.10, 1.0, v1.0, 1.0.*
_PKG_SPEC_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_.\-]*)\s*(?:==|>=|<=|!=|~=|<|>)\s*[A-Za-z0-9_.\*]+")
_NO_MATCH_RE = re.compile(r"No matching distribution found for ([A-Za-z0-9_.\-]+)", re.IGNORECASE)


def parse_install_errors(stderr: str) -> list[ConflictInfo]:
    """Extract structured conflict info from raw uv/pip stderr."""
    conflicts: list[ConflictInfo] = []

    if not stderr.strip():
        return conflicts

    # Collapse all whitespace (including newlines) so wrapped sentences can
    # be matched as one continuous string.
    normalized = re.sub(r"\s+", " ", stderr).strip()

    if "no solution found" in normalized.lower():
        packages = sorted({m.group(1) for m in _PKG_SPEC_RE.finditer(normalized)})
        conflicts.append(
            ConflictInfo(reason="no_solution", packages_involved=packages, detail=normalized)
        )

    for match in _NO_MATCH_RE.finditer(normalized):
        conflicts.append(
            ConflictInfo(
                reason="no_matching_distribution",
                packages_involved=[match.group(1)],
                detail=match.group(0),
            )
        )

    # Fallback: install failed but nothing matched a known pattern. Still provide 
    # at least the raw error text plus whatever package names could be pulled out of it.
    if not conflicts:
        packages = sorted({m.group(1) for m in _PKG_SPEC_RE.finditer(normalized)})
        conflicts.append(
            ConflictInfo(
                reason="unrecognized_error",
                packages_involved=packages,
                detail=normalized[:1000],
            )
        )

    return conflicts