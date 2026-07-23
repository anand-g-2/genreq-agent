#!/usr/bin/env python3
"""
Standalone CLI harness for Phase 3: validate a candidate requirements list.

Usage:
    python scripts/run_validator.py requests>=2.31.0
    python scripts/run_validator.py boto3==1.34.100 botocore==1.20.0
    python scripts/run_validator.py torch==2.1.0 --extra-index https://download.pytorch.org/whl/cu121
    python scripts/run_validator.py torch==2.1.0 --real-install   # slower, catches runtime failures too
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reqgen.validator import validate_requirements  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a candidate requirements list in a throwaway venv.")
    parser.add_argument("requirements", nargs="+", help="Requirement strings, e.g. 'torch==2.1.0'")
    parser.add_argument(
        "--extra-index", action="append", dest="extra_index_urls", default=None,
        help="Extra package index URL (repeatable), e.g. for CUDA-specific torch wheels.",
    )
    parser.add_argument("--python", dest="python_version", default=None, help="Python version for the venv, e.g. '3.10'")
    parser.add_argument(
        "--real-install", action="store_true",
        help="Do a real install instead of --dry-run (slower, but catches runtime import failures too).",
    )
    args = parser.parse_args()

    result = validate_requirements(
        requirements=args.requirements,
        extra_index_urls=args.extra_index_urls,
        python_version=args.python_version,
        dry_run=not args.real_install,
    )

    print(f"Requirements: {result.requirements}")
    print(f"Success:      {result.success}")
    print(f"Duration:     {result.duration_seconds}s")

    if result.success:
        print("STDOUT:")
        print(result.stdout)
    else:
        print(f"Found {len(result.conflicts)} conflict(s):\n")
        for c in result.conflicts:
            print(f"  reason: {c.reason}")
            print(f"  packages_involved: {c.packages_involved}")
            print(f"  detail: {c.detail[:300]}")


if __name__ == "__main__":
    main()