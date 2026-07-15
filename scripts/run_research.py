"""
Standalone CLI harness for Phase 2: research a package via PyPI + GitHub.

Usage:
    python scripts/run_research.py ultralytics --github ultralytics/ultralytics
    python scripts/run_research.py boxmot --github mikel-brostrom/boxmot
    python scripts/run_research.py numpy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reqgen.research import research_package  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser(description="Research a package's PyPI metadata + GitHub compat notes.")
    parser.add_argument("package", help="PyPI package name, e.g. 'ultralytics'")
    parser.add_argument(
        "--github", metavar="OWNER/REPO", default=None, help="GitHub repo, e.g. 'ultralytics/ultralytics'"
    )
    args = parser.parse_args()

    owner, repo = (args.github.split("/", 1) if args.github else (None, None))

    metadata, compat_notes, warnings = research_package(args.package, owner, repo)

    print(f"=== {metadata.name} (PyPI) ===")
    print(f"Latest version:  {metadata.latest_version}")
    print(f"Summary:         {metadata.summary}")
    print(f"Versions found:  {len(metadata.available_versions)} "
          f"(newest few: {metadata.available_versions[-5:]})")
    print(f"\nDeclared dependencies ({len(metadata.declared_dependencies)}):")
    for dep in metadata.declared_dependencies[:15]:
        print(f"  - {dep}")

    if compat_notes:
        print(f"\nCompat-relevant lines found in README/releases ({len(compat_notes)}):")
        for note in compat_notes[:15]:
            print(f"  [{note.source_url}]")
            print(f"    {note.note}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()