"""Tests for Phase 3 validator.

Unlike Phase 1/2 tests, these deliberately hit the REAL `uv` CLI and real
PyPI — mocking subprocess calls here would just test that our code calls
subprocess correctly, not that the actual conflict detection works, which is
the entire point of Phase 3.

Test package choices are ones with an actual, real, tightly-pinned
transitive dependency (boto3 -> botocore) rather than a synthetic example,
verified manually via scripts/run_validator.py during development.
"""

from __future__ import annotations

from reqgen.validator import validate_requirements
from reqgen.validator.parser import parse_install_errors


class TestValidateRequirements:
    def test_satisfiable_requirements_succeed(self):
        result = validate_requirements(["requests>=2.31.0"], dry_run=True)

        assert result.success is True
        assert result.return_code == 0
        assert result.conflicts == []

    def test_direct_double_pin_conflict_is_caught(self):
        result = validate_requirements(["packaging==21.3", "packaging==23.0"], dry_run=True)

        assert result.success is False
        assert len(result.conflicts) >= 1
        assert result.conflicts[0].reason == "no_solution"
        assert "packaging" in result.conflicts[0].packages_involved

    def test_transitive_conflict_is_caught(self):
        # boto3==1.34.100 requires botocore>=1.34.100,<1.35.0 — pinning an
        # incompatible botocore version transitively conflicts.
        result = validate_requirements(
            ["boto3==1.34.100", "botocore==1.20.0"], dry_run=True
        )

        assert result.success is False
        conflict = result.conflicts[0]
        assert conflict.reason == "no_solution"
        assert "boto3" in conflict.packages_involved
        assert "botocore" in conflict.packages_involved


class TestParseInstallErrors:
    def test_empty_stderr_returns_no_conflicts(self):
        assert parse_install_errors("") == []
        assert parse_install_errors("   ") == []

    def test_handles_wrapped_multiline_conflict_text(self):
        # Real uv output wraps sentences across lines WITHOUT sentence-ending
        # punctuation before the break — a naive line-by-line regex parser would miss this entirely.
        stderr = (
            "  \u00d7 No solution found when resolving dependencies:\n"
            "  \u2570\u2500\u25b6 Because boto3==1.34.100 depends on botocore>=1.34.100,<1.35.0\n"
            "      and you require boto3==1.34.100, we can conclude that you require\n"
            "      botocore>=1.34.100,<1.35.0.\n"
            "      And because you require botocore==1.20.0, we can conclude that your\n"
            "      requirements are unsatisfiable."
        )
        conflicts = parse_install_errors(stderr)

        assert len(conflicts) == 1
        assert conflicts[0].reason == "no_solution"
        assert set(conflicts[0].packages_involved) >= {"boto3", "botocore"}

    def test_no_matching_distribution_extracted(self):
        stderr = "ERROR: No matching distribution found for nonexistent-package-xyz==99.0.0"
        conflicts = parse_install_errors(stderr)

        assert len(conflicts) == 1
        assert conflicts[0].reason == "no_matching_distribution"
        assert conflicts[0].packages_involved == ["nonexistent-package-xyz"]

    def test_unrecognized_error_still_produces_fallback_conflict(self):
        # Even if the error doesn't match a known pattern, callers should
        # still get *something* structured back — never silently empty.
        stderr = "Some completely novel error message mentioning somepackage==1.0.0"
        conflicts = parse_install_errors(stderr)

        assert len(conflicts) == 1
        assert conflicts[0].reason == "unrecognized_error"
        assert "somepackage" in conflicts[0].packages_involved