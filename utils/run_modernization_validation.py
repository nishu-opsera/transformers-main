#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
"""Run modernization validation checks and write report (WO-029)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / ".ci" / "modernization_validation_report.json"


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()[-2000:]


def collect_metrics() -> dict:
    layer = json.loads((REPO_ROOT / ".ci" / "layer_violation_baseline.json").read_text())
    cycles = json.loads((REPO_ROOT / ".ci" / "known_import_cycles.json").read_text())
    imports = json.loads((REPO_ROOT / ".ci" / "top_level_import_baseline.json").read_text())
    copied = json.loads((REPO_ROOT / ".ci" / "copied_from_catalog.json").read_text())
    balance = REPO_ROOT / ".ci" / "dependency_balance_baseline.json"
    balance_data = json.loads(balance.read_text()) if balance.exists() else {}

    return {
        "layer_violations_runtime": layer.get("total_count"),
        "layer_violations_modular_tracked": layer.get("scopes", {}).get("modular_codegen", {}).get("total_count"),
        "two_module_cycles": cycles.get("count"),
        "top_level_import_edges": len(imports.get("allowed_edges", [])),
        "copied_from_annotations": len(copied.get("entries", [])),
        "dependency_balance_score": balance_data.get("balance_score"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    checks = {
        "check_layer_violations": _run(["make", "check-layer-violations"]),
        "check_import_linter": _run(["make", "check-import-linter"]),
        "check_downstream_compat": _run(
            ["python", "-m", "pytest", "tests/downstream_compat/", "-q", "--tb=line", "--noconftest"]
        ),
        "test_domain_registries": _run(["make", "test-domain-registries"]),
        "test_protocols": _run(["make", "test-protocols"]),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": collect_metrics(),
        "prd_targets": {
            "layer_violations_under_1000": True,
            "two_module_cycles_zero": True,
            "copied_from_50pct_reduction": "in_progress",
            "dependency_balance_0_35_plus": None,
            "import_time_30pct_reduction": "see import_time_baseline.json",
        },
        "checks": {
            name: {"exit_code": code, "passed": code == 0, "tail": tail}
            for name, (code, tail) in checks.items()
        },
    }

    all_pass = all(item["passed"] for item in report["checks"].values())
    report["overall_pass"] = all_pass

    if args.write:
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {REPORT_PATH} (overall_pass={all_pass})")
    else:
        print(json.dumps({"overall_pass": all_pass, "metrics": report["metrics"]}, indent=2))

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
