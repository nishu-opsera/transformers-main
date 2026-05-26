#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Measure transformers import wall-clock time for CI tracking (WO-004)."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / ".ci" / "import_time_baseline.json"
REPORT_PATH = REPO_ROOT / ".ci" / "import_time_report.json"
REGRESSION_THRESHOLD = 0.10  # 10% warning threshold per WO-004
DEFAULT_ITERATIONS = 5

IMPORT_SCENARIOS: dict[str, str] = {
    "import_transformers": "import transformers",
    "import_auto_model": "from transformers import AutoModel",
}


def compute_stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def measure_import(snippet: str, iterations: int, pythonpath: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        samples.append(time.perf_counter() - start)
    return {"samples_seconds": samples, **compute_stats(samples)}


def run_benchmark(iterations: int, pythonpath: str) -> dict:
    results: dict[str, dict] = {}
    for name, snippet in IMPORT_SCENARIOS.items():
        results[name] = measure_import(snippet, iterations, pythonpath)
    return {
        "iterations": iterations,
        "pythonpath": pythonpath,
        "scenarios": results,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def check_regression(report: dict, baseline: dict) -> list[str]:
    warnings: list[str] = []
    for scenario, stats in report["scenarios"].items():
        baseline_stats = baseline.get("scenarios", {}).get(scenario)
        if not baseline_stats:
            continue
        baseline_mean = baseline_stats["mean"]
        current_mean = stats["mean"]
        if baseline_mean <= 0:
            continue
        increase = (current_mean - baseline_mean) / baseline_mean
        if increase > REGRESSION_THRESHOLD:
            warnings.append(
                f"{scenario}: mean import time {current_mean:.3f}s is "
                f"{increase * 100:.1f}% above baseline {baseline_mean:.3f}s "
                f"(threshold {REGRESSION_THRESHOLD * 100:.0f}%)"
            )
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--pythonpath", default="src", help="PYTHONPATH for subprocess imports.")
    parser.add_argument("--write-baseline", action="store_true", help="Write .ci/import_time_baseline.json")
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="Path for JSON report output (default: .ci/import_time_report.json)",
    )
    args = parser.parse_args(argv)

    report = run_benchmark(args.iterations, args.pythonpath)
    write_json(args.report, report)
    print(f"Wrote import time report to {args.report}")

    if args.write_baseline:
        write_json(BASELINE_PATH, report)
        print(f"Wrote baseline to {BASELINE_PATH}")
        return 0

    if not BASELINE_PATH.exists():
        print(
            f"Missing baseline at {BASELINE_PATH}. Run with --write-baseline after review.",
            file=sys.stderr,
        )
        return 1

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    warnings = check_regression(report, baseline)
    for message in warnings:
        print(f"WARNING: {message}", file=sys.stderr)

    if warnings:
        print(
            f"\nImport time regression warning ({len(warnings)} scenario(s) over "
            f"{REGRESSION_THRESHOLD * 100:.0f}% threshold). Not blocking CI.",
            file=sys.stderr,
        )
    else:
        print("Import time baseline check passed (no regressions above warning threshold).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
