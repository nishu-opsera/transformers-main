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
"""Measure top-level import fan-in balance for transformers (WO-025)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import grimp

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_DIR = REPO_ROOT / ".ci"
BASELINE_PATH = CI_DIR / "dependency_balance_baseline.json"
REPORT_PATH = CI_DIR / "dependency_balance_report.json"
SRC_ROOT = REPO_ROOT / "src"
TARGET_SCORE = 0.35
REGRESSION_TOLERANCE = 0.02  # warn if score drops more than 2 points vs baseline


def top_level_module(module: str) -> str | None:
    if not module.startswith("transformers."):
        return None
    parts = module.split(".")
    if len(parts) < 2:
        return None
    return f"transformers.{parts[1]}"


def collect_top_level_fan_in() -> Counter[str]:
    graph = grimp.build_graph("transformers")
    fan_in: Counter[str] = Counter()
    for module in graph.modules:
        source = top_level_module(module)
        if source is None:
            continue
        for imported in graph.find_modules_directly_imported_by(module):
            destination = top_level_module(imported)
            if destination is None or destination == source:
                continue
            fan_in[destination] += 1
    return fan_in


def gini_coefficient(counts: list[int]) -> float:
    """Gini coefficient of a non-negative distribution (0 = perfectly equal)."""
    if not counts:
        return 0.0
    ordered = sorted(counts)
    total = sum(ordered)
    if total == 0:
        return 0.0
    cumulative = 0
    gini_sum = 0
    module_count = len(ordered)
    for value in ordered:
        cumulative += value
        gini_sum += cumulative
    return (module_count + 1 - 2 * gini_sum / total) / module_count


def compute_balance_score(fan_in: dict[str, int]) -> float:
    """Dependency balance score in [0, 1] as ``1 - Gini(fan_in)``.

    Matches the ForgeScore/PRD baseline (~0.18 today, target 0.35+):
    concentrated fan-in (e.g. ``transformers.utils``) yields a high Gini and a low score.
    """
    positive_counts = [count for count in fan_in.values() if count > 0]
    if len(positive_counts) <= 1:
        return 0.0 if len(positive_counts) == 1 else 1.0
    return 1.0 - gini_coefficient(positive_counts)


def build_report(fan_in: Counter[str]) -> dict:
    fan_in_dict = dict(sorted(fan_in.items(), key=lambda item: (-item[1], item[0])))
    top_10 = [
        {"module": module, "fan_in": count}
        for module, count in sorted(fan_in.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    return {
        "balance_score": round(compute_balance_score(fan_in_dict), 4),
        "target_score": TARGET_SCORE,
        "module_count_with_fan_in": sum(1 for count in fan_in.values() if count > 0),
        "total_top_level_import_edges": sum(fan_in.values()),
        "fan_in_by_module": fan_in_dict,
        "top_10_fan_in": top_10,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def compare_to_baseline(report: dict, baseline: dict) -> list[str]:
    messages: list[str] = []
    current = report["balance_score"]
    baseline_score = baseline["balance_score"]
    delta = current - baseline_score
    messages.append(
        f"balance_score={current:.4f} (baseline={baseline_score:.4f}, delta={delta:+.4f}, target={TARGET_SCORE})"
    )
    if delta < -REGRESSION_TOLERANCE:
        messages.append(
            f"Dependency balance regressed by {abs(delta):.4f} "
            f"(tolerance {REGRESSION_TOLERANCE}). Review recent import-structure changes."
        )
    elif current >= TARGET_SCORE:
        messages.append("Target balance score reached.")
    else:
        gap = TARGET_SCORE - current
        messages.append(f"Below PRD target by {gap:.4f}; continue __init__ decomposition / cycle work.")
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="JSON report path (default: .ci/dependency_balance_report.json)",
    )
    parser.add_argument("--write-baseline", action="store_true", help="Write .ci/dependency_balance_baseline.json")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare report to committed baseline and print improvement/regression summary",
    )
    args = parser.parse_args(argv)

    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    report = build_report(collect_top_level_fan_in())
    write_json(args.report, report)
    print(f"Wrote dependency balance report to {args.report}")
    print(f"balance_score={report['balance_score']}")

    if args.write_baseline:
        write_json(BASELINE_PATH, report)
        print(f"Wrote baseline to {BASELINE_PATH}")
        return 0

    if args.check:
        if not BASELINE_PATH.exists():
            print(f"Missing baseline at {BASELINE_PATH}. Run with --write-baseline first.", file=sys.stderr)
            return 1
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        messages = compare_to_baseline(report, baseline)
        for message in messages:
            print(message)
        if any("regressed" in message for message in messages):
            print("WARNING: dependency balance regression detected (non-blocking).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
