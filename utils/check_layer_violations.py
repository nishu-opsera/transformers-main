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
"""Detect config-to-model layer boundary violations (WO-002).

Configuration-layer modules must not depend on modeling-layer code. This script
flags imports from modeling modules and uses of symbols imported from modeling
in configuration_*.py and configuration_utils.py (not modular_*.py codegen sources).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "transformers"
BASELINE_PATH = REPO_ROOT / ".ci" / "layer_violation_baseline.json"
BASELINE_TOLERANCE = 0.05  # 5% per WO-002 acceptance criteria


@dataclass(frozen=True)
class LayerViolation:
    file: str
    line: int
    kind: str
    symbol: str
    modeling_module: str

    def key(self) -> tuple[str, int, str, str, str]:
        return (self.file, self.line, self.kind, self.symbol, self.modeling_module)

    def dedupe_key(self) -> tuple[str, int, str]:
        """One violation per config file line and symbol (import + use are not double-counted)."""
        return (self.file, self.line, self.symbol)


def is_configuration_layer_file(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    try:
        rel = path.relative_to(SRC_ROOT)
    except ValueError:
        return False
    name = rel.name
    if name == "configuration_utils.py":
        return True
    if name.startswith("configuration_"):
        return True
    # modular_*.py are codegen/model sources, not runtime configuration boundaries (WO-013).
    return False


def is_modeling_module(module: str | None) -> bool:
    if not module:
        return False
    normalized = module.replace("...", ".")
    parts = [part for part in normalized.split(".") if part]
    return any(part.startswith("modeling") or part == "modeling_utils" for part in parts)


def resolve_import_module(file_path: Path, level: int, module: str | None) -> str:
    if module is None:
        return ""
    if level == 0:
        return module
    parts = list(file_path.relative_to(SRC_ROOT).parent.parts)
    for _ in range(level - 1):
        if parts:
            parts.pop()
    if module.startswith("."):
        parent_depth = len(module) - len(module.lstrip("."))
        for _ in range(parent_depth):
            if parts:
                parts.pop()
        suffix = module.lstrip(".").split(".")
        return ".".join(parts + suffix) if suffix and suffix != [""] else ".".join(parts)
    return module


def collect_violations(file_path: Path) -> list[LayerViolation]:
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    rel_file = str(file_path.relative_to(REPO_ROOT))
    modeling_symbols: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = resolve_import_module(file_path, node.level, node.module)
            if not is_modeling_module(resolved):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                symbol = alias.asname or alias.name
                modeling_symbols[symbol] = resolved
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not is_modeling_module(alias.name):
                    continue
                symbol = alias.asname or alias.name.split(".")[-1]
                modeling_symbols[symbol] = alias.name

    violations: list[LayerViolation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in modeling_symbols:
                violations.append(
                    LayerViolation(rel_file, node.lineno, "use", node.id, modeling_symbols[node.id])
                )
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if isinstance(node.value, ast.Name) and node.value.id in modeling_symbols:
                violations.append(
                    LayerViolation(
                        rel_file,
                        node.lineno,
                        "use",
                        f"{node.value.id}.{node.attr}",
                        modeling_symbols[node.value.id],
                    )
                )

    return dedupe_violations(violations)


def dedupe_violations(violations: list[LayerViolation]) -> list[LayerViolation]:
    seen: set[tuple[str, int, str]] = set()
    unique: list[LayerViolation] = []
    for violation in sorted(violations, key=lambda item: (item.dedupe_key(), item.kind)):
        key = violation.dedupe_key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(violation)
    return unique


def scan_repository() -> list[LayerViolation]:
    violations: list[LayerViolation] = []
    for file_path in sorted(SRC_ROOT.rglob("*.py")):
        if is_configuration_layer_file(file_path):
            violations.extend(collect_violations(file_path))
    return dedupe_violations(violations)


def build_report(violations: list[LayerViolation]) -> dict:
    per_file: dict[str, int] = {}
    for violation in violations:
        per_file[violation.file] = per_file.get(violation.file, 0) + 1
    return {
        "total_count": len(violations),
        "per_file": dict(sorted(per_file.items())),
        "violations": [asdict(v) for v in violations],
    }


def violation_fingerprint(violation: LayerViolation) -> str:
    return f"{violation.file}:{violation.line}:{violation.symbol}"


def write_baseline(report: dict, violations: list[LayerViolation]) -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_count": report["total_count"],
        "per_file": report["per_file"],
        "forgescore_reference_count": 3958,
        "fingerprints": sorted(violation_fingerprint(v) for v in violations),
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def check_against_baseline(report: dict, violations: list[LayerViolation]) -> int:
    if not BASELINE_PATH.exists():
        print(f"Missing baseline at {BASELINE_PATH}. Run: python utils/check_layer_violations.py --write-baseline", file=sys.stderr)
        return 1

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    allowed = baseline["total_count"]
    current = report["total_count"]
    lower = int(allowed * (1 - BASELINE_TOLERANCE))
    upper = int(allowed * (1 + BASELINE_TOLERANCE))

    print(f"Layer violations: {current} (baseline {allowed}, allowed range {lower}-{upper})")

    allowed_fps = set(baseline.get("fingerprints", []))
    if allowed_fps:
        current_fps = {violation_fingerprint(v) for v in violations}
        new_fps = sorted(current_fps - allowed_fps)
        if new_fps:
            print("\nNew layer violation fingerprints (not in baseline):", file=sys.stderr)
            for fp in new_fps[:25]:
                print(f"  {fp}", file=sys.stderr)
            if len(new_fps) > 25:
                print(f"  ... and {len(new_fps) - 25} more", file=sys.stderr)
            print(
                "\nConfig-layer code must not add dependencies on modeling modules.\n"
                "If intentional, update the baseline via an explicit PR:\n"
                "  python utils/check_layer_violations.py --write-baseline\n",
                file=sys.stderr,
            )
            return 1

    if current > allowed:
        print(
            f"\nViolation count increased: {current} > {allowed}.\n"
            "  python utils/check_layer_violations.py --write-baseline\n",
            file=sys.stderr,
        )
        return 1

    if current < lower:
        print(
            f"\nViolation count dropped below baseline tolerance ({current} < {lower}).\n"
            "Update the baseline to ratchet down:\n"
            "  python utils/check_layer_violations.py --write-baseline\n",
            file=sys.stderr,
        )
        return 1

    print("Layer violation baseline check passed (no new violations).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write .ci/layer_violation_baseline.json from the current scan.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path to write the full JSON report.",
    )
    args = parser.parse_args(argv)

    violations = scan_repository()
    report = build_report(violations)

    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote report to {args.report}")

    if args.write_baseline:
        write_baseline(report, violations)
        print(f"Wrote baseline to {BASELINE_PATH} ({report['total_count']} violations)")
        return 0

    return check_against_baseline(report, violations)


if __name__ == "__main__":
    sys.exit(main())
