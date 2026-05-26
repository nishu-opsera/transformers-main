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

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS = REPO_ROOT / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from check_layer_violations import (  # noqa: E402
    collect_violations,
    is_configuration_layer_file,
    is_modeling_module,
)

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "layer_violations"


def test_is_modeling_module():
    assert is_modeling_module("transformers.modeling_utils")
    assert is_modeling_module("...modeling_rope_utils")
    assert not is_modeling_module("transformers.configuration_utils")


def test_is_configuration_layer_file():
    assert is_configuration_layer_file(REPO_ROOT / "src/transformers/models/llama/configuration_llama.py")
    assert is_configuration_layer_file(REPO_ROOT / "src/transformers/models/llama/modular_llama.py")
    assert not is_configuration_layer_file(REPO_ROOT / "src/transformers/models/llama/modeling_llama.py")


def test_clean_fixture_has_no_violations():
    violations = collect_violations(FIXTURES / "clean_config.py")
    assert violations == []


def test_violating_fixture_detects_modeling_symbol_use():
    violations = collect_violations(FIXTURES / "violating_config.py")
    assert violations
    assert all(v.kind == "use" for v in violations)
    assert any("PreTrainedModel" in v.symbol for v in violations)


def test_baseline_matches_current_scan():
    baseline_path = REPO_ROOT / ".ci" / "layer_violation_baseline.json"
    assert baseline_path.exists(), "Run: python utils/check_layer_violations.py --write-baseline"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    from check_layer_violations import scan_repository, violation_fingerprint  # noqa: E402

    violations = scan_repository()
    assert len(violations) == baseline["total_count"]
    assert {violation_fingerprint(v) for v in violations} == set(baseline["fingerprints"])


def test_baseline_within_forgescore_tolerance():
    baseline_path = REPO_ROOT / ".ci" / "layer_violation_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    count = baseline["total_count"]
    target = baseline.get("forgescore_reference_count", 3958)
    tolerance = 0.05
    assert target * (1 - tolerance) <= count <= target * (1 + tolerance), (
        f"Baseline {count} outside 5% of ForgeScore target {target}"
    )
