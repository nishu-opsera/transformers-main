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

from catalog_circular_dependencies import (  # noqa: E402
    STRATEGY_DESCRIPTIONS,
    build_catalog,
    severity_label,
)


def test_severity_thresholds():
    assert severity_label(51) == "critical"
    assert severity_label(50) == "high"
    assert severity_label(10) == "high"
    assert severity_label(9) == "medium"


def test_catalog_matches_known_cycles_baseline():
    catalog = build_catalog()
    known = json.loads((REPO_ROOT / ".ci" / "known_import_cycles.json").read_text(encoding="utf-8"))
    assert catalog["cross_reference"]["catalogued_cycles"] == len(known["two_module_cycles"])
    assert len(catalog["cycles"]) == len(known["two_module_cycles"])


def test_resolution_strategies_documented():
    assert len(STRATEGY_DESCRIPTIONS) >= 3
