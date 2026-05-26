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

from benchmark_import_time import (  # noqa: E402
    check_regression,
    compute_stats,
)


def test_compute_stats_known_values():
    stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["mean"] == 3.0
    assert stats["median"] == 3.0
    assert stats["p95"] == 5.0
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0


def test_report_json_shape(tmp_path):
    report = {
        "iterations": 5,
        "pythonpath": "src",
        "scenarios": {
            "import_transformers": {
                "samples_seconds": [1.0, 1.1, 1.2, 1.0, 1.1],
                "mean": 1.08,
                "median": 1.1,
                "p95": 1.2,
                "min": 1.0,
                "max": 1.2,
            }
        },
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["iterations"] == 5
    assert "mean" in loaded["scenarios"]["import_transformers"]


def test_check_regression_emits_warning_above_threshold():
    baseline = {"scenarios": {"import_transformers": {"mean": 1.0}}}
    report = {"scenarios": {"import_transformers": {"mean": 1.2}}}
    warnings = check_regression(report, baseline)
    assert len(warnings) == 1
    assert "import_transformers" in warnings[0]


def test_check_regression_passes_within_threshold():
    baseline = {"scenarios": {"import_transformers": {"mean": 1.0}}}
    report = {"scenarios": {"import_transformers": {"mean": 1.05}}}
    assert check_regression(report, baseline) == []
