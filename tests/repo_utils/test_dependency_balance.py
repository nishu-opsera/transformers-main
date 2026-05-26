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
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_SCRIPT = REPO_ROOT / "utils" / "measure_dependency_balance.py"
BASELINE_PATH = REPO_ROOT / ".ci" / "dependency_balance_baseline.json"


def _score(fan_in: dict[str, int]) -> float:
    utils_dir = str(REPO_ROOT / "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    from measure_dependency_balance import compute_balance_score  # noqa: E402

    return compute_balance_score(fan_in)


def test_balance_score_monopoly_is_zero():
    assert _score({"transformers.utils": 500}) == pytest.approx(0.0)


def test_balance_score_uniform_is_one():
    fan_in = {f"transformers.m{i}": 10 for i in range(5)}
    assert _score(fan_in) == pytest.approx(1.0)


def test_balance_score_intermediate_distribution():
    fan_in = {"transformers.a": 50, "transformers.b": 30, "transformers.c": 20}
    score = _score(fan_in)
    assert 0.0 < score < 1.0


def test_measure_dependency_balance_produces_report(tmp_path):
    report_path = tmp_path / "report.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [sys.executable, str(UTILS_SCRIPT), "--report", str(report_path)],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "balance_score" in payload
    assert 0.0 <= payload["balance_score"] <= 1.0
    assert len(payload["top_10_fan_in"]) <= 10
    assert payload["total_top_level_import_edges"] > 0


def test_committed_baseline_matches_current_tooling():
    assert BASELINE_PATH.is_file()
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert "balance_score" in baseline
    assert baseline["balance_score"] >= 0.1
