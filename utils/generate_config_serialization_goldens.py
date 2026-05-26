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
"""Regenerate WO-027 config serialization golden fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "config_serialization"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transformers.models.auto.configuration_auto import CONFIG_MAPPING  # noqa: E402

REPRESENTATIVE_MODEL_TYPES = [
    "bert",
    "gpt2",
    "roberta",
    "distilbert",
    "albert",
    "t5",
    "bart",
    "xlnet",
    "electra",
    "deberta",
    "bloom",
    "llama",
    "mistral",
    "qwen2",
    "gemma",
    "vit",
    "clip",
    "whisper",
    "wav2vec2",
    "gpt_neox",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIXTURES_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_type in REPRESENTATIVE_MODEL_TYPES:
        config = CONFIG_MAPPING[model_type]()
        to_dict_path = args.output_dir / f"{model_type}_to_dict.json"
        to_json_path = args.output_dir / f"{model_type}_to_json_string.json"
        to_dict_path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        to_json_path.write_text(config.to_json_string(use_diff=False), encoding="utf-8")
        print(f"wrote {model_type}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
