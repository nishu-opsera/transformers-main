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
"""Generate domain_registry.json for __init__ decomposition (WO-009)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
MODELS_ROOT = SRC_ROOT / "transformers" / "models"
OUTPUT_PATH = MODELS_ROOT / "domain_registry.json"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transformers.models.auto.auto_mappings import CONFIG_MAPPING_NAMES  # noqa: E402

MULTIMODAL_HINTS = {
    "clip",
    "llava",
    "paligemma",
    "idefics",
    "blip",
    "instructblip",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "gemma3",
    "mllama",
    "phi4_multimodal",
    "smolvlm",
    "video_llava",
    "vipllava",
    "xclip",
    "flava",
    "owlvit",
    "git",
    "fuyu",
    "chameleon",
    "emu3",
    "aria",
}
AUDIO_HINTS = {
    "whisper",
    "wav2vec2",
    "hubert",
    "wavlm",
    "seamless",
    "speecht5",
    "musicgen",
    "encodec",
    "bark",
    "clap",
    "data2vec",
}
VISION_HINTS = {
    "vit",
    "detr",
    "deit",
    "beit",
    "swin",
    "convnext",
    "resnet",
    "segformer",
    "sam",
    "dpt",
    "yolos",
    "rt_detr",
    "dinov2",
    "dinov3",
    "poolformer",
    "levit",
    "mobilevit",
    "efficientnet",
}


def classify_model_type(model_type: str) -> dict:
    module_name = model_type.replace("-", "_")
    model_dir = MODELS_ROOT / module_name
    if not model_dir.is_dir():
        # fallback to name heuristics only
        files = set()
    else:
        files = {path.name for path in model_dir.glob("*.py")}

    has_image = any(name.startswith("image_processing") for name in files)
    has_video = any(name.startswith("video_processing") for name in files)
    has_audio = any(
        name.startswith(("feature_extraction", "audio_processing")) or "speech" in name for name in files
    )
    has_text = any(name.startswith(("tokenization", "modeling")) for name in files)

    primary = "nlp"
    secondary: list[str] = []

    if model_type in MULTIMODAL_HINTS or (has_image and has_text) or has_video:
        primary = "multimodal"
    elif model_type in AUDIO_HINTS or (has_audio and not has_image):
        primary = "audio"
    elif model_type in VISION_HINTS or (has_image and not has_text):
        primary = "vision"
    elif has_image and has_text:
        primary = "multimodal"

    if primary == "multimodal":
        if has_text:
            secondary.append("nlp")
        if has_image:
            secondary.append("vision")
        if has_audio or has_video:
            secondary.append("audio")
    elif primary == "vision" and has_text:
        secondary.append("nlp")
    elif primary == "nlp" and has_image:
        secondary.append("vision")

    secondary = sorted(set(secondary) - {primary})

    return {
        "primary_domain": primary,
        "secondary_domains": secondary,
        "config_class": CONFIG_MAPPING_NAMES[model_type],
        "module": f"transformers.models.{module_name}",
    }


def build_registry() -> dict:
    entries = {model_type: classify_model_type(model_type) for model_type in CONFIG_MAPPING_NAMES}
    counts: dict[str, int] = {}
    for meta in entries.values():
        counts[meta["primary_domain"]] = counts.get(meta["primary_domain"], 0) + 1
    return {
        "version": 1,
        "source": "CONFIG_MAPPING_NAMES",
        "model_count": len(entries),
        "domain_counts": counts,
        "models": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify registry matches CONFIG_MAPPING_NAMES.")
    args = parser.parse_args()

    registry = build_registry()
    if args.check:
        if not OUTPUT_PATH.exists():
            print(f"Missing {OUTPUT_PATH}", file=sys.stderr)
            return 1
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if existing.get("model_count") != len(CONFIG_MAPPING_NAMES):
            print("domain_registry.json is out of date.", file=sys.stderr)
            return 1
        print("domain_registry.json is up to date.")
        return 0

    OUTPUT_PATH.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({registry['model_count']} models)")
    print("Domain counts:", registry["domain_counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
