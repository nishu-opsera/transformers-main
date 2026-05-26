# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Minimal stub for static protocol conformance checks (WO-012).
# Analyzed by utils/check_protocol_typing.py with ``ty`` when available.

from __future__ import annotations

from dataclasses import dataclass

from transformers.protocols import ConfigProtocol, ModelConfigConsumer


@dataclass
class _ValidConfig:
    model_type: str = "bert"

    def to_dict(self) -> dict:
        return {"model_type": self.model_type}

    def to_json_string(self, use_diff: bool = True) -> str:
        return '{"model_type": "bert"}'


def accept_config(config: ConfigProtocol) -> str:
    return config.model_type


def accept_consumer(config: ModelConfigConsumer) -> dict:
    return config.to_dict()


def _demo() -> None:
    cfg = _ValidConfig()
    accept_config(cfg)
    accept_consumer(cfg)
