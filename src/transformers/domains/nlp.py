# Copyright 2026 The HuggingFace Team. All rights reserved.
"""NLP domain lazy registry (alias for ``transformers._registries.nlp``)."""

import sys

from .._registries import nlp as _nlp_registry

sys.modules[__name__] = sys.modules[_nlp_registry.__name__]
