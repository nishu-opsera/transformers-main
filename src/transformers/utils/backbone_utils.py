import importlib
import warnings

_DEPRECATED_BACKBONE_PATH = (
    "Importing `{name}` from `utils/backbone_utils.py` is deprecated and will be removed in "
    "Transformers v5.10. Import as `from transformers.backbone_utils import {name}` instead."
)


def __getattr__(name: str):
    if name not in ("BackboneConfigMixin", "BackboneMixin"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(_DEPRECATED_BACKBONE_PATH.format(name=name), FutureWarning, stacklevel=2)
    return getattr(importlib.import_module("transformers.backbone_utils"), name)
