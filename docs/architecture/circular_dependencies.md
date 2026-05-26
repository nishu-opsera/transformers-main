# Circular dependency catalog

> **WO-006** — authoritative map of top-level import cycles. Regenerate with `PYTHONPATH=src python utils/catalog_circular_dependencies.py`.

_Generated: 2026-05-26T20:41:11.973979+00:00_

## How this document is produced

1. `utils/generate_importlinter_config.py` scans the live package with **grimp** and writes `.ci/known_import_cycles.json` (49 two-module cycles, WO-001 baseline).
2. `utils/catalog_circular_dependencies.py` (this catalog) enriches each cycle with severity, example submodule chains, and resolution strategies.
3. CI blocks **new** two-module cycles via `utils/check_import_linter.py` (baseline ratchet).

## Strongly connected components (top-level)

At the top-level `transformers.*` boundary, the import graph contains one large SCC (59 modules) plus isolated two-node cycles documented below. Breaking the hub modules inside the large SCC (`modeling_utils`, `configuration_utils`, `models`, `integrations`, `utils`) is the critical path for WO-007/WO-008.

| SCC size | Modules (first 12) |
|---------:|-------------------|
| 59 | `_typing`, `activations`, `audio_utils`, `backbone_utils`, `cache_utils`, `configuration_utils`, `conversion_mapping`, `convert_slow_tokenizer`, `core_model_loading`, `data`, `debug_utils`, `dependency_versions_check`, … (+47 more) |

## Severity summary (two-module cycles)

| Severity | Count | Definition |
|----------|------:|------------|
| critical | 12 | >50 top-level importers affected |
| high | 33 | 10–50 importers affected |
| medium | 3 | <10 importers affected |

## Resolution strategies

Three approaches used in subsequent work orders (WO-007, WO-008):

### `shared_abstraction`

Extract shared types/protocols into a neutral module (e.g. `modeling_protocols.py`) that both sides import instead of importing each other.

### `interface_or_lazy_import`

Introduce a `typing.Protocol` or move the import under `if TYPE_CHECKING:` so runtime import order no longer forms a cycle.

### `lazy_import`

Defer the import to function scope or use lazy module attributes so the cycle exists only for type checkers, not at runtime.

**Example — shared abstraction (WO-007 target):** `configuration_utils` ↔ `modeling_utils` — extract `PreTrainedConfig` protocol surface and weight-loading hooks into `transformers.modeling_protocols` so configs never import `PreTrainedModel` at module import time.

**Example — `typing.Protocol`:** `integrations` ↔ `trainer` — declare callback protocols in `integrations` without importing `Trainer` until runtime inside factory functions.

**Example — `TYPE_CHECKING` lazy import:** `utils` ↔ `image_utils` — move annotation-only imports under `if TYPE_CHECKING:` and quote forward references in public APIs.

## Two-module cycle catalog

Cross-referenced with `.ci/known_import_cycles.json`. **48** cycles documented.

| Severity | Modules | Importers | Strategy | Example chain (forward) |
|----------|---------|----------:|----------|-------------------------|
| critical | `modeling_utils` ↔ `utils` | 65 | `shared_abstraction` | modeling_utils → utils |
| critical | `activations` ↔ `utils` | 63 | `interface_or_lazy_import` | activations → hub_kernels → utils |
| critical | `backbone_utils` ↔ `utils` | 63 | `interface_or_lazy_import` | backbone_utils → transformers → utils |
| critical | `convert_slow_tokenizer` ↔ `utils` | 63 | `interface_or_lazy_import` | convert_slow_tokenizer → utils |
| critical | `image_utils` ↔ `utils` | 63 | `interface_or_lazy_import` | image_utils → utils |
| critical | `masking_utils` ↔ `utils` | 63 | `interface_or_lazy_import` | masking_utils → utils |
| critical | `model_debugging_utils` ↔ `utils` | 63 | `interface_or_lazy_import` | model_debugging_utils → import_utils → convert_slow_tokenizer → utils |
| critical | `models` ↔ `utils` | 63 | `interface_or_lazy_import` | models → utils |
| critical | `processing_utils` ↔ `utils` | 63 | `interface_or_lazy_import` | processing_utils → utils |
| critical | `tokenization_utils_base` ↔ `utils` | 63 | `interface_or_lazy_import` | tokenization_utils_base → utils |
| critical | `trainer_callback` ↔ `utils` | 63 | `interface_or_lazy_import` | trainer_callback → integrations → utils |
| critical | `trainer_utils` ↔ `utils` | 63 | `interface_or_lazy_import` | trainer_utils → utils |
| high | `integrations` ↔ `models` | 32 | `lazy_import` | integrations → utils → transformers → models |
| high | `modeling_utils` ↔ `models` | 27 | `shared_abstraction` | modeling_utils → utils → transformers → models |
| high | `models` ↔ `tokenization_utils_base` | 27 | `lazy_import` | models → kosmos2_5 → processing_kosmos2_5 → tokenization_utils_base |
| high | `integrations` ↔ `modeling_utils` | 26 | `shared_abstraction` | integrations → deepspeed → modeling_utils |
| high | `configuration_utils` ↔ `models` | 24 | `shared_abstraction` | configuration_utils → transformers → models |
| high | `integrations` ↔ `trainer_utils` | 23 | `lazy_import` | integrations → neftune → trainer_utils |
| high | `models` ↔ `processing_utils` | 23 | `lazy_import` | models → chmv2 → image_processing_chmv2 → processing_utils |
| high | `convert_slow_tokenizer` ↔ `models` | 22 | `lazy_import` | convert_slow_tokenizer → utils → transformers → models |
| high | `generation` ↔ `integrations` | 22 | `lazy_import` | generation → utils → transformers → integrations |
| high | `integrations` ↔ `training_args` | 22 | `lazy_import` | integrations → integration_utils → training_args |
| high | `core_model_loading` ↔ `integrations` | 21 | `lazy_import` | core_model_loading → modeling_utils → integrations |
| high | `feature_extraction_utils` ↔ `models` | 21 | `lazy_import` | feature_extraction_utils → utils → transformers → models |
| high | `image_processing_base` ↔ `models` | 21 | `lazy_import` | image_processing_base → utils → transformers → models |
| high | `integrations` ↔ `tokenization_utils_tokenizers` | 21 | `lazy_import` | integrations → utils → transformers → tokenization_utils_tokenizers |
| high | `integrations` ↔ `trainer_callback` | 21 | `lazy_import` | integrations → integration_utils → trainer_callback |
| high | `integrations` ↔ `trainer_pt_utils` | 21 | `lazy_import` | integrations → utils → transformers → trainer_pt_utils |
| high | `models` ↔ `quantizers` | 21 | `lazy_import` | models → bitnet → modeling_bitnet → modeling_utils → quantizers |
| high | `activations` ↔ `integrations` | 20 | `lazy_import` | activations → hub_kernels → transformers → integrations |
| high | `integrations` ↔ `quantizers` | 20 | `lazy_import` | integrations → sinq → core_model_loading → quantizers |
| high | `backbone_utils` ↔ `models` | 19 | `lazy_import` | backbone_utils → transformers → models |
| high | `generation` ↔ `modeling_utils` | 19 | `shared_abstraction` | generation → watermarking → modeling_utils |
| high | `integrations` ↔ `modeling_flash_attention_utils` | 19 | `lazy_import` | integrations → liger → modeling_utils → modeling_flash_attention_utils |
| high | `integrations` ↔ `trainer` | 19 | `lazy_import` | integrations → integration_utils → trainer |
| high | `modeling_layers` ↔ `models` | 19 | `lazy_import` | modeling_layers → utils → transformers → models |
| high | `models` ↔ `video_processing_utils` | 19 | `lazy_import` | models → internvl → video_processing_internvl → video_processing_utils |
| high | `conversion_mapping` ↔ `modeling_utils` | 16 | `shared_abstraction` | conversion_mapping → modeling_utils |
| high | `core_model_loading` ↔ `modeling_utils` | 16 | `shared_abstraction` | core_model_loading → modeling_utils |
| high | `fusion_mapping` ↔ `modeling_utils` | 16 | `shared_abstraction` | fusion_mapping → modeling_utils |
| high | `initialization` ↔ `modeling_utils` | 16 | `shared_abstraction` | initialization → modeling_utils |
| high | `modeling_utils` ↔ `quantizers` | 16 | `shared_abstraction` | modeling_utils → quantizers |
| high | `configuration_utils` ↔ `generation` | 15 | `shared_abstraction` | configuration_utils → transformers → generation |
| high | `configuration_utils` ↔ `modeling_rope_utils` | 11 | `shared_abstraction` | configuration_utils → modeling_rope_utils |
| high | `trainer_pt_utils` ↔ `training_args` | 10 | `lazy_import` | trainer_pt_utils → training_args |
| medium | `feature_extraction_sequence_utils` ↔ `feature_extraction_utils` | 8 | `lazy_import` | feature_extraction_sequence_utils → feature_extraction_utils |
| medium | `conversion_mapping` ↔ `core_model_loading` | 7 | `lazy_import` | conversion_mapping → core_model_loading |
| medium | `core_model_loading` ↔ `quantizers` | 7 | `lazy_import` | core_model_loading → quantizers |

## Work order mapping

| Work order | Scope |
|------------|-------|
| WO-001 | Baseline ratchet — block new top-level edges / 2-cycles |
| WO-006 | This catalog |
| WO-007 | Break **critical** cycles via shared abstractions |
| WO-008 | Break **high/medium** cycles via lazy imports and interfaces |

## Maintainer checklist

- After intentional graph changes: run `generate_importlinter_config.py`, then this script.
- Confirm every entry in `.ci/known_import_cycles.json` appears in the table above.
- Review resolution strategy for any new **critical** cycle before merging.
