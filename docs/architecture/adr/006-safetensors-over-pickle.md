# ADR 006: Prefer safetensors over pickle for model weights

| Field | Value |
|-------|-------|
| Status | Accepted (ForgeScore modernization WO-020) |
| Date | 2026-05-26 |
| Work order | WO-020 |

## Context

PyTorch checkpoints were historically serialized with `torch.save` (pickle-based `.bin` files).
Pickle can execute arbitrary code during deserialization and is a common attack vector for
malicious Hub uploads. Safetensors provides a format with **no arbitrary code execution**, faster
mmap-friendly loads, and explicit tensor layout—now the Hub default for new models.

Transformers must load legacy `.bin` checkpoints for backward compatibility while preferring
safetensors whenever both exist, and defaulting saves to safetensors for new artifacts.

## Decision

1. **Load preference order** — In `PreTrainedModel.from_pretrained` / checkpoint resolution
   (`modeling_utils.py`), when `use_safetensors` is not explicitly `False`, prefer
   `model.safetensors` or sharded `*.safetensors` + index JSON over `pytorch_model.bin` /
   `model.bin` shards.

2. **Explicit opt-out** — `use_safetensors=False` forces pickle loading when a legacy checkpoint
   exists. Required for older checkpoints that cannot be converted (e.g. certain quantized or
   aliased tensor layouts).

3. **Save defaults** — `save_pretrained` writes safetensors via `safetensors.torch.save_file`
   unless the caller disables safetensors or the format cannot represent the state dict (documented
   exceptions in quantization docs).

4. **Hub auto-conversion** — `safetensors_conversion.auto_conversion` attempts on-the-fly
   conversion when only pickle weights are present and safetensors is allowed, reducing friction
   for legacy repos without changing default security posture.

5. **Validation of explicit filenames** — When `config.json` names a weights file, it must be
   safetensors or a safetensors index unless `use_safetensors=False`; mismatches raise clear
   errors rather than silently loading pickle.

6. **Documentation** — User-facing docs state that safetensors is preferred for security and
   performance (`docs/source/en/models.md`).

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Safetensors-only (drop pickle) | Breaks thousands of legacy Hub repos and local checkpoints |
| Pickle-first for speed on small models | Security regression; marginal gain vs mmap safetensors |
| Opaque auto-pickle when safetensors missing | Hides security choice; users must consciously opt in to pickle |
| Custom binary format | Ecosystem fragmentation; safetensors is Hub standard |

## Consequences

- **Positive:** Safer default load path; faster IO on large models; aligns with Hub and
  `compressed-tensors` extensions; clearer error when pickle would be required.
- **Negative:** Legacy-only repos need conversion or `use_safetensors=False`; some advanced
  checkpoint layouts still require pickle or manual handling; dual-format repos must keep files in
  sync.
- **Follow-up:** Encourage Hub model authors to publish safetensors; monitor quantization paths
  that still restrict safetensors save (see torchao version notes in docs).

## Validation

- Load a model with both `.safetensors` and `.bin` present — safetensors used by default.
- `use_safetensors=False` — pickle path taken when available.
- `save_pretrained` on a small model produces `model.safetensors` (or sharded equivalent).
- WO-027 config/serialization tests remain green after checkpoint logic changes.

## References

- `src/transformers/modeling_utils.py` — checkpoint resolution, `load_state_dict`
- `src/transformers/safetensors_conversion.py` — `auto_conversion`
- `docs/source/en/models.md` — safetensors preference
- [Hub safetensors documentation](https://huggingface.co/docs/safetensors)
