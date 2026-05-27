# ADR 007: `trust_remote_code` security gate for Hub custom code

| Field | Value |
|-------|-------|
| Status | Accepted (ForgeScore modernization WO-020) |
| Date | 2026-05-26 |
| Work order | WO-020 |

## Context

Many Hub repositories ship Python modules not yet merged into Transformers (`modeling_*.py`,
`configuration_*.py`, tokenizers, custom pipelines). Loading them requires downloading and
**executing** remote source. That enables rapid research sharing but introduces code-execution
risk if a user loads an untrusted repo.

Transformers must default to **safe refusal** while allowing an explicit opt-in for vetted models.
The gate applies consistently across `AutoConfig`, `AutoModel`, tokenizers, processors, and
pipelines.

## Decision

1. **Default deny** — `trust_remote_code` defaults to `False`. If a checkpoint requires custom
   code that is not present in the installed package, loading fails with an actionable error unless
   the user passes `trust_remote_code=True`.

2. **Central resolution** — `resolve_trust_remote_code()` in `dynamic_module_utils.py` is the
   authoritative policy function. Callers pass whether local and/or remote custom code exists;
   the helper returns the resolved boolean or raises `ValueError`.

3. **Interactive prompt (optional)** — When remote code is needed, no local copy exists, and
   `trust_remote_code is None`, an interactive `[y/N]` prompt may appear (with timeout via
   `TIME_OUT_REMOTE_CODE`). Non-interactive environments (CI, servers) must pass
   `trust_remote_code` explicitly—prompt failure raises a clear `ValueError`.

4. **Dynamic module cache** — Approved remote code is copied into `HF_MODULES_CACHE`, imported as
   `transformers_modules.*`, and guarded by `_HF_REMOTE_CODE_LOCK`. Module names are sanitized
   (`_sanitize_module_name`) to avoid cache collisions and invalid identifiers.

5. **Local code precedence** — If the required class exists in the installed package
   (`has_local_code`), remote execution is not required and `trust_remote_code` stays `False` by
   default even when the repo also contains duplicate Hub files.

6. **No silent bypass** — Pipelines, tokenizers, and generation hooks that load custom modules
   document and enforce the same flag. Hub malware scanning is complementary, not a substitute for
   user opt-in.

7. **Revision pinning** — Users loading custom code should pin `revision=` to a commit hash after
   review (documented in custom models guides).

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Always execute Hub Python files | Unacceptable security default |
| Never execute remote code | Blocks the custom-model ecosystem and slow upstreaming |
| Implicit trust for official org repos | Spoofed namespaces and compromised accounts break org trust |
| WASM/sandboxed Python runtime | Not available in standard PyTorch user environments; huge scope |

## Consequences

- **Positive:** Explicit security boundary; consistent errors across Auto classes; cached imports
  amortize repeated loads; aligns with Hub security documentation.
- **Negative:** Extra argument for custom models; CI and servers must set the flag; interactive
  prompt unsuitable for batch jobs; users may blindly pass `True` without review.
- **Follow-up:** Keep docs prominent on custom models pages; ensure new load paths call
  `resolve_trust_remote_code`; consider stricter defaults in enterprise wrappers (out of scope here).

## Validation

- Load a known custom Hub model without `trust_remote_code` — expect `ValueError` with inspect URL.
- Same load with `trust_remote_code=True` — succeeds; module appears under cache path.
- In-tree model (e.g. BERT) — loads with default `False`; no remote execution.
- CI sets `TIME_OUT_REMOTE_CODE=0` or explicit flags so jobs never hang on stdin.

## References

- `src/transformers/dynamic_module_utils.py` — `resolve_trust_remote_code`, `get_class_from_dynamic_module`
- `src/transformers/models/auto/configuration_auto.py` — AutoConfig custom code path
- `docs/source/en/custom_models.md`
- [Hub security — malware scanning](https://huggingface.co/docs/hub/security)
- [ADR 005: Auto* registry pattern](./005-auto-class-registry-pattern.md)
