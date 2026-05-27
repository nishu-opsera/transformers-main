# DeviceContext migration guide (WO-019)

`DeviceContext` centralizes device and dtype placement at model boundaries. See
[ADR 002](../../architecture/adr/002-device-context.md).

## When to use DeviceContext

| Situation | Recommendation |
|-----------|----------------|
| New model `forward` / `generate` entry | Wrap inputs with `DeviceContext` |
| Internal submodule | Keep tensors on active device; avoid redundant `.to()` |
| Training with Accelerate/FSDP | Use framework device; `DeviceContext` for local batch tensors only |

## Encoder-only (BERT-style)

**Before:**

```python
def forward(self, input_ids=None, **kwargs):
    input_ids = input_ids.to(self.device)
    return self.encoder(input_ids)
```

**After:**

```python
from transformers.utils.device_context import DeviceContext

def forward(self, input_ids=None, **kwargs):
    with DeviceContext(self.device) as ctx:
        batch = ctx.move({"input_ids": input_ids})
        return self.encoder(**batch)
```

## Decoder-only (GPT-style)

Use `DeviceContext` at the start of `forward` and move `past_key_values` containers
recursively via `ctx.move()`.

## Encoder–decoder

Apply context separately for encoder inputs and decoder inputs if they may use
different devices during export; typically one context per `forward` call.

## Multimodal

Move each modality dict (pixel values, input ids) inside a single context so nested
lists and tuples are handled consistently.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Tensor on wrong device | Ensure `forward` entered `DeviceContext` before submodules run |
| Nested context conflict | Inner context should match outer device or only wrap a subtree |
| dtype mismatch | Pass `dtype=self.dtype` to `DeviceContext` |

## Further reading

- API: `src/transformers/utils/device_context.py`
- Tests: `tests/utils/test_device_context.py`
