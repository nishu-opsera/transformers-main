# Modular migration progress (WO-017)

Tracking top **50** `modeling_*.py` files by `# Copied from` count.

| File | Annotations (baseline) | Modular source | Status |
|------|------------------------|----------------|--------|
| `src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py` | 58 | `—` | pending_modular |
| `src/transformers/models/mt5/modeling_mt5.py` | 31 | `—` | pending_modular |
| `src/transformers/models/owlv2/modeling_owlv2.py` | 30 | `src/transformers/models/owlv2/modular_owlv2.py` | modular_source |
| `src/transformers/models/roc_bert/modeling_roc_bert.py` | 28 | `—` | pending_modular |
| `src/transformers/models/xmod/modeling_xmod.py` | 22 | `—` | pending_modular |
| `src/transformers/models/flaubert/modeling_flaubert.py` | 20 | `—` | pending_modular |
| `src/transformers/models/umt5/modeling_umt5.py` | 20 | `—` | pending_modular |
| `src/transformers/models/idefics3/modeling_idefics3.py` | 18 | `—` | pending_modular |
| `src/transformers/models/markuplm/modeling_markuplm.py` | 18 | `—` | pending_modular |
| `src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py` | 17 | `—` | pending_modular |
| `src/transformers/models/bridgetower/modeling_bridgetower.py` | 15 | `—` | pending_modular |
| `src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` | 15 | `—` | pending_modular |
| `src/transformers/models/deberta_v2/modeling_deberta_v2.py` | 13 | `—` | pending_modular |
| `src/transformers/models/sew_d/modeling_sew_d.py` | 13 | `—` | pending_modular |
| `src/transformers/models/speecht5/modeling_speecht5.py` | 13 | `—` | pending_modular |
| `src/transformers/models/clap/modeling_clap.py` | 12 | `—` | pending_modular |
| `src/transformers/models/electra/modeling_electra.py` | 12 | `—` | pending_modular |
| `src/transformers/models/instructblip/modeling_instructblip.py` | 12 | `—` | pending_modular |
| `src/transformers/models/kosmos2_5/modeling_kosmos2_5.py` | 12 | `—` | pending_modular |
| `src/transformers/models/layoutlm/modeling_layoutlm.py` | 12 | `—` | pending_modular |
| `src/transformers/models/mimi/modeling_mimi.py` | 12 | `—` | pending_modular |
| `src/transformers/models/patchtsmixer/modeling_patchtsmixer.py` | 12 | `—` | pending_modular |
| `src/transformers/models/align/modeling_align.py` | 11 | `—` | pending_modular |
| `src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py` | 11 | `—` | pending_modular |
| `src/transformers/models/longt5/modeling_longt5.py` | 11 | `—` | pending_modular |
| `src/transformers/models/musicgen_melody/modeling_musicgen_melody.py` | 11 | `—` | pending_modular |
| `src/transformers/models/owlvit/modeling_owlvit.py` | 11 | `—` | pending_modular |
| `src/transformers/models/bert_generation/modeling_bert_generation.py` | 10 | `—` | pending_modular |
| `src/transformers/models/oneformer/modeling_oneformer.py` | 10 | `—` | pending_modular |
| `src/transformers/models/pop2piano/modeling_pop2piano.py` | 10 | `—` | pending_modular |
| `src/transformers/models/tapas/modeling_tapas.py` | 10 | `—` | pending_modular |
| `src/transformers/models/autoformer/modeling_autoformer.py` | 9 | `—` | pending_modular |
| `src/transformers/models/big_bird/modeling_big_bird.py` | 9 | `—` | pending_modular |
| `src/transformers/models/blip_2/modeling_blip_2.py` | 9 | `—` | pending_modular |
| `src/transformers/models/data2vec/modeling_data2vec_vision.py` | 9 | `—` | pending_modular |
| `src/transformers/models/idefics2/modeling_idefics2.py` | 9 | `—` | pending_modular |
| `src/transformers/models/lilt/modeling_lilt.py` | 9 | `—` | pending_modular |
| `src/transformers/models/mbart/modeling_mbart.py` | 9 | `—` | pending_modular |
| `src/transformers/models/megatron_bert/modeling_megatron_bert.py` | 9 | `—` | pending_modular |
| `src/transformers/models/stablelm/modeling_stablelm.py` | 9 | `—` | pending_modular |
| `src/transformers/models/udop/modeling_udop.py` | 9 | `—` | pending_modular |
| `src/transformers/models/blenderbot/modeling_blenderbot.py` | 8 | `—` | pending_modular |
| `src/transformers/models/blenderbot_small/modeling_blenderbot_small.py` | 8 | `—` | pending_modular |
| `src/transformers/models/chameleon/modeling_chameleon.py` | 8 | `—` | pending_modular |
| `src/transformers/models/donut/modeling_donut_swin.py` | 8 | `—` | pending_modular |
| `src/transformers/models/git/modeling_git.py` | 8 | `—` | pending_modular |
| `src/transformers/models/kosmos2/modeling_kosmos2.py` | 8 | `—` | pending_modular |
| `src/transformers/models/mllama/modeling_mllama.py` | 8 | `—` | pending_modular |
| `src/transformers/models/moshi/modeling_moshi.py` | 8 | `—` | pending_modular |
| `src/transformers/models/mra/modeling_mra.py` | 8 | `—` | pending_modular |

Regenerate:

```bash
python utils/track_modular_migration.py --write
python utils/catalog_copied_from.py  # refresh annotation counts
```
