# HeadGenome 2: Mechanistic Origin & Policy Synthesis — Revised Plan

## Goal Description
Determine whether structural and mechanistic features of attention heads can predict functional taxonomy (Local, Sink, Retrieval, Induction) across architectures, **beyond what depth alone explains**. A strict falsifiability gate (Phase 0) controls whether Papers 2/3 have a viable claim.

**Taxonomy for Phase 0:** 4 classes: `local`, `sink`, `retrieval`, `induction`. Early/Late Induction is **collapsed** back to a single `induction` class here, consistent with Table 4's canonical labels in `canonical_labels.json`.

---

## Fixes Incorporated (from review)

### Fix 1 — Primary Metric: Balanced Accuracy + Macro-F1
Raw accuracy is not used as primary. Given Local = 84.1% of heads, a trivial classifier reaches ~84% accuracy. `01_phase0_gate.py` uses:
- **Primary:** Macro-averaged F1 (equal weight per class)
- **Secondary:** Balanced accuracy (average per-class recall)
- **Per-class recall** reported in JSON for all 4 classes
- Retrieval's extreme rarity (~5-6 heads per LOAO fold) is **explicitly flagged** in the output

### Fix 2 — Regularization & Stronger Depth Null
- L2 regularization set explicitly: `C=0.1` (moderate), reported in output JSON
- **Model A**: depth (continuous relative, 0→1) + **binned depth** (10 deciles, one-hot) — this prevents winning just on nonlinear depth encoding
- **Model B**: Model A features + full structural feature bank
- Both use identical preprocessing pipeline

### Fix 3 — Shuffled-Label Pass/Fail Bar
Shuffled-label Model B must land within ±2% of theoretical chance Macro-F1:
- 4-class uniform chance: `0.25`
- Adjusted chance (accounting for imbalance): `sum(pi^2) = 0.84^2 + ...` ≈ reported in JSON
- Hard-coded pass/fail criterion in `01_phase0_gate.py` — not a visual check

### Fix 4 — All 4 Individual LOAO Results Reported
Every architecture appears individually in the output JSON:
- `loao_gpt2`, `loao_qwen-0.5b`, `loao_qwen-1.5b`, `loao_llama-3.2-1b`
- GPT-2 (MHA) vs. GQA asymmetry explicitly noted if present
- **No means-only reporting** — the full distribution is preserved

### Fix 5 — Entropy/Locality Reuses Original 50-Prompt Logs
Feature extraction in `00_extract_feature_bank.py` pulls locality and entropy from the existing atlas JSONs (`phase2_atlas/*_head_atlas.json`) which were computed on 200 prompts, same ones used for labeling. A fallback minimal forward-pass is only triggered if a head is missing from the atlas, and that discrepancy is counted and flagged.

### Fix 6 — Paired Bootstrap CI on the Gap (B − A)
The gap statistic `Macro-F1(Model B) − Macro-F1(Model A)` uses a **paired bootstrap** (1000 resamples of heads, refit both models each time) to produce a true CI on the gap itself — not two separate CIs eyeballed for overlap.

---

## Feature Bank (Phase 0)

### Static Weight Features (from model weights, no forward pass)
| Feature | Source |
|---|---|
| `vq_ratio` | `head_atlas.json` |
| `wq_norm`, `wk_norm`, `wv_norm`, `wo_norm` | computed per head |
| `ov_norm` (`||W_O W_V||_F`) | computed per head |
| `qk_effective_rank` | SVD of sampled QK weight product |
| `ov_effective_rank` | SVD of `W_O @ W_V` |
| `qk_top3_sv` | Top-3 singular values of QK |
| `ov_top3_sv` | Top-3 singular values of OV |
| `head_dim` | from config |
| `is_gqa` | architecture flag |
| `gqa_group_id` | KV-sharing group index (GQA only) |

### Behavioral/Activation Features (from existing atlas logs)
| Feature | Source |
|---|---|
| `match_entropy` | `head_atlas.json → entropy_profile` |
| `delta_collapse` | `head_atlas.json → entropy_profile` |
| `mean_distance` | `head_atlas.json → attention_geometry` |
| `bos_mass` | `head_atlas.json → attention_geometry` |
| `local_mass` | `head_atlas.json → attention_geometry` |
| `long_range_mass` | `head_atlas.json → attention_geometry` |
| `mean_max_attn` (softmax peak) | `head_atlas.json → softmax_saturation` |
| `locality_score` | `regime_switching_*.json` (top-10 stable/switcher lists) |

---

## Proposed Files

### `headgenome4_policy_synthesis/`
- **[NEW]** `00_extract_feature_bank.py` — Loads all 4 atlas JSONs + canonical labels, extracts the full feature bank, and writes `outputs/phase0/feature_bank.csv`
- **[NEW]** `01_phase0_gate.py` — Trains Model A vs B, LOAO, paired bootstrap, shuffled control, writes `outputs/phase0/gate_results.json`

---

## Gate Outcome Interpretation
| Result | Action |
|---|---|
| Macro-F1(B) − Macro-F1(A) > 0.10, CI excludes 0 | **Phase 0 PASSES** → proceed to Phase 1 |
| Gap < 0.10 or CI includes 0 | **Phase 0 FAILS** → pivot to OV+composition features (Phase 0B) |
| Shuffled control > chance + 4% | **Implementation error** → debug before interpreting |
| GPT-2 LOAO fails but GQA models pass | **Report as architectural asymmetry** — potential GQA-specific predictor story |

---

## Verification Plan
- `outputs/phase0/feature_bank.csv` — inspectable DataFrame before running the gate
- `outputs/phase0/gate_results.json` — contains all LOAO scores individually, bootstrap CIs, shuffled control result, pass/fail verdict
