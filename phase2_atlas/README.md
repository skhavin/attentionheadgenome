# Phase 2 Atlas — Run Order

Run these scripts in order. Each step reads from `outputs/phase2_atlas/dataset.json`
(written by step 0) and writes its own output JSON.

```
python phase2_atlas/step0_extract_dataset.py   # run once — creates dataset.json
python phase2_atlas/step1_distance_profile.py  # ~5 min
python phase2_atlas/step2_ov_output_norm.py    # ~5 min
python phase2_atlas/step3_grammar_map.py       # ~5 min (uses UD-EWT)
python phase2_atlas/step4_softmax_saturation.py # ~5 min
python phase2_atlas/step5_sink_falsification.py # ~10 min (3x forward passes per prompt)
python phase2_atlas/step6_compile_atlas.py     # instant — merges all outputs
```

All outputs land in `outputs/phase2_atlas/`.

## File Map

| Script | Input | Output | Measures |
|--------|-------|--------|----------|
| step0 | HuggingFace | dataset.json | WikiText + UD-EWT samples |
| step1 | dataset.json | distance_profile.json | BOS mass, local mass, mean distance |
| step2 | dataset.json | ov_output_norm.json | V/Q structural ratio + runtime output norm |
| step3 | dataset.json (UD) | grammar_map.json | Attention mass by UD dep label |
| step4 | dataset.json | softmax_saturation.json | Max attention weight + entropy |
| step5 | dataset.json | sink_falsification.json | Entropy change when BOS removed/replaced |
| step6 | all above | head_atlas.json | Unified per-head card |
