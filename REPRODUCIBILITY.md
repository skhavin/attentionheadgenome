# Reproducing the HeadGenome Research

This guide explains how to run the pipeline to completely reproduce all experiments, data, and visualizations found in the `outputs/final_artifacts/HeadGenome_Master_Report.md`. 

The entire framework is designed to run locally or on a standard compute node with PyTorch and HuggingFace Transformers installed.

---

## Part 1: Generating the Multi-Model Atlases (Phase 2)

The foundation of the research is generating the `head_atlas.json` for all models. This involves running steps 0-6.

### 1. Extract the Standardized Dataset
First, generate the standardized dataset (Wikitext and Universal Dependencies). All models will be evaluated on this exact split to ensure fair comparison.
```bash
python phase2_atlas/step0_extract_dataset.py
```

### 2. Run the Multi-Model Pipeline
The script `phase2_atlas/run_all_models.py` orchestrates steps 1 through 6 (distance profiling, output norms, grammar mapping, softmax saturation, sink falsification, and compiling the final atlas).

**How to change models:**
Open `phase2_atlas/run_all_models.py` and edit the `MODELS` list at the top of the file:
```python
MODELS = [
    "gpt2-medium",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "unsloth/Llama-3.2-1B"
]
```
Then execute the pipeline:
```bash
python phase2_atlas/run_all_models.py
```
*Note: This will sequentially process all models and output their respective JSON files into the `outputs/phase2_atlas/` directory.*

---

## Part 2: Rigorous Statistical Analysis (Generating the Master Report Phase 2 Stats)

Once the JSON atlases are generated, you can run the rigorous statistical scripts that calculate the partial correlations, permutation nulls, and Z-tests mentioned in the Master Report.

1. **Law 1 & 11 (V/Q Scaling & Softmax Saturation):**
   ```bash
   python phase2_atlas/analyze_atlas_rigorous.py
   ```
   *This outputs the Pearson correlations, Partial Correlations (controlling for depth), and T-tests for softmax saturation.*

2. **Law 16 (KV Cache Mini-Sink / Punctuation Z-test):**
   ```bash
   python phase2_atlas/analyze_punctuation_rigorous.py
   ```
   *Calculates exact tokenizer-aligned base rates and tests the 96% punctuation allocation.*

---

## Part 3: Advanced Causal Interventions (Phase 3)

The mechanistic causal interventions are hardcoded to test specific behaviors on specific architectures (e.g., Qwen2.5).

1. **Multi-Head Ablation (Falsifying Law 2):**
   ```bash
   python phase2_atlas/step8_causal_patching.py
   ```
   *Runs the Needle-In-A-Haystack ablation on all 6 Retrieval heads simultaneously to test co-gating.*

2. **Polysemantic Multiplexing (Micro-SAE):**
   ```bash
   python phase2_atlas/step10_micro_sae.py
   ```
   *Trains the True SAE and Null SAE on `L9H7` and compares L0 sparsity.*

---

## Part 4: Universal Geometry & Lexical Tracking

1. **Universal Architecture Map:**
   ```bash
   python phase2_atlas/compare_atlases.py
   ```
   *Aggregates all 1,568 heads across models to calculate universal spatial enrichment and differences (e.g. Llama's Sink overload).*

2. **Lexical Target Separation:**
   ```bash
   python phase2_atlas/lexical_tracker.py
   ```
   *Proves that Induction heads track specific nouns while Local heads track structural scaffolding.*

---

## Part 5: Validating Atlas Roles via Attention Routing (Phase 4)

To verify that the classifications dictate actual model behavior, run the native routing engine which intercepts forward passes and constrains attention kernels.

1. **Run the Routing Engine:**
   ```bash
   python phase2_atlas/step18_routing_engine.py
   ```
2. **Validate Accuracy Drops:**
   ```bash
   python phase2_atlas/step19_routing_validation.py
   ```
   *Verifies that 32-token sliding windows on Local heads barely degrade HellaSwag, while restricting Sink heads lobotomizes the model.*

---

## Part 6: Unsupervised Emergent Discovery (Phase 5)

To reproduce the UMAP and HDBSCAN clustering that proves the continuous "Giant Megacluster" boundaries:

1. **Extract Rich Features:**
   ```bash
   python phase2_atlas/step15_rich_features.py
   ```
2. **Run Unsupervised Clustering:**
   ```bash
   python phase2_atlas/step16_emergent_discovery.py
   ```

---

## Part 7: Generating All Visualizations

Once all the data has been generated across the above steps, run the master visualization script to output all `.png` files (Sankeys, 2D Maps, Curves, UMAP clusters) into `outputs/final_artifacts/visualizations/`.

```bash
python outputs/final_artifacts/generate_visualizations.py
```
*These generated images are the exact ones referenced inside `HeadGenome_Master_Report.md`.*
