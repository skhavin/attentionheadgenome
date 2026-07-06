"""
Experiment 0: The Temporal Handoff — nMAD Edition
==================================================
Metric: Normalized Mean Attention Distance (nMAD)

  nMAD_h = sum_j(alpha[h, t, j] * (t - j)) / t

where t = last active token index (seq_len - 1 in Prefill, single new token index in Decode).
Division by t makes nMAD in [0,1] regardless of context length, enabling direct
Prefill/Decode comparison without sequence-length confound.

Pre-registered thresholds (DO NOT CHANGE after running):
  Induction PASS:
    - Decode nMAD / Prefill nMAD > 1.5
    - Wilcoxon signed-rank (per-prompt Decode > Prefill): p < 0.05
    - Mann-Whitney U (Induction shift > Control shift): p < 0.05
  NIAH cross-task:
    - Same nMAD shift > 1.5 on NIAH prompts

Partial-pass rule (pre-registered):
  If criteria 1-3 pass but NIAH (criterion 4) fails:
    -> Classified as "task-specific, arithmetic domain only"
    -> Experiments 1-3 restricted to arithmetic until NIAH replication succeeds.
  If any of criteria 1-3 fail:
    -> HARD GATE FAILED. Do NOT proceed to Experiments 1-3.
"""
import os
import sys
import json
import torch
import numpy as np
import random
from tqdm import tqdm
from scipy.stats import wilcoxon, mannwhitneyu

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from headgenome2_circuits.utils.model_loader import load_model_and_tokenizer

OUTPUT_DIR = "outputs/phase3_execution"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ARITH_DATASET_PATH = "headgenome2_circuits/datasets/arithmetic.json"

# ============================================================
# Pre-registered Thresholds — locked before running
# ============================================================
NMAD_RATIO_THRESHOLD = 1.5       # Decode nMAD / Prefill nMAD must exceed this
P_VALUE_THRESHOLD    = 0.05

# ============================================================
# NIAH Dataset Builder
# ============================================================
def build_niah_dataset(n: int = 50, seed: int = 42) -> list:
    """
    Builds N NIAH prompts. Each prompt has a 40-token filler context with a
    5-digit numeric UUID embedded, then a retrieval query. Forces Retrieval
    head engagement for the cross-task validation criterion.
    """
    random.seed(seed)
    words = ["the", "cat", "sat", "on", "mat", "a", "in", "by", "of", "and"]
    data = []
    for _ in range(n):
        uuid = f"{random.randint(10000, 99999)}-{random.randint(10000, 99999)}"
        pre  = " ".join(random.choices(words, k=20))
        post = " ".join(random.choices(words, k=20))
        prompt = f"{pre} The target code is {uuid}. {post} What is the target code? It is"
        data.append({"prompt": prompt, "uuid": uuid})
    return data

# ============================================================
# nMAD Computation Hook
# ============================================================
class NMADHook:
    """
    Registers forward pre-hooks on every self_attn module to capture the full
    attention weight tensor. Computes per-head nMAD at the last active token,
    split by phase (Prefill: seq_len > 1 / Decode: seq_len == 1).

    nMAD_h = (sum_j alpha[h, t, j] * (t - j)) / t
    where t = last active token index (0-indexed).

    Values are accumulated per-prompt via flush_prompt(), giving
      self.norms[(layer, head)]["prefill"] = [prompt_0_nmad, prompt_1_nmad, ...]
      self.norms[(layer, head)]["decode"]  = [prompt_0_nmad, ...]
    (one value per prompt, averaged over any multi-step Decode passes)
    """
    def __init__(self, num_layers: int, num_heads: int):
        self.num_layers = num_layers
        self.num_heads  = num_heads
        self._hooks     = []
        self.norms: dict = {}
        self._buf: dict  = {"prefill": {}, "decode": {}}  # accumulate within a prompt

    def _hook_fn(self, layer_idx: int):
        def hook(module, args, kwargs, output):
            # output from Qwen2Attention is (hidden_states, attn_weights, past_kv)
            # attn_weights shape: (batch, heads, seq, seq) — only present when
            # output_attentions=True; HF returns None otherwise.
            attn_weights = None
            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, torch.Tensor) and item.ndim == 4:
                        attn_weights = item
                        break
            if attn_weights is None:
                return  # model was called without output_attentions; skip

            _, num_heads, q_len, k_len = attn_weights.shape
            t = k_len - 1                   # absolute position of the current token
            if t == 0:
                return                        # single-token edge case: distance is 0

            phase = "prefill" if q_len > 1 else "decode"

            # distances shape: (k_len,)  — distance of each past token from t
            distances = torch.arange(t, -1, -1, dtype=attn_weights.dtype,
                                     device=attn_weights.device)  # [t, t-1, ..., 0]

            # The query we care about is the last token in the Q sequence
            q_idx = q_len - 1
            alpha = attn_weights[0, :, q_idx, :]  # (heads, k_len)

            # raw_mad per head: (heads,)
            raw_mad = (alpha * distances).sum(dim=-1)

            # normalize by t to get nMAD in [0, 1]
            nmad = (raw_mad / t).cpu().numpy()

            buf = self._buf[phase]
            for h in range(num_heads):
                key = (layer_idx, h)
                if key not in buf:
                    buf[key] = []
                buf[key].append(float(nmad[h]))

        return hook

    def register(self, model) -> None:
        for layer_idx, layer in enumerate(model.model.layers):
            handle = layer.self_attn.register_forward_hook(
                self._hook_fn(layer_idx),
                with_kwargs=True
            )
            self._hooks.append(handle)

    def flush_prompt(self) -> None:
        """Call once after each prompt's generate() to commit per-prompt averages."""
        for phase in ("prefill", "decode"):
            for key, vals in self._buf[phase].items():
                if key not in self.norms:
                    self.norms[key] = {"prefill": [], "decode": []}
                self.norms[key][phase].append(float(np.mean(vals)))
            self._buf[phase] = {}

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

# ============================================================
# Head Population Classifier
# ============================================================
def classify_heads(model, tokenizer, dataset: list,
                   num_heads: int, device) -> tuple[list, list]:
    """
    Classifies heads into Induction-candidates and Local/Sink control using
    static attention patterns on the arithmetic dataset.

    Induction/Counting: high attention from last token to distant content.
    Local/Sink: high attention from last token to immediate neighbors or token 0.

    Returns (induction_heads, control_heads) as lists of (layer, head) tuples.
    """
    num_layers = model.config.num_hidden_layers
    induction_scores = np.zeros((num_layers, num_heads))
    local_scores     = np.zeros((num_layers, num_heads))
    sink_scores      = np.zeros((num_layers, num_heads))
    count = 0

    for item in tqdm(dataset, desc="Classifying heads"):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)

        if out.attentions is None:
            raise RuntimeError("Model must be called with output_attentions=True for classification.")

        seq_len = inputs.input_ids.shape[1]
        t = seq_len - 1
        count += 1

        for l, attn in enumerate(out.attentions):
            a = attn[0, :, t, :]  # (heads, seq)
            # Induction: mass on tokens more than 3 positions back, EXCLUDING tokens 0 and 1 (Sink)
            far_mass = a[:, 2:max(2, t - 3)].sum(-1) if max(2, t - 3) > 2 else torch.zeros(num_heads, device=device)
            induction_scores[l] += far_mass.cpu().numpy()
            
            # Local: mass on the 3 immediately preceding tokens
            near = a[:, max(0, t-3):t]
            local_scores[l]     += near.sum(-1).cpu().numpy() if near.shape[-1] > 0 else 0
            
            # Sink: mass on position 0 and 1
            sink_scores[l]      += a[:, :2].sum(-1).cpu().numpy()

    induction_scores /= count
    local_scores     /= count
    sink_scores      /= count

    flat_i = induction_scores.flatten()
    flat_l = local_scores.flatten()
    flat_s = sink_scores.flatten()

    top_i = sorted(np.argsort(flat_i)[::-1][:10])
    top_l = list(np.argsort(flat_l)[::-1][:10])
    top_s = list(np.argsort(flat_s)[::-1][:10])

    induction_heads = [(int(idx // num_heads), int(idx % num_heads)) for idx in top_i]
    induction_set   = set(map(tuple, induction_heads))
    control_raw     = list(set(top_l + top_s))
    control_heads   = [
        (int(idx // num_heads), int(idx % num_heads))
        for idx in control_raw
        if (int(idx // num_heads), int(idx % num_heads)) not in induction_set
    ][:10]

    return induction_heads, control_heads

# ============================================================
# nMAD Measurement
# ============================================================
def measure_nmad(model, tokenizer, dataset: list,
                 target_heads: list, label: str, device) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs generate() on each prompt (max_new_tokens=2 to guarantee one real Decode step),
    hooks attention weights, computes per-prompt nMAD for target_heads, returns
    (per_prompt_prefill_nmad, per_prompt_decode_nmad) arrays of shape (N_prompts,).
    """
    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads

    tracker = NMADHook(num_layers, num_heads)
    tracker.register(model)

    for item in tqdm(dataset, desc=f"nMAD [{label}]"):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=2,
                pad_token_id=tokenizer.eos_token_id,
                output_attentions=True,
                return_dict_in_generate=True,
            )
        tracker.flush_prompt()

    tracker.remove()

    n_prompts = len(dataset)
    pf_list, dc_list = [], []
    for h in target_heads:
        key = tuple(h)
        if key in tracker.norms:
            pf = tracker.norms[key]["prefill"]
            dc = tracker.norms[key]["decode"]
            if len(pf) == n_prompts and len(dc) == n_prompts:
                pf_list.append(pf)
                dc_list.append(dc)

    if not pf_list:
        print(f"  WARNING [{label}]: No valid per-prompt norms found for target heads.")
        return np.array([]), np.array([])

    return np.mean(pf_list, axis=0), np.mean(dc_list, axis=0)

# ============================================================
# Gate Evaluator
# ============================================================
def evaluate_gate(name: str, prefill: np.ndarray, decode: np.ndarray,
                  ratio_threshold: float) -> dict:
    """
    Applies pre-registered criteria:
      - Mean Decode/Prefill nMAD ratio > ratio_threshold
      - Wilcoxon signed-rank (Decode > Prefill, per prompt): p < P_VALUE_THRESHOLD
    """
    if len(prefill) == 0 or len(decode) == 0:
        return {"name": name, "gate": False, "reason": "Empty — heads not engaged on task."}

    ratios   = decode / (prefill + 1e-9)
    mean_r   = float(np.mean(ratios))
    std_r    = float(np.std(ratios))
    mean_pf  = float(np.mean(prefill))
    mean_dc  = float(np.mean(decode))

    try:
        _, p_wil = wilcoxon(decode, prefill, alternative="greater")
    except Exception:
        p_wil = 1.0

    ratio_pass   = mean_r > ratio_threshold
    wilcoxon_pass = p_wil < P_VALUE_THRESHOLD

    print(f"\n  [{name}]")
    print(f"    Prefill nMAD : {mean_pf:.4f}")
    print(f"    Decode  nMAD : {mean_dc:.4f}")
    print(f"    Ratio (D/P)  : {mean_r:.2f} ± {std_r:.2f}  (threshold > {ratio_threshold})")
    print(f"    Wilcoxon p   : {p_wil:.4f}  (threshold < {P_VALUE_THRESHOLD})")
    print(f"    ratio_pass={ratio_pass}  wilcoxon_pass={wilcoxon_pass}")

    return {
        "name":          name,
        "mean_prefill":  mean_pf,
        "mean_decode":   mean_dc,
        "mean_ratio":    mean_r,
        "std_ratio":     std_r,
        "wilcoxon_p":    float(p_wil),
        "ratio_pass":    bool(ratio_pass),
        "wilcoxon_pass": bool(wilcoxon_pass),
        "gate":          bool(ratio_pass and wilcoxon_pass),
    }

# ============================================================
# Main
# ============================================================
def run_experiment_0(model_key: str = "qwen-0.5b", n_prompts: int = 50) -> None:
    print(f"\n{'='*60}")
    print(f"Experiment 0 (nMAD Edition) — {model_key}")
    print(f"Pre-registered thresholds: ratio>{NMAD_RATIO_THRESHOLD}, p<{P_VALUE_THRESHOLD}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(model_key, output_attentions=True)
    device    = model.device
    num_heads = model.config.num_attention_heads

    # Datasets
    with open(ARITH_DATASET_PATH) as f:
        arith_dataset = json.load(f)[:n_prompts]
    niah_dataset = build_niah_dataset(n_prompts)

    # Step 1: Classify head populations on the long-context NIAH dataset
    print("\n--- Step 1: Classifying Head Populations (NIAH prompts) ---")
    induction_heads, control_heads = classify_heads(
        model, tokenizer, niah_dataset, num_heads, device
    )
    print(f"  Induction candidates : {induction_heads[:5]}")
    print(f"  Control (Local/Sink) : {control_heads[:5]}")

    # Step 2: nMAD on arithmetic
    print("\n--- Step 2: nMAD Measurement — Arithmetic ---")
    ind_pf_arith, ind_dc_arith  = measure_nmad(model, tokenizer, arith_dataset,
                                                induction_heads, "Induction/Arith", device)
    ctl_pf_arith, ctl_dc_arith  = measure_nmad(model, tokenizer, arith_dataset,
                                                control_heads,   "Control/Arith",   device)

    # Step 3: nMAD on NIAH
    print("\n--- Step 3: nMAD Measurement — NIAH ---")
    ind_pf_niah, ind_dc_niah    = measure_nmad(model, tokenizer, niah_dataset,
                                                induction_heads, "Induction/NIAH",  device)

    # Step 4: Statistical evaluation
    print("\n" + "="*60)
    print("EXPERIMENT 0 RESULTS")
    print("="*60)

    r_ind_arith = evaluate_gate("Induction / Arithmetic",
                                ind_pf_arith, ind_dc_arith, NMAD_RATIO_THRESHOLD)
    r_ctl_arith = evaluate_gate("Control / Arithmetic  [should NOT pass]",
                                ctl_pf_arith, ctl_dc_arith, NMAD_RATIO_THRESHOLD)
    r_ind_niah  = evaluate_gate("Induction / NIAH (cross-task)",
                                ind_pf_niah,  ind_dc_niah,  NMAD_RATIO_THRESHOLD)

    # Mann-Whitney specificity: Induction shift > Control shift?
    mwu_p = 1.0
    mwu_pass = False
    if len(ind_pf_arith) > 0 and len(ctl_pf_arith) > 0:
        ind_shift = ind_dc_arith / (ind_pf_arith + 1e-9)
        ctl_shift = ctl_dc_arith / (ctl_pf_arith + 1e-9)
        try:
            _, mwu_p = mannwhitneyu(ind_shift, ctl_shift, alternative="greater")
        except Exception:
            mwu_p = 1.0
        mwu_pass = bool(mwu_p < P_VALUE_THRESHOLD)
        print(f"\n  Mann-Whitney U (Induction shift > Control shift): p = {mwu_p:.4f}  pass={mwu_pass}")

    # Gate decision
    core_pass = r_ind_arith["gate"] and mwu_pass          # criteria 1-3
    niah_pass = r_ind_niah["gate"]                         # criterion 4

    print("\n" + "="*60)
    print("FINAL GATE DECISION")
    print("="*60)
    print(f"  Criterion 1 (ratio > {NMAD_RATIO_THRESHOLD}) : {r_ind_arith['ratio_pass']}")
    print(f"  Criterion 2 (Wilcoxon p < {P_VALUE_THRESHOLD}) : {r_ind_arith['wilcoxon_pass']}")
    print(f"  Criterion 3 (Mann-Whitney specificity)   : {mwu_pass}")
    print(f"  Criterion 4 (NIAH cross-task)            : {niah_pass}")
    print(f"  Control correctly does NOT pass          : {not r_ctl_arith['gate']}")

    if core_pass and niah_pass:
        gate_result = "FULL PASS — proceed to Experiments 1-3 across all tasks."
    elif core_pass and not niah_pass:
        gate_result = "PARTIAL PASS — arithmetic domain only. Restrict Experiments 1-3 to arithmetic until NIAH replication succeeds."
    else:
        gate_result = "FAILED — do NOT proceed to Experiments 1-3. Reassess hypothesis."

    print(f"\n  HARD GATE: {gate_result}")

    # Persist results
    def to_python(obj):
        if isinstance(obj, dict):  return {k: to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [to_python(i) for i in obj]
        if isinstance(obj, (np.bool_, bool)):    return bool(obj)
        if isinstance(obj, np.integer):          return int(obj)
        if isinstance(obj, np.floating):         return float(obj)
        if isinstance(obj, np.ndarray):          return obj.tolist()
        return obj

    results = to_python({
        "model": model_key,
        "pre_registered_thresholds": {
            "nmad_ratio": NMAD_RATIO_THRESHOLD,
            "p_value":    P_VALUE_THRESHOLD,
        },
        "induction_heads": induction_heads,
        "control_heads":   control_heads,
        "results": {
            "induction_arith": r_ind_arith,
            "control_arith":   r_ctl_arith,
            "induction_niah":  r_ind_niah,
            "mann_whitney_p":  float(mwu_p),
            "mann_whitney_pass": mwu_pass,
        },
        "gate_result":    gate_result,
        "core_pass":      core_pass,
        "niah_pass":      niah_pass,
        "full_gate_pass": core_pass and niah_pass,
    })

    out_path = os.path.join(OUTPUT_DIR, f"exp0_nmad_{model_key}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment_0("qwen-0.5b")
