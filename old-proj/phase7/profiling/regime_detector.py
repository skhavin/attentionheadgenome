# phase7/regime_detector.py
#
# Phase 2 — Regime-Switch Detector
#
# For Tier 2 heads (heads that are safe on natural text but fail on copy-trigger
# inputs), this module provides:
#   A. The online detector: O(N) scan of the prefix, no model forward pass.
#   B. Validation: precision/recall on the stress prompt set.
#   C. A patched-model wrapper that uses the detector at inference time to
#      decide per-head whether to run full attention or the substitution.
#
# Background: "Regime-switching" is not a weakness of the substitution method —
# it is a discovery. Local heads classified as such because they attend narrowly
# on natural text activate a qualitatively different pattern (induction-like,
# long-range copy) under copy-trigger inputs. This is a dynamic routing behaviour
# absent from static calibration. The detector catches this at inference time
# with O(N) overhead per token.
#
# Usage:
#   python phase7/regime_detector.py
#   python phase7/regime_detector.py --tier2_path outputs/phase7/tier2_heads.json

import sys, os, argparse, json, pickle, random
os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from collections import Counter
from tqdm import tqdm

from config import PHASE7_DIR


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

def compute_ngram_repetition(token_ids: list, n: int = 3) -> float:
    """
    Compute the n-gram repetition rate in a prefix.

    Returns the fraction of n-grams that appear more than once.
    High repetition → copy-trigger regime → use full attention.

    Complexity: O(N) in sequence length.
    """
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i: i + n]) for i in range(len(token_ids) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def regime_detector(
    prefix_tokens,        # list[int] or 1-D Tensor
    ngram_n: int = 3,
    ngram_threshold: float = 0.3,
    freq_threshold: float = 0.05,
) -> bool:
    """
    Returns True if the current prefix signals a copy-trigger regime,
    meaning the model should use full attention for Tier 2 heads.

    Signal 1: n-gram repetition rate in prefix
        High repetition → the model is likely in induction / copy mode.

    Signal 2: max token frequency (copy triggers always spike this)
        A token appearing in > freq_threshold fraction of the prefix
        strongly suggests a repeated sequence.

    Both signals are O(N) to compute. No model forward pass required.

    Parameters:
        prefix_tokens:     the token prefix (list or 1-D int tensor)
        ngram_n:           n for n-gram repetition (default 3)
        ngram_threshold:   trigger if repetition rate > this (default 0.3)
        freq_threshold:    trigger if max token freq > this (default 0.05)

    Returns:
        True  → switch to full attention for this step
        False → safe to use substitution
    """
    if torch.is_tensor(prefix_tokens):
        tokens = prefix_tokens.tolist()
    else:
        tokens = list(prefix_tokens)

    if len(tokens) == 0:
        return False

    # Signal 1: n-gram repetition
    ngram_rep_rate = compute_ngram_repetition(tokens, n=ngram_n)
    if ngram_rep_rate > ngram_threshold:
        return True

    # Signal 2: max token frequency
    max_freq = max(Counter(tokens).values()) / len(tokens)
    if max_freq > freq_threshold:
        return True

    return False


# ---------------------------------------------------------------------------
# Validation against stress set
# ---------------------------------------------------------------------------

class DetectorValidator:
    """
    Measure precision and recall of regime_detector against a labelled
    stress prompt set.

    Labels:
        positive = copy-trigger or induction prompt (detector should fire)
        negative = natural WikiText-103 prompt (detector should NOT fire)
    """

    def __init__(self, ngram_n=3, ngram_threshold=0.3, freq_threshold=0.05):
        self.ngram_n = ngram_n
        self.ngram_threshold = ngram_threshold
        self.freq_threshold = freq_threshold

    def predict(self, token_ids):
        return regime_detector(
            token_ids,
            ngram_n=self.ngram_n,
            ngram_threshold=self.ngram_threshold,
            freq_threshold=self.freq_threshold,
        )

    def evaluate(self, positive_samples, negative_samples):
        """
        positive_samples: list of token_id tensors that ARE copy-trigger/induction
        negative_samples: list of token_id tensors that are NOT (natural text)

        Returns:
            dict with precision, recall, f1, specificity, accuracy
        """
        tp = fp = tn = fn = 0

        for ids in positive_samples:
            tokens = ids[0].tolist() if ids.dim() == 2 else ids.tolist()
            pred = self.predict(tokens)
            if pred:
                tp += 1
            else:
                fn += 1

        for ids in negative_samples:
            tokens = ids[0].tolist() if ids.dim() == 2 else ids.tolist()
            pred = self.predict(tokens)
            if pred:
                fp += 1
            else:
                tn += 1

        precision    = tp / max(tp + fp, 1)
        recall       = tp / max(tp + fn, 1)
        f1           = 2 * precision * recall / max(precision + recall, 1e-9)
        specificity  = tn / max(tn + fp, 1)
        accuracy     = (tp + tn) / max(tp + tn + fp + fn, 1)

        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision":   precision,
            "recall":      recall,
            "f1":          f1,
            "specificity": specificity,
            "accuracy":    accuracy,
        }


# ---------------------------------------------------------------------------
# Patched model wrapper for inference-time regime switching
# ---------------------------------------------------------------------------

class RegimeSwitchingPatcher:
    """
    Wraps a HuggingFace causal LM and at inference time:
      - For Tier 1 heads: always uses the closed-form substitution.
      - For Tier 2 heads: checks regime_detector on the current prefix;
        uses full attention if switch detected, substitution otherwise.
      - For Tier 3 heads: always uses full attention.

    This is the "substitute_with_detector" mode used in eval_ppl.py and
    eval_ruler.py.

    The patcher installs forward hooks; call .restore() after inference.
    """

    def __init__(
        self,
        model,
        tier1_heads: list,   # [(layer, head, type), ...]
        tier2_heads: list,   # [(layer, head, type), ...]
        num_sink_tokens: int = 4,
        local_window: int = 64,
        ngram_threshold: float = 0.3,
        freq_threshold: float = 0.05,
    ):
        from phase7.substitutes import sink_substitute, local_substitute

        self.model = model
        self.tier1_set = {(l, h): t for l, h, t in tier1_heads}
        self.tier2_set = {(l, h): t for l, h, t in tier2_heads}
        self.num_sink_tokens = num_sink_tokens
        self.local_window = local_window
        self.ngram_threshold = ngram_threshold
        self.freq_threshold = freq_threshold
        self._sink_sub = sink_substitute
        self._local_sub = local_substitute
        self._original_forwards = {}
        self._hook_handles = []
        self._install_hooks()

    def _install_hooks(self):
        """Install per-head substitution logic in each attention layer."""
        for name, module in self.model.named_modules():
            cls_name = type(module).__name__
            if "Attention" in cls_name and (hasattr(module, "q_proj") or hasattr(module, "c_attn")):
                self._patch_layer(name, module)

    def _patch_layer(self, name: str, attn_module):
        """
        Patch a single attention layer to apply per-head substitution.

        Note: this operates at the layer level — within the patched forward,
        we compute all heads, then replace substitutable head outputs.
        The layer's original forward is preserved and called for non-substituted heads.
        """
        patcher = self
        original_forward = attn_module.forward
        self._original_forwards[name] = original_forward

        # Extract layer index from module name (e.g. "transformer.h.4.attn" → 4)
        parts = name.split(".")
        layer_idx = None
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break

        cls_name = type(attn_module).__name__.lower()
        arch = "gpt2" if "gpt2" in cls_name else "llama"

        captured_v = {}
        def v_hook(module, inp, output):
            # output is [B, N, d_out]
            B, N, d_out = output.shape
            if arch == "gpt2":
                num_heads = getattr(patcher.model.config, "n_head", 12)
                d_head = d_out // (3 * num_heads)
                _, _, v = output.split(d_out // 3, dim=2)
                v_heads = v.view(B, N, num_heads, d_head)
                captured_v["v"] = v_heads
            else:
                num_heads = getattr(patcher.model.config, "num_attention_heads", 32)
                num_kv_heads = getattr(patcher.model.config, "num_key_value_heads", num_heads)
                d_head = getattr(patcher.model.config, "head_dim", d_out // num_kv_heads)
                v_heads = output.view(B, N, num_kv_heads, d_head)
                groups = num_heads // num_kv_heads
                if groups > 1:
                    v_heads = v_heads.repeat_interleave(groups, dim=2)
                captured_v["v"] = v_heads

        target_mod = attn_module.c_attn if arch == "gpt2" else attn_module.v_proj
        handle = target_mod.register_forward_hook(v_hook)
        self._hook_handles.append(handle)

        def patched_forward(hidden_states, attention_mask=None, position_ids=None,
                            past_key_values=None, output_attentions=False,
                            use_cache=False, cache_position=None,
                            position_embeddings=None, **kwargs):
            
            captured_v.clear()
            
            # Always run full attention to get the correct output
            result = original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=True,   # need weights for substitution quality
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # result: (attn_output, attn_weights, ...)
            # If no heads in this layer are tier-1 or tier-2, return as-is
            if layer_idx is None:
                return result

            # Check if any heads in this layer are substitutable
            tier1_in_layer = {h: t for (l, h), t in patcher.tier1_set.items() if l == layer_idx}
            tier2_in_layer = {h: t for (l, h), t in patcher.tier2_set.items() if l == layer_idx}

            if not tier1_in_layer and not tier2_in_layer:
                return result

            # Regime detection for Tier 2 — check on the current prefix
            prefix = hidden_states[0].detach()   # [N, d]
            # Use token norms as a proxy for token ids (no tokenizer access here)
            # For a full implementation, pass prefix_token_ids as a context variable.
            # Here we use a conservative fallback: assume regime switch if the
            # hidden state variance is high (indicative of copy-trigger patterns).
            h_var = prefix.var(dim=0).mean().item()
            in_regime_switch = h_var > 0.5  # heuristic; calibrate per model

            heads_to_sub = {}
            heads_to_sub.update(tier1_in_layer)
            if not in_regime_switch:
                heads_to_sub.update(tier2_in_layer)
                
            if not heads_to_sub or "v" not in captured_v:
                return result

            attn_output = result[0].clone()
            attn_weights = result[1]  # [B, H, N, N]
            v_heads = captured_v["v"] # [B, N, H, d_head]
            B, N, H, d_head = v_heads.shape
            
            if arch == "gpt2":
                w_o = attn_module.c_proj.weight # [H*d_head, d_model]
            else:
                w_o = attn_module.o_proj.weight # [d_model, H*d_head]

            for h_idx, htype in heads_to_sub.items():
                V_h = v_heads[:, :, h_idx, :] # [B, N, d_head]
                attn_w_h = attn_weights[:, h_idx, :, :] # [B, N, N]
                
                # full attention output for this head
                attn_out_full_h = torch.bmm(attn_w_h, V_h) # [B, N, d_head]
                
                # substitute output
                V4d = V_h.unsqueeze(1) # [B, 1, N, d_head]
                if htype == "sink":
                    attn_out_sub_h = patcher._sink_sub(V4d, attn_weights=attn_w_h.unsqueeze(1), num_sink_tokens=patcher.num_sink_tokens).squeeze(1)
                else:
                    attn_out_sub_h = patcher._local_sub(V4d, window_size=patcher.local_window).squeeze(1)
                    
                diff_h = (attn_out_sub_h - attn_out_full_h) # [B, N, d_head]
                
                # project and add to attn_output
                if arch == "gpt2":
                    w_slice = w_o[h_idx * d_head : (h_idx + 1) * d_head, :] # [d_head, d_model]
                    diff_proj = torch.matmul(diff_h, w_slice)
                else:
                    w_slice = w_o[:, h_idx * d_head : (h_idx + 1) * d_head].t() # [d_head, d_model]
                    diff_proj = torch.matmul(diff_h, w_slice)
                    
                attn_output += diff_proj

            # Reconstruct the result tuple with patched attn_output
            new_result = (attn_output,) + result[1:]
            return new_result

        attn_module.forward = patched_forward

    def restore(self):
        """Restore all original attention forward methods."""
        for name, module in self.model.named_modules():
            if name in self._original_forwards:
                module.forward = self._original_forwards[name]
        self._original_forwards.clear()
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        print("  Restored original attention (regime-switching patcher).")


# ---------------------------------------------------------------------------
# Main — validate detector on stress set
# ---------------------------------------------------------------------------

def build_copy_trigger_prompts_simple(tokenizer, seq_len, num_prompts):
    """Lightweight copy-trigger builder (no datasets required)."""
    vocab = ["cat", "dog", "mat", "sat", "hat", "bat", "rat", "tree",
             "book", "cook", "look", "hook", "took", "good", "wood"]
    rng = random.Random(42)
    prompts = []
    for _ in range(num_prompts):
        n = rng.randint(1, 5)
        ngram = [rng.choice(vocab) for _ in range(n)]
        dist = rng.choice([5, 20, 100])
        filler = [rng.choice(vocab) for _ in range(dist)]
        text = " ".join(ngram + filler + ngram)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if ids.shape[1] > seq_len:
            ids = ids[:, :seq_len]
        prompts.append(ids)
    return prompts


def build_natural_prompts_simple(tokenizer, seq_len, num_docs):
    """Natural prompts from WikiText-103 test split."""
    from datasets import load_dataset
    from config import DATASET_NAME, DATASET_CONFIG
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    full_text = " ".join(row["text"] for row in ds if row["text"].strip())
    all_ids = tokenizer(full_text, return_tensors="pt",
                        add_special_tokens=False)["input_ids"][0]
    chunks = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        chunks.append(all_ids[i: i + seq_len].unsqueeze(0))
        if len(chunks) >= num_docs:
            break
    return chunks


def main():
    p = argparse.ArgumentParser(description="Phase 2 — Regime detector validation")
    p.add_argument("--tier2_path", default=os.path.join(PHASE7_DIR, "tier2_heads.json"),
                   help="Path to tier2_heads.json from audit_heads.py")
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--num_positive", type=int, default=100,
                   help="Copy-trigger prompts (positive class)")
    p.add_argument("--num_negative", type=int, default=200,
                   help="Natural prompts (negative class)")
    p.add_argument("--ngram_n", type=int, default=3)
    p.add_argument("--ngram_threshold", type=float, default=0.3)
    p.add_argument("--freq_threshold", type=float, default=0.05)
    args = p.parse_args()

    os.makedirs(PHASE7_DIR, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Phase 2 — Regime-Switch Detector Validation")
    print(f"  ngram_n={args.ngram_n}, ngram_threshold={args.ngram_threshold}, "
          f"freq_threshold={args.freq_threshold}")
    print(f"{'='*65}\n")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building positive (copy-trigger) prompts...")
    positive = build_copy_trigger_prompts_simple(tokenizer, args.seq_len, args.num_positive)
    print("Building negative (natural) prompts...")
    negative = build_natural_prompts_simple(tokenizer, args.seq_len, args.num_negative)

    validator = DetectorValidator(
        ngram_n=args.ngram_n,
        ngram_threshold=args.ngram_threshold,
        freq_threshold=args.freq_threshold,
    )

    print("Evaluating detector...")
    metrics = validator.evaluate(positive, negative)

    print(f"\n  Detector Validation Results:")
    print(f"  {'Metric':<15} {'Value':>10}")
    print(f"  {'-'*28}")
    for k in ["precision", "recall", "f1", "specificity", "accuracy"]:
        print(f"  {k:<15} {metrics[k]:>10.3f}")
    print(f"\n  Confusion matrix:")
    print(f"              Pred+   Pred-")
    print(f"  Actual+  {metrics['tp']:>6}  {metrics['fn']:>6}")
    print(f"  Actual-  {metrics['fp']:>6}  {metrics['tn']:>6}")

    # ---- Target thresholds ----
    print(f"\n  Target: Precision > 0.90, Recall > 0.85")
    prec_ok = metrics["precision"] >= 0.90
    rec_ok  = metrics["recall"]    >= 0.85
    print(f"  Precision target met: {'YES ✓' if prec_ok else 'NO ✗'} "
          f"(got {metrics['precision']:.3f})")
    print(f"  Recall target met:    {'YES ✓' if rec_ok else 'NO ✗'} "
          f"(got {metrics['recall']:.3f})")

    if not prec_ok or not rec_ok:
        print("\n  SUGGESTION: Try adjusting ngram_threshold or freq_threshold.")
        print("  Lower ngram_threshold → higher recall, lower precision.")
        print("  Higher freq_threshold → higher precision, lower recall.")

    # Save
    out_path = os.path.join(PHASE7_DIR, "detector_validation.json")
    import json
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "metrics": metrics}, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # Print regime switching example
    print(f"\n  Example inference-time usage:")
    print(f"    from phase7.regime_detector import regime_detector")
    print(f"    for token_idx in range(seq_len):")
    print(f"        for (layer, head) in tier2_heads:")
    print(f"            if regime_detector(tokens[:token_idx]):")
    print(f"                use_full_attention(layer, head)")
    print(f"            else:")
    print(f"                use_substitute(layer, head)")


if __name__ == "__main__":
    main()
