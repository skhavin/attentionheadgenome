"""
headgenome.taxonomy
───────────────────
Assigns functional roles to attention heads via entropy-collapse probing.
Roles: sink | local | retrieval | induction
"""

from __future__ import annotations
import json, os, random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class HeadLabel:
    layer: int
    head: int
    role: str          # "sink" | "local" | "retrieval" | "induction"
    delta: float       # match_entropy - nonmatch_entropy
    match_entropy: Optional[float] = None
    nonmatch_entropy: Optional[float] = None
    relative_depth: float = 0.0


@dataclass
class TaxonomyResult:
    labels: Dict[Tuple[int, int], HeadLabel] = field(default_factory=dict)
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0

    # Role counts
    @property
    def sink(self):     return [h for h in self.labels.values() if h.role == "sink"]
    @property
    def local(self):    return [h for h in self.labels.values() if h.role == "local"]
    @property
    def retrieval(self):return [h for h in self.labels.values() if h.role == "retrieval"]
    @property
    def induction(self):return [h for h in self.labels.values() if h.role == "induction"]
    @property
    def critical(self): return self.retrieval + self.induction

    def role_of(self, layer: int, head: int) -> str:
        return self.labels.get((layer, head), HeadLabel(layer, head, "local", 0.0)).role

    def to_dict(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "counts": {
                "sink": len(self.sink),
                "local": len(self.local),
                "retrieval": len(self.retrieval),
                "induction": len(self.induction),
            },
            "heads": {
                f"{h.layer}_{h.head}": {
                    "role": h.role,
                    "delta": h.delta,
                    "match_entropy": h.match_entropy,
                    "nonmatch_entropy": h.nonmatch_entropy,
                    "relative_depth": h.relative_depth,
                }
                for h in self.labels.values()
            },
        }


PROBE_SENTENCES = [
    "The cat sat on the mat.",
    "Artificial intelligence is transforming the world.",
    "The quick brown fox jumps over the lazy dog.",
    "Deep learning requires massive computational resources.",
    "The city skyline looked beautiful at night.",
    "Science has advanced rapidly in the past century.",
    "Music brings people together from all walks of life.",
    "The ocean covers over seventy percent of the Earth.",
]

def _generate_probe_pairs(num_pairs: int, seq_len: int, tok) -> List[Tuple[str, str]]:
    """Generate (match_prompt, nonmatch_prompt) pairs for entropy-collapse probing."""
    pairs = []
    for _ in range(num_pairs):
        ctx = " ".join(random.choices(PROBE_SENTENCES, k=max(1, seq_len // 20)))
        trigger = "The secret number is 42."
        ctx_with = ctx + " " + trigger + " The secret number is"
        ctx_without = ctx + " The answer is"
        pairs.append((ctx_with, ctx_without))
    return pairs


def probe_model(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    num_pairs: int = 50,
    retrieval_threshold: float = 0.30,
    induction_threshold: float = -0.50,
    sink_entropy_threshold: float = 0.10,
    device: str = "cuda",
) -> TaxonomyResult:
    """
    Run entropy-collapse probing to assign roles to every head.

    Returns a TaxonomyResult with labels for every (layer, head) pair.
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_heads  = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)

    result = TaxonomyResult(num_layers=num_layers, num_heads=num_heads, num_kv_heads=num_kv_heads)

    pairs = _generate_probe_pairs(num_pairs, 128, tok)

    match_entropies    = torch.zeros(num_layers, num_heads)
    nonmatch_entropies = torch.zeros(num_layers, num_heads)
    count = 0

    attn_weights_store = {}

    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            # output may be (hidden, attn_weights, ...) depending on model
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                w = output[1].detach().float()  # (B, H, q, k)
                attn_weights_store[layer_idx] = w
        return hook

    # Try to capture attention weights via output hooks
    handles = []
    for li in range(num_layers):
        try:
            attn_mod = _get_attn_module(model, li)
            h = attn_mod.register_forward_hook(make_hook(li), with_kwargs=True)
            handles.append(h)
        except Exception:
            pass

    model.eval()
    with torch.no_grad():
        for match_prompt, nonmatch_prompt in pairs[:num_pairs]:
            for prompt, is_match in [(match_prompt, True), (nonmatch_prompt, False)]:
                try:
                    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
                    _ = model(**ids, output_attentions=True)
                    # Try to get from output_attentions
                except Exception:
                    pass

    for h in handles:
        h.remove()

    # Fallback: use output_attentions=True directly
    match_ents    = {li: [] for li in range(num_layers)}
    nonmatch_ents = {li: [] for li in range(num_layers)}

    with torch.no_grad():
        for match_prompt, nonmatch_prompt in pairs[:num_pairs]:
            for prompt, store in [(match_prompt, match_ents), (nonmatch_prompt, nonmatch_ents)]:
                try:
                    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
                    out = model(**ids, output_attentions=True)
                    if out.attentions is not None:
                        for li, attn in enumerate(out.attentions):
                            # attn: (B, H, q, k)
                            avg_over_q = attn[0].mean(dim=-2)  # (H, k)
                            # Entropy per head
                            eps = 1e-9
                            ent = -(avg_over_q * (avg_over_q + eps).log()).sum(dim=-1)  # (H,)
                            store[li].append(ent.cpu())
                except Exception:
                    continue

    for li in range(num_layers):
        if not match_ents[li]:
            continue
        m_stack = torch.stack(match_ents[li]).mean(0)    # (H,)
        nm_stack = torch.stack(nonmatch_ents[li]).mean(0) if nonmatch_ents[li] else m_stack.clone()

        for hi in range(num_heads):
            me  = m_stack[hi].item()  if hi < m_stack.shape[0]  else 0.0
            nme = nm_stack[hi].item() if hi < nm_stack.shape[0] else 0.0
            delta = me - nme

            # Classify
            if me < sink_entropy_threshold and nme < sink_entropy_threshold:
                role = "sink"
            elif delta > retrieval_threshold:
                role = "retrieval"
            elif delta < induction_threshold:
                role = "induction"
            else:
                role = "local"

            rel_depth = li / max(num_layers - 1, 1)
            result.labels[(li, hi)] = HeadLabel(
                layer=li, head=hi, role=role, delta=delta,
                match_entropy=me, nonmatch_entropy=nme,
                relative_depth=rel_depth,
            )

    # Fill any un-probed heads as "local"
    for li in range(num_layers):
        for hi in range(num_heads):
            if (li, hi) not in result.labels:
                rel_depth = li / max(num_layers - 1, 1)
                result.labels[(li, hi)] = HeadLabel(li, hi, "local", 0.0, relative_depth=rel_depth)

    return result


def _get_attn_module(model, layer_idx):
    """Navigate to the attention sub-module for a given layer."""
    # Try common patterns
    for path in ["model.layers", "transformer.h", "model.decoder.layers"]:
        try:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            layer = obj[layer_idx]
            for attr in ["self_attn", "attn", "attention", "self_attention"]:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
        except Exception:
            continue
    raise AttributeError(f"Cannot find attention module for layer {layer_idx}")


def load_taxonomy(path: str) -> TaxonomyResult:
    with open(path) as f:
        d = json.load(f)
    result = TaxonomyResult(
        num_layers=d["num_layers"],
        num_heads=d["num_heads"],
        num_kv_heads=d.get("num_kv_heads", d["num_heads"]),
    )
    for key, v in d["heads"].items():
        li, hi = map(int, key.split("_"))
        result.labels[(li, hi)] = HeadLabel(
            layer=li, head=hi,
            role=v["role"], delta=v["delta"],
            match_entropy=v.get("match_entropy"),
            nonmatch_entropy=v.get("nonmatch_entropy"),
            relative_depth=v.get("relative_depth", 0.0),
        )
    return result
