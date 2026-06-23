# -*- coding: utf-8 -*-
# phase6/step2_ruler_niah.py
#
# PURPOSE: Validate that HeadGenome Sparse Prefill preserves long-context
# retrieval capabilities using RULER-style tasks (NIAH, Multi-Key, Variable Tracking).

import os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"]          = r"d:\.cache\huggingface"
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "outputs", "phase1")
OUT_DIR = os.path.join(ROOT, "outputs", "phase6")

MODEL_ID   = "Qwen/Qwen2.5-1.5B"
LABEL_FILE = "qwen1.5b_retrieval_entropy.json"
NUM_HEADS  = 12
NUM_LAYERS = 28
ATTN_PATH  = "model.layers"
ATTN_ATTR  = "self_attn"
W_SIZE     = 384
SINK_SIZE  = 4

THRESHOLD_RETRIEVAL = 0.10
THRESHOLD_INDUCTION = -0.50
THRESHOLD_SINK_ENT  = 0.10

def load_labels():
    path = os.path.join(IN_DIR, LABEL_FILE)
    with open(path) as f:
        data = json.load(f)

    labels = {}
    if "heads" in data:
        for key, v in data["heads"].items():
            l, h = map(int, key.split("_"))
            delta, me, nme = v.get("delta"), v.get("match_entropy"), v.get("nonmatch_entropy")
            if v.get("nan") or delta is None or me is None:
                role = "sink"
            elif me < THRESHOLD_SINK_ENT and nme < THRESHOLD_SINK_ENT:
                role = "sink"
            elif delta > THRESHOLD_RETRIEVAL:
                role = "retrieval"
            elif delta < THRESHOLD_INDUCTION:
                role = "induction"
            else:
                role = "local"
            labels[(l, h)] = role
    return labels

def get_attn_module(model, layer_idx):
    obj = model
    for part in ATTN_PATH.split("."):
        obj = getattr(obj, part)
    return getattr(obj[layer_idx], ATTN_ATTR)

class SparsePrefillMaskHooks:
    def __init__(self, model, head_labels, W):
        self.handles = []
        self.W = W
        self.sink_size = SINK_SIZE

        for layer_idx in range(NUM_LAYERS):
            attn_module = get_attn_module(model, layer_idx)
            layer_roles = {h: head_labels.get((layer_idx, h), "local") for h in range(NUM_HEADS)}
            handle = attn_module.register_forward_pre_hook(
                self._make_pre_hook(layer_roles),
                with_kwargs=True
            )
            self.handles.append(handle)

    def _make_pre_hook(self, layer_roles):
        W = self.W
        sink_size = self.sink_size

        def pre_hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None: return args, kwargs

            q_len = hidden_states.shape[1]
            device = hidden_states.device
            dtype = hidden_states.dtype

            cache_pos = kwargs.get("cache_position")
            if cache_pos is not None:
                q_pos = cache_pos.unsqueeze(1)
                kv_len = cache_pos[-1].item() + 1
                k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
            else:
                q_pos = torch.arange(q_len, device=device).unsqueeze(1)
                kv_len = q_len
                k_pos = torch.arange(kv_len, device=device).unsqueeze(0)

            causal_allow = q_pos >= k_pos
            allow_local = ((k_pos < sink_size) | ((q_pos - k_pos) < W)) & causal_allow
            allow_full  = causal_allow

            zero_t = torch.tensor(0.0, dtype=dtype, device=device)
            inf_t  = torch.tensor(float("-inf"), dtype=dtype, device=device)

            local_mask = torch.where(allow_local, zero_t, inf_t)
            full_mask  = torch.where(allow_full, zero_t, inf_t)

            role_mask = torch.empty(1, NUM_HEADS, q_len, kv_len, dtype=dtype, device=device)
            for h in range(NUM_HEADS):
                role = layer_roles.get(h, "local")
                if role in ["sink", "local"]:
                    role_mask[0, h] = local_mask
                else:
                    role_mask[0, h] = full_mask

            existing = kwargs.get("attention_mask")
            if existing is not None:
                try:
                    kwargs = dict(kwargs, attention_mask=existing + role_mask)
                except Exception:
                    kwargs = dict(kwargs, attention_mask=role_mask)
            else:
                kwargs = dict(kwargs, attention_mask=role_mask)

            return args, kwargs
        return pre_hook

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

# ── RULER Task Generators ──────────────────────────────────────────────────
import random

def generate_haystack(num_sentences=200):
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the way we work and live.",
        "The weather today is surprisingly sunny despite the forecast.",
        "I enjoy reading science fiction novels on rainy afternoons.",
        "A healthy diet consists of fruits, vegetables, and lean proteins.",
        "The city skyline looks beautiful at night when all the lights are on.",
        "Many people find programming to be both challenging and rewarding.",
        "Music has the power to evoke deep emotions and bring people together.",
        "Exploring the outdoors is a great way to relieve stress.",
        "The history of ancient civilizations continues to fascinate historians.",
    ]
    haystack = " ".join(random.choices(sentences, k=num_sentences))
    return haystack

def build_niah_prompt(haystack_len=200):
    haystack1 = generate_haystack(haystack_len // 2)
    haystack2 = generate_haystack(haystack_len // 2)
    needle = "The secret access code to the vault is 8841-OMEGA-99."
    prompt = f"{haystack1} {needle} {haystack2}\n\nQuestion: What is the secret access code to the vault?\nAnswer:"
    return prompt, "8841-OMEGA-99"

def build_multikey_prompt(haystack_len=200):
    haystack1 = generate_haystack(haystack_len // 3)
    haystack2 = generate_haystack(haystack_len // 3)
    haystack3 = generate_haystack(haystack_len // 3)
    needle1 = "The first ingredient for the potion is crushed moonstone."
    needle2 = "The second ingredient for the potion is dragon scale."
    prompt = f"{haystack1} {needle1} {haystack2} {needle2} {haystack3}\n\nQuestion: What are the two ingredients for the potion?\nAnswer:"
    return prompt, "crushed moonstone and dragon scale"

def build_vt_prompt(haystack_len=200):
    haystack1 = generate_haystack(haystack_len // 4)
    haystack2 = generate_haystack(haystack_len // 4)
    haystack3 = generate_haystack(haystack_len // 4)
    haystack4 = generate_haystack(haystack_len // 4)
    var1 = "Target_Alpha is located in sector 4."
    var2 = "Target_Alpha has moved to sector 7."
    var3 = "Target_Alpha is now in sector 9."
    prompt = f"{var1} {haystack1} {var2} {haystack2} {haystack3} {var3} {haystack4}\n\nQuestion: What sector is Target_Alpha currently located in?\nAnswer:"
    return prompt, "sector 9"

def eval_task(model, tok, device, name, prompt, expected):
    inputs = tok(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=15, pad_token_id=tok.eos_token_id)
        
    generated = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    
    # Simple soft match
    expected_norm = expected.lower().replace("-", " ")
    gen_norm = generated.lower().replace("-", " ")
    hit = any(word in gen_norm for word in expected_norm.split() if len(word) > 3) or expected_norm in gen_norm
    
    print(f"  Task: {name} ({prompt_len} tokens)")
    print(f"    Expected: {expected}")
    print(f"    Generated: {generated}")
    print(f"    Pass: {hit}")
    return hit

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device).eval()
    
    head_labels = load_labels()
    
    # Build tasks (around ~3000 tokens)
    tasks = [
        ("NIAH", *build_niah_prompt(400)),
        ("Multi-Key", *build_multikey_prompt(400)),
        ("Variable Tracking", *build_vt_prompt(400))
    ]
    
    results = {}
    
    print("\n" + "="*60)
    print("BASELINE: Dense Full Attention")
    print("="*60)
    base_hits = 0
    for name, prompt, expected in tasks:
        if eval_task(model, tok, device, name, prompt, expected):
            base_hits += 1
    results["dense"] = {"score": base_hits, "total": len(tasks)}
            
    print("\n" + "="*60)
    print(f"HEADGENOME: Sparse Prefill (W={W_SIZE})")
    print("="*60)
    hooks = SparsePrefillMaskHooks(model, head_labels, W_SIZE)
    sparse_hits = 0
    for name, prompt, expected in tasks:
        if eval_task(model, tok, device, name, prompt, expected):
            sparse_hits += 1
    hooks.remove()
    results["sparse"] = {"score": sparse_hits, "total": len(tasks)}
    
    out_path = os.path.join(OUT_DIR, "ruler_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*60)
    print("RULER RESULTS SUMMARY")
    print(f"  Dense Attention: {base_hits}/{len(tasks)}")
    print(f"  Sparse Prefill:  {sparse_hits}/{len(tasks)}")
    print("="*60)

if __name__ == "__main__":
    main()
