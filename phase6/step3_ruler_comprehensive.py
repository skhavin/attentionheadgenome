# -*- coding: utf-8 -*-
# phase6/step3_ruler_comprehensive.py
#
# PURPOSE: Statistically validate HeadGenome sparse prefill capability preservation
# using a comprehensive RULER-style evaluation.
# 
# Experiment 1: 100 NIAH samples (Dense vs Sparse)
# Experiment 2: Varying needle position (depths 0.1, 0.25, 0.5, 0.75, 0.9)
# Experiment 3: Comparing W=256, W=384, W=512

import os, sys, json, random
import torch
import numpy as np
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

SENTENCES = [
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
    "Global markets rallied today following positive economic news.",
    "Deep learning models require significant computational resources.",
    "Space exploration has revealed thousands of exoplanets in our galaxy.",
    "Classical architecture emphasizes symmetry, proportion, and geometry.",
    "The recipe calls for a dash of cinnamon and a cup of sugar."
]

def generate_haystack(num_sentences=200):
    return " ".join(random.choices(SENTENCES, k=num_sentences))

def build_needle_test(depth=0.5, haystack_sentences=200, seed=42):
    random.seed(seed)
    
    # 30 unique needle templates
    needles = [
        ("The secret access code to the vault is {value}.", "What is the secret access code to the vault?", "{value}"),
        ("The hidden activation phrase is {value}.", "What is the hidden activation phrase?", "{value}"),
        ("The emergency override password is {value}.", "What is the emergency override password?", "{value}"),
        ("The magic word for the spell is {value}.", "What is the magic word for the spell?", "{value}"),
        ("The name of the mysterious artifact is {value}.", "What is the name of the mysterious artifact?", "{value}"),
        ("The coordinates to the hidden base are {value}.", "What are the coordinates to the hidden base?", "{value}"),
        ("The winning lottery numbers are {value}.", "What are the winning lottery numbers?", "{value}"),
        ("The combination to the safe is {value}.", "What is the combination to the safe?", "{value}"),
        ("The codename for the operation is {value}.", "What is the codename for the operation?", "{value}"),
        ("The missing ingredient is {value}.", "What is the missing ingredient?", "{value}")
    ]
    
    words = ["OMEGA-99", "CRIMSON", "774-ALPHA", "PHOENIX", "ZETA-42", "DRAGON", "ECLIPSE", "NOVA-11", "QUANTUM", "SILVER"]
    
    template = random.choice(needles)
    val = random.choice(words) + "-" + str(random.randint(1000, 9999))
    
    needle_text = template[0].replace("{value}", val)
    question = template[1]
    expected = template[2].replace("{value}", val)
    
    idx = int(haystack_sentences * depth)
    hay1 = generate_haystack(idx)
    hay2 = generate_haystack(haystack_sentences - idx)
    
    prompt = f"{hay1} {needle_text} {hay2}\n\nQuestion: {question}\nAnswer:"
    
    return prompt, expected

def eval_task(model, tok, device, prompt, expected):
    inputs = tok(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=15, pad_token_id=tok.eos_token_id)
        
    generated = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    
    expected_norm = expected.lower().replace("-", " ")
    gen_norm = generated.lower().replace("-", " ")
    hit = any(word in gen_norm for word in expected_norm.split() if len(word) > 3) or expected_norm in gen_norm
    
    return hit, prompt_len, generated

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device).eval()
    head_labels = load_labels()
    
    # Experiment settings
    num_samples = 20 # 20 per depth = 100 total
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    windows = [None, 512, 384, 256] # None = Dense
    
    print("\nGenerating 100 test prompts (20 at 5 different depths)...")
    tests = []
    for depth in depths:
        for i in range(num_samples):
            seed = int(depth * 1000) + i
            prompt, expected = build_needle_test(depth=depth, haystack_sentences=300, seed=seed)
            tests.append({"depth": depth, "prompt": prompt, "expected": expected, "id": i})
            
    print(f"Created {len(tests)} test prompts (each ~4000 tokens long).")

    final_results = {}

    for w in windows:
        w_str = "Dense" if w is None else f"W={w}"
        print(f"\n============================================================")
        print(f"EVALUATING: {w_str}")
        print(f"============================================================")
        
        hooks = None
        if w is not None:
            hooks = SparsePrefillMaskHooks(model, head_labels, w)
            
        success_by_depth = {d: 0 for d in depths}
        total_by_depth = {d: 0 for d in depths}
        
        for idx, test in enumerate(tests):
            hit, prompt_len, gen = eval_task(model, tok, device, test["prompt"], test["expected"])
            depth = test["depth"]
            
            if hit:
                success_by_depth[depth] += 1
            total_by_depth[depth] += 1
            
            if (idx + 1) % 10 == 0:
                print(f"  Progress: {idx+1}/{len(tests)}")
                
        if hooks:
            hooks.remove()
            
        total_success = sum(success_by_depth.values())
        total_tests = sum(total_by_depth.values())
        overall_acc = total_success / total_tests
        
        print(f"\n  Results for {w_str}:")
        print(f"    Overall Accuracy: {overall_acc*100:.1f}% ({total_success}/{total_tests})")
        for d in depths:
            acc = success_by_depth[d] / total_by_depth[d]
            print(f"    Depth {d:.2f}: {acc*100:.1f}%")
            
        final_results[w_str] = {
            "overall_accuracy": overall_acc,
            "success_by_depth": {str(d): success_by_depth[d]/total_by_depth[d] for d in depths}
        }
        
    out_path = os.path.join(OUT_DIR, "ruler_comprehensive.json")
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)
        
    print(f"\nSaved comprehensive results to {out_path}")

if __name__ == "__main__":
    main()
