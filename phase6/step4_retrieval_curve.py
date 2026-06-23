# -*- coding: utf-8 -*-
# phase6/step4_retrieval_curve.py
#
# PURPOSE: Measure the Retrieval Contribution Curve to test the "Distributed Retrieval" hypothesis.
# We sort heads by their retrieval delta and preserve the Top K heads as dense, 
# while masking the rest to W=384. We measure NIAH accuracy to see if it scales with K.

import os, sys, json, random
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
SINK_SIZE  = 4
W_SIZE     = 384

def load_top_k_labels(K):
    if K == "dense":
        return None
        
    path = os.path.join(IN_DIR, LABEL_FILE)
    with open(path) as f:
        data = json.load(f)

    # Extract all heads and their deltas
    head_deltas = []
    if "heads" in data:
        for key, v in data["heads"].items():
            l, h = map(int, key.split("_"))
            delta = v.get("delta")
            if v.get("nan") or delta is None:
                delta = -999.0
            head_deltas.append(((l, h), delta))
            
    # Sort descending by delta
    head_deltas.sort(key=lambda x: x[1], reverse=True)
    
    # Take top K
    top_k_heads = set(hk for hk, _ in head_deltas[:K])
    
    labels = {}
    for l in range(NUM_LAYERS):
        for h in range(NUM_HEADS):
            if (l, h) in top_k_heads:
                labels[(l, h)] = "retrieval" # Full dense
            else:
                labels[(l, h)] = "local" # W=384
                
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

def build_needle_test(depth=0.5, haystack_sentences=350, seed=42):
    random.seed(seed)
    needles = [
        ("The secret access code to the vault is {value}.", "What is the secret access code to the vault?", "{value}"),
        ("The hidden activation phrase is {value}.", "What is the hidden activation phrase?", "{value}"),
        ("The emergency override password is {value}.", "What is the emergency override password?", "{value}"),
        ("The magic word for the spell is {value}.", "What is the magic word for the spell?", "{value}"),
        ("The name of the mysterious artifact is {value}.", "What is the name of the mysterious artifact?", "{value}")
    ]
    words = ["OMEGA", "CRIMSON", "ALPHA", "PHOENIX", "ZETA", "DRAGON", "ECLIPSE", "NOVA", "QUANTUM", "SILVER"]
    
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
        out = model.generate(
            **inputs, 
            max_new_tokens=20, 
            pad_token_id=tok.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None
        )
        
    generated = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    hit = expected.lower() in generated.lower()
    
    return {
        "hit": hit,
        "generated": generated,
        "expected": expected,
        "prompt_tokens": prompt_len,
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device).eval()
    
    num_samples = 10 
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    K_values = ["dense", 10, 20, 40, 80, 120]
    
    print("\nGenerating 50 test prompts (10 at 5 different depths)...")
    tests = []
    for depth in depths:
        for i in range(num_samples):
            seed = int(depth * 1000) + i + 999
            prompt, expected = build_needle_test(depth=depth, haystack_sentences=350, seed=seed)
            tests.append({"depth": depth, "prompt": prompt, "expected": expected, "id": f"d{depth}_i{i}"})
            
    final_results = {}

    for K in K_values:
        name = "Dense Baseline" if K == "dense" else f"Top {K} Retrieval Heads (Rest W=384)"
        print(f"\n============================================================")
        print(f"EVALUATING: {name}")
        print(f"============================================================")
        
        head_labels = load_top_k_labels(K)
        hooks = SparsePrefillMaskHooks(model, head_labels, W_SIZE) if K != "dense" else None
            
        success_by_depth = {d: 0 for d in depths}
        total_by_depth = {d: 0 for d in depths}
        
        results_log = []
        
        for idx, test in enumerate(tests):
            res = eval_task(model, tok, device, test["prompt"], test["expected"])
            depth = test["depth"]
            
            if res["hit"]:
                success_by_depth[depth] += 1
            total_by_depth[depth] += 1
            
            log_entry = {
                "id": test["id"],
                "depth": depth,
                "hit": res["hit"],
                "expected": res["expected"],
                "generated": res["generated"],
                "prompt_tokens": res["prompt_tokens"]
            }
            results_log.append(log_entry)
            
            if (idx + 1) % 10 == 0:
                print(f"  Progress: {idx+1}/{len(tests)} | Last prompt length: {res['prompt_tokens']} tokens")
                
        if hooks:
            hooks.remove()
            
        total_success = sum(success_by_depth.values())
        total_tests = sum(total_by_depth.values())
        overall_acc = total_success / total_tests
        
        print(f"\n  Results for {name}:")
        print(f"    Overall Accuracy: {overall_acc*100:.1f}% ({total_success}/{total_tests})")
        
        final_results[str(K)] = {
            "overall_accuracy": overall_acc,
            "success_by_depth": {str(d): success_by_depth[d]/total_by_depth[d] for d in depths},
            "logs": results_log
        }
        
    out_path = os.path.join(OUT_DIR, "retrieval_curve_synthetic_ruler.json")
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)
        
    print(f"\nSaved retrieval curve to {out_path}")

if __name__ == "__main__":
    main()
