"""
step20_universal_profiler.py
-----------------------------
Collects the same attention geometry features (bos_mass, local_mass,
long_range_mass, mean_distance) across 4 diverse text domains.

This is the universality test: do heads behave the same way regardless
of whether the text is Wikipedia, code, dialogue, or math?
"""
import json, os, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "d:\\.cache\\huggingface"
os.makedirs("outputs/routing", exist_ok=True)

MODEL = "Qwen/Qwen2.5-0.5B"
SAFE_MODEL = MODEL.split("/")[-1]
N_PROMPTS = 50  # per domain, same as step1

print(f"Loading {MODEL}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tok   = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(device)
model.eval()

L = model.config.num_hidden_layers
H = model.config.num_attention_heads

def get_texts(domain):
    """Load N_PROMPTS text samples for a given domain."""
    if domain == "wikipedia":
        # Use our existing wikitext dataset
        with open("outputs/phase2_atlas/dataset.json") as f:
            d = json.load(f)
        return [s["text"] for s in d["wikitext"][:N_PROMPTS]]
    
    elif domain == "code":
        # Use starcoder2-self-instruct which is Parquet-native
        ds = load_dataset("bigcode/self-oss-instruct-sc2-exec-filter-50k",
                          split="train", streaming=True)
        texts = []
        for row in ds:
            t = row.get("response", "").strip()
            if len(t.split()) > 30:
                texts.append(t[:1000])
            if len(texts) >= N_PROMPTS:
                break
        return texts
    
    elif domain == "dialogue":
        # Use HuggingFaceH4/ultrachat_200k which is Parquet-native
        ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                          split="train_sft", streaming=True)
        texts = []
        for row in ds:
            # Extract first user message
            msgs = row.get("messages", [])
            joined = " ".join(m["content"] for m in msgs[:4] if m.get("content"))
            if len(joined.split()) > 20:
                texts.append(joined[:800])
            if len(texts) >= N_PROMPTS:
                break
        return texts
    
    elif domain == "math":
        ds = load_dataset("gsm8k", "main", split="train")
        texts = [row["question"] for row in ds.select(range(N_PROMPTS))]
        return texts
    
    return []

def profile_domain(texts, domain_name):
    acc = {(l, h): {"bos": [], "local": [], "long": [], "dist": []}
           for l in range(L) for h in range(H)}
    
    valid = 0
    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
        ids = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        T = ids["input_ids"].shape[1]
        if T < 8:
            continue
        
        with torch.no_grad():
            out = model(**ids, output_attentions=True)
        
        for l, attn in enumerate(out.attentions):
            a = attn[0].float().cpu().numpy()  # (H, T, T)
            for h in range(H):
                row = a[h, -1, :]
                t   = len(row) - 1
                positions = np.arange(T)
                dists = t - positions
                
                bos_m   = float(row[0])
                local_m = float(row[max(0, t-4):t].sum())
                long_m  = float(row[:max(0, t-32)].sum()) if t > 32 else 0.0
                mean_d  = float((row * dists).sum())
                
                acc[(l, h)]["bos"].append(bos_m)
                acc[(l, h)]["local"].append(local_m)
                acc[(l, h)]["long"].append(long_m)
                acc[(l, h)]["dist"].append(mean_d)
        valid += 1
        
    print(f"  {domain_name}: {valid} valid prompts processed.")
    
    results = {}
    for (l, h), v in acc.items():
        if not v["bos"]:
            continue
        results[f"{l}_{h}"] = {
            "mean_distance":   round(float(np.mean(v["dist"])), 4),
            "bos_mass":        round(float(np.mean(v["bos"])),  4),
            "local_mass":      round(float(np.mean(v["local"])), 4),
            "long_range_mass": round(float(np.mean(v["long"])), 4),
        }
    return results

DOMAINS = ["wikipedia", "code", "dialogue", "math"]

for domain in DOMAINS:
    out_path = f"outputs/routing/{SAFE_MODEL}_{domain}_geometry.json"
    if os.path.exists(out_path):
        print(f"\n--- Domain: {domain} --- (already done, skipping)")
        continue
    print(f"\n--- Domain: {domain} ---")
    try:
        texts = get_texts(domain)
        print(f"  Loaded {len(texts)} texts.")
        geometry = profile_domain(texts, domain)
        with open(out_path, "w") as f:
            json.dump({"model": SAFE_MODEL, "domain": domain, "heads": geometry}, f, indent=2)
        print(f"  Saved to {out_path}")
    except Exception as e:
        print(f"  ERROR loading {domain}: {e}")
