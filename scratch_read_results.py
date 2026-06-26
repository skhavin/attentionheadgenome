import json

files = {
    "GPT-2 Medium": "outputs/phase8_paper_suite/regime_switching_gpt2-medium.json",
    "Qwen-0.5B": "outputs/phase8_paper_suite/regime_switching_Qwen_Qwen2.5-0.5B.json",
    "Qwen-1.5B": "outputs/phase8_paper_suite/regime_switching_Qwen_Qwen2.5-1.5B.json",
    "Llama-3.2-1B": "outputs/phase8_paper_suite/regime_switching_unsloth_Llama-3.2-1B.json",
}

for model, path in files.items():
    with open(path) as f:
        d = json.load(f)
    print(f"\n=== {model} ===")
    print("Top 5 Regime-Switchers:")
    for h in d["top_10_regime_switchers"][:5]:
        layer = h["layer"]
        head = h["head"]
        var = h["variance"]
        gm = h["group_means"]
        print(f"  L{layer}H{head}  var={var:.4f}  {gm}")
    print("Top 3 Most Stable:")
    for h in d["top_10_most_stable"][:3]:
        layer = h["layer"]
        head = h["head"]
        var = h["variance"]
        gm = h["group_means"]
        print(f"  L{layer}H{head}  var={var:.6f}  {gm}")
    top_switch_var = d["top_10_regime_switchers"][0]["variance"]
    top_stable_var = d["top_10_most_stable"][0]["variance"]
    print(f"  Switcher/Stable variance ratio: {top_switch_var:.4f} / {top_stable_var:.6f} = {top_switch_var/max(top_stable_var,1e-9):.0f}x")
