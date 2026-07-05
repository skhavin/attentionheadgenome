import json, os
os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

with open("outputs/routing/Qwen2.5-0.5B_stability.json") as f:
    stab = json.load(f)
with open("outputs/phase2_atlas/Qwen2.5-0.5B_head_atlas.json") as f:
    atlas = json.load(f)

dia  = json.load(open("outputs/routing/Qwen2.5-0.5B_dialogue_geometry.json"))["heads"]
math = json.load(open("outputs/routing/Qwen2.5-0.5B_math_geometry.json"))["heads"]

print("Local heads that flip to Sink on dialogue or math (BOS mass jump):")
print(f"{'Head':<8} {'wiki_bos':>9} {'dial_bos':>9} {'math_bos':>9} {'L':>4}")
count = 0
for k, v in stab["routing_map"].items():
    if v["class"] == "Local" and v["routing"] == "FULL_SOFTMAX":
        wiki_bos = atlas["heads"][k]["attention_geometry"]["bos_mass"]
        dial_bos = dia.get(k, {}).get("bos_mass", 0.0)
        math_bos = math.get(k, {}).get("bos_mass", 0.0)
        if dial_bos > 0.4 or math_bos > 0.4:
            layer = atlas["heads"][k]["layer"]
            print(f"{k:<8} {wiki_bos:>9.3f} {dial_bos:>9.3f} {math_bos:>9.3f} {layer:>4}")
            count += 1
            if count >= 12:
                break
print(f"\nTotal ambiguous Local heads shown: {count}")
