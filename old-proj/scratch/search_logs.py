import json

path = "C:/Users/KHAVIN S/.gemini/antigravity-ide/brain/fa2a53ca-d90b-4620-8bb5-2f64b3457935/.system_generated/logs/transcript.jsonl"
with open(path, errors="replace") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            content = data.get("content", "")
            if "HEAD AUDIT SUMMARY" in content or "Tier 1 (" in content:
                print(f"Step {data.get('step_index')}:")
                # print lines containing the summary
                summary_lines = [l for l in content.split("\n") if "Tier" in l or "Total" in l or "SUMMARY" in l]
                print("\n".join(summary_lines))
                print("-" * 50)
        except Exception as e:
            pass
