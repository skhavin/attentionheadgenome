import pandas as pd
import re
import sys
import ast

predictions_file = sys.argv[1]
try:
    df = pd.read_csv(predictions_file)
except Exception as e:
    print(f"Error reading predictions: {e}")
    sys.exit(1)

# Filter for evaluated samples
df = df[df["predicted_answer"].notna()]

if len(df) == 0:
    print("No predictions generated yet.")
    sys.exit(0)

# Parse standard list strings in df['answer']
def parse_answer(ans):
    if isinstance(ans, str):
        ans = ans.strip("[]")
        parts = re.findall(r"['\"](.*?)['\"]", ans)
        if not parts:
            parts = ans.split()
        return [p.strip() for p in parts if p.strip()]
    return ans

df["answer"] = df["answer"].apply(parse_answer)

# Compute metrics
sys.path.insert(0, "/teamspace/studios/this_studio/kvpress/evaluation")
from benchmarks.ruler.calculate_metrics import calculate_metrics
metrics = calculate_metrics(df)

print(f"Intermediate score based on {len(df)} completed samples:")
for task, score in metrics.items():
    print(f"  - {task}: {score['string_match']}%")
