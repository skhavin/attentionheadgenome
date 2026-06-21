import pandas as pd
import re
import sys
import ast

predictions_file = "/teamspace/studios/this_studio/kvpress/evaluation/results/ruler__4096__meta-llama--Meta-Llama-3.1-8B-Instruct__proactive_cache__0.75/1/predictions.csv"
df = pd.read_csv(predictions_file)
df = df[df["predicted_answer"].notna()]
print(f"Total completed predictions: {len(df)}")
print(f"Tasks present: {df['task'].unique().tolist()}\n")

# -------------------------------------------------------
# 1. Show raw examples for niah_single_1
# -------------------------------------------------------
print("=" * 70)
print("SAMPLE RAW PREDICTIONS: niah_single_1 (first 5 rows)")
print("=" * 70)
task_df = df[df["task"] == "niah_single_1"].head(5)
for i, row in task_df.iterrows():
    print(f"  Answer column raw  : {repr(row['answer'])}")
    print(f"  Predicted raw      : {repr(row['predicted_answer'])}")
    # Check if answer is in prediction
    try:
        ans = row["answer"]
        if isinstance(ans, str):
            parts = re.findall(r"['\"](.*?)['\"]", ans.strip("[]"))
            if not parts:
                parts = ans.strip("[]").split()
        else:
            parts = ans
        match = any(p.lower() in str(row["predicted_answer"]).lower() for p in parts)
        print(f"  Match              : {match}")
        print(f"  Parsed answers     : {parts}")
    except Exception as e:
        print(f"  Parse error: {e}")
    print()

# -------------------------------------------------------
# 2. Show raw examples for vt (variable tracking - best scorer)
# -------------------------------------------------------
print("=" * 70)
print("SAMPLE RAW PREDICTIONS: vt (Variable Tracking, first 5 rows)")
print("=" * 70)
task_df = df[df["task"] == "vt"].head(5)
for i, row in task_df.iterrows():
    print(f"  Answer column raw  : {repr(row['answer'])}")
    print(f"  Predicted raw      : {repr(row['predicted_answer'])}")
    print()

# -------------------------------------------------------
# 3. Check: does NoCompression baseline also score low?
# Run the SAME tasks with a direct greedy check
# -------------------------------------------------------
print("=" * 70)
print("SCORE BREAKDOWN PER TASK (string_match_all logic)")
print("=" * 70)

def parse_answer(ans):
    if isinstance(ans, str):
        ans_stripped = ans.strip("[]")
        parts = re.findall(r"['\"](.*?)['\"]", ans_stripped)
        if not parts:
            parts = [x.strip() for x in ans_stripped.split() if x.strip()]
        return parts
    return ans

df["answer_parsed"] = df["answer"].apply(parse_answer)

for task, gdf in df.groupby("task"):
    total = len(gdf)
    matched = 0
    for _, row in gdf.iterrows():
        refs = row["answer_parsed"]
        pred = str(row["predicted_answer"]).lower()
        if not refs:
            continue
        hit = sum(1 for r in refs if r.lower() in pred) / len(refs)
        matched += hit
    score = round(matched / total * 100, 2)
    print(f"  {task:<25}: {score}% ({total} samples)")
