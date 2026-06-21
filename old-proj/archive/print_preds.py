import pandas as pd
import sys

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

print("SAMPLE RAW PREDICTIONS:")
task_df = df[df["task"].str.contains("niah")].head(5)
for i, row in task_df.iterrows():
    print(f"Task: {row['task']}")
    print(f"Answer: {row['answer']}")
    print(f"Predicted: {repr(row['predicted_answer'])}")
    print("-" * 50)
