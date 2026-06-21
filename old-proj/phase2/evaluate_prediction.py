# Measure how well our predicted prototype matches the actual attention behavior.
# Metric: recall@k — of the tokens the head actually attended to, how many did we predict?
# Uses validation split (never the same data as profiling).

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_articles
from config import (MODEL_NAME, DEVICE, USE_FP16, PHASE2_DIR,
                    PROTOTYPES_PATH, PREDICTION_TEST_DOCS,
                    RECALL_K_VALUES, MAX_SEQ_LEN, TOP_K_ATTENTION)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval().to(DEVICE)
    if USE_FP16:
        model.half()

    with open(PROTOTYPES_PATH, "rb") as f:
        prototypes = pickle.load(f)
    with open(os.path.join(PHASE2_DIR, "predictions.pkl"), "rb") as f:
        all_predictions = pickle.load(f)

    # Use validation split
    articles = load_articles(split="validation", max_articles=PREDICTION_TEST_DOCS)

    keys = sorted(prototypes.keys())
    recall_scores = {k: {kk: [] for kk in RECALL_K_VALUES} for k in keys}

    for doc_idx in tqdm(range(min(20, len(articles))), desc="Evaluating"):
        tokens = tokenizer(articles[doc_idx], return_tensors="pt",
                          truncation=True, max_length=MAX_SEQ_LEN)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        with torch.no_grad():
            output = model(**tokens, output_attentions=True)

        predictions = all_predictions[doc_idx]

        for layer_idx, layer_attn in enumerate(output.attentions):
            attn = layer_attn[0].float().cpu()
            for head_idx in range(attn.shape[0]):
                if (layer_idx, head_idx) not in prototypes:
                    continue

                head = attn[head_idx]
                avg_attn = head.mean(dim=0).numpy()

                pred_cluster = predictions.get((layer_idx, head_idx), 0)
                centroid = prototypes[(layer_idx, head_idx)]["centroids"][pred_cluster]

                actual_top = set(np.argsort(avg_attn)[-TOP_K_ATTENTION:])
                for k in RECALL_K_VALUES:
                    pred_important_dists = np.argsort(centroid)[-k * TOP_K_ATTENTION:]
                    pred_top = set(pred_important_dists[pred_important_dists < len(avg_attn)])
                    overlap = len(actual_top & pred_top)
                    recall = overlap / max(len(actual_top), 1)
                    recall_scores[(layer_idx, head_idx)][k].append(recall)

    print("\n=== Recall@k Results (averaged across docs) ===\n")
    print(f"{'Layer':>5} {'Head':>5}", end="")
    for k in RECALL_K_VALUES:
        print(f"  R@{k:d}", end="")
    print()

    for (layer, head) in sorted(keys)[:20]:
        print(f"{layer:5d} {head:5d}", end="")
        for k in RECALL_K_VALUES:
            vals = recall_scores[(layer, head)][k]
            avg = np.mean(vals) if vals else 0
            print(f"  {avg:.3f}", end="")
        print()

if __name__ == "__main__":
    main()
