# step1_weight_features.py
# NOTE: Uses only ASCII in print() to avoid Windows cp1252 UnicodeEncodeError.
# PURPOSE: Extract architecture-agnostic weight features from each attention head,
#          pair them with Phase 1 functional labels, and train/evaluate a classifier.
#
# FEATURES:
#   1. SVD top-16 normalized singular values of W_q @ W_k.T
#   2. Mean row entropy of softmax(W_q @ W_k.T)
#   3. Diagonal vs. off-diagonal magnitude ratio of W_q @ W_k.T
#   4. Frobenius norm ratio ||W_v|| / ||W_q||
#   5. Frobenius norm ratio ||W_q|| / ||W_k||
#   6. Relative depth (optional feature)
#
# OUTPUTS:
#   outputs/phase3/weight_features.json

import os
import sys

# Set cache directories BEFORE importing transformers
os.environ["HF_HOME"] = "d:\\.cache\\huggingface"

import json
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from transformers import AutoConfig, AutoModelForCausalLM

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "outputs", "phase1")
OUT_DIR = os.path.join(ROOT, "outputs", "phase3")

MODEL_SLUGS = {
    "GPT-2":        "gpt2-medium",
    "Qwen-0.5B":    "qwen2.5-0.5b",
    "Qwen-1.5B":    "qwen2.5-1.5b",
    "Llama-3.2-1B": "llama-3.2-1b",
}

HF_IDS = {
    "GPT-2":        "gpt2-medium",
    "Qwen-0.5B":    "Qwen/Qwen2.5-0.5B",
    "Qwen-1.5B":    "Qwen/Qwen2.5-1.5B",
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B",
}

K_CLUSTERS = 4


def map_cluster_roles(centroids):
    """
    Map each of the 4 cluster indices to a semantic role:
    sink, local, retrieval, induction.
    Uses std and sink_mass to resolve overlaps.
    """
    n = centroids.shape[0]
    stds = [float(c.std()) for c in centroids]
    sink_masses = [float(c[0:4].sum()) for c in centroids]
    
    # 1. Induction has the absolute lowest std (flattest distribution)
    induction_idx = int(np.argmin(stds))
    
    # 2. Sink has the highest sink mass (early positions)
    sink_idx = int(np.argmax(sink_masses))
    if sink_idx == induction_idx:
        # Fallback: pick the second highest sink mass
        sorted_sink_indices = np.argsort(sink_masses)[::-1]
        sink_idx = int(sorted_sink_indices[1])
        
    # 3. Of the remaining two:
    #    - The one with higher std is retrieval
    #    - The one with lower std is local
    remaining = [i for i in range(n) if i not in (induction_idx, sink_idx)]
    if stds[remaining[0]] > stds[remaining[1]]:
        retrieval_idx = remaining[0]
        local_idx = remaining[1]
    else:
        retrieval_idx = remaining[1]
        local_idx = remaining[0]
        
    return {
        sink_idx: "sink",
        local_idx: "local",
        retrieval_idx: "retrieval",
        induction_idx: "induction"
    }


def get_functional_labels(slug):
    """Load summary histograms and assign semantic labels via k=4 KMeans."""
    json_path = os.path.join(IN_DIR, f"{slug}_patterns_summary.json")
    if not os.path.exists(json_path):
        return None

    with open(json_path) as f:
        data = json.load(f)

    heads = {}
    for key, hist in data["heads"].items():
        layer, head = map(int, key.split("_"))
        heads[(layer, head)] = np.array(hist, dtype=np.float32)

    keys = sorted(heads.keys())
    X = np.array([heads[k] for k in keys])

    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    role_map = map_cluster_roles(centroids)
    return {k: role_map[labels[i]] for i, k in enumerate(keys)}


def extract_weight_features(model, model_type):
    """
    Extract architecture-agnostic weight features for each head.
    Returns dict: (layer, head) -> feature_dict
    """
    features = {}

    if model_type == "GPT-2":
        # GPT-2 Medium parameters
        config = model.config
        num_layers = config.n_layer
        num_heads = config.n_head
        d_model = config.n_embd
        head_dim = d_model // num_heads

        for layer_idx in range(num_layers):
            attn_layer = model.transformer.h[layer_idx].attn
            # W is of shape (d_model, 3 * d_model)
            W = attn_layer.c_attn.weight.detach().cpu()
            W_q, W_k, W_v = torch.split(W, d_model, dim=-1)

            for head_idx in range(num_heads):
                W_q_h = W_q[:, head_idx * head_dim : (head_idx + 1) * head_dim].T
                W_k_h = W_k[:, head_idx * head_dim : (head_idx + 1) * head_dim].T
                W_v_h = W_v[:, head_idx * head_dim : (head_idx + 1) * head_dim].T
                features[(layer_idx, head_idx)] = compute_head_metrics(W_q_h, W_k_h, W_v_h)

    else:
        # Llama / Qwen self-attention linear layers
        # q_proj.weight: (num_heads * head_dim, d_model)
        # k_proj.weight, v_proj.weight: (num_kv_heads * head_dim, d_model)
        config = model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        d_model = config.hidden_size
        head_dim = d_model // num_heads
        g = num_heads // num_kv_heads

        for layer_idx in range(num_layers):
            self_attn = model.model.layers[layer_idx].self_attn
            W_q = self_attn.q_proj.weight.detach().cpu()
            W_k = self_attn.k_proj.weight.detach().cpu()
            W_v = self_attn.v_proj.weight.detach().cpu()

            for head_idx in range(num_heads):
                kv_head_idx = head_idx // g
                W_q_h = W_q[head_idx * head_dim : (head_idx + 1) * head_dim, :]
                W_k_h = W_k[kv_head_idx * head_dim : (kv_head_idx + 1) * head_dim, :]
                W_v_h = W_v[kv_head_idx * head_dim : (kv_head_idx + 1) * head_dim, :]
                features[(layer_idx, head_idx)] = compute_head_metrics(W_q_h, W_k_h, W_v_h)

    return features


def compute_head_metrics(W_q_h, W_k_h, W_v_h):
    """Compute SVD and norm features from head projection matrices."""
    # Q_K = W_q_h @ W_k_h.T  (shape: head_dim, head_dim)
    Q_K = torch.matmul(W_q_h, W_k_h.T).float()
    head_dim = Q_K.shape[0]

    # 1. SVD top-16 normalized singular values
    U, S, V = torch.linalg.svd(Q_K)
    S_np = S.numpy()
    if len(S_np) < 16:
        # pad with zeros
        S_16 = np.zeros(16)
        S_16[:len(S_np)] = S_np
    else:
        S_16 = S_np[:16]
    # Normalize S_16
    sum_s = S_16.sum()
    if sum_s > 0:
        S_16 /= sum_s

    # 2. Mean row entropy of softmax(Q_K)
    p = torch.softmax(Q_K, dim=-1)
    entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1).mean().item()

    # 3. Diagonal vs off-diagonal magnitude ratio
    diag = torch.diagonal(Q_K)
    diag_mean = torch.abs(diag).mean().item()
    off_diag_mask = ~torch.eye(head_dim, dtype=torch.bool)
    off_diag_mean = torch.abs(Q_K[off_diag_mask]).mean().item()
    ratio = diag_mean / (off_diag_mean + 1e-12)

    # 4. Frobenius norms
    norm_q = torch.linalg.norm(W_q_h).item()
    norm_k = torch.linalg.norm(W_k_h).item()
    norm_v = torch.linalg.norm(W_v_h).item()

    v_q_ratio = norm_v / (norm_q + 1e-12)
    q_k_ratio = norm_q / (norm_k + 1e-12)

    # Combine into a feature dict
    feats = {
        "entropy":        round(entropy, 6),
        "diag_off_ratio": round(ratio, 6),
        "v_q_ratio":      round(v_q_ratio, 6),
        "q_k_ratio":      round(q_k_ratio, 6),
    }
    for idx, s_val in enumerate(S_16):
        feats[f"svd_{idx}"] = round(float(s_val), 6)

    return feats


def run_cross_validation(data_by_model):
    """
    Evaluate prediction accuracy using Leave-One-Model-Out validation.
    Also tests with and without relative depth feature.
    """
    models = list(data_by_model.keys())
    role_to_id = {"sink": 0, "local": 1, "retrieval": 2, "induction": 3}

    results = {}

    for use_depth in [False, True]:
        tag = "with_depth" if use_depth else "weights_only"
        results[tag] = {}

        print(f"\nEvaluating Random Forest Classifier ({tag}):")

        model_accuracies = []

        for test_model in models:
            # Split train / test
            train_X, train_y = [], []
            test_X, test_y = [], []

            for model_name, heads in data_by_model.items():
                for (layer, head), head_data in heads.items():
                    # Build feature vector
                    feat_dict = head_data["features"]
                    label_str = head_data["label"]

                    feat_vec = [
                        feat_dict["entropy"],
                        feat_dict["diag_off_ratio"],
                        feat_dict["v_q_ratio"],
                        feat_dict["q_k_ratio"]
                    ] + [feat_dict[f"svd_{i}"] for i in range(16)]

                    if use_depth:
                        # Append relative depth
                        total_layers = max(k[0] for k in heads.keys()) + 1
                        rel_depth = layer / (total_layers - 1)
                        feat_vec.append(rel_depth)

                    if model_name == test_model:
                        test_X.append(feat_vec)
                        test_y.append(role_to_id[label_str])
                    else:
                        train_X.append(feat_vec)
                        train_y.append(role_to_id[label_str])

            # Train classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(train_X, train_y)

            # Predict
            preds = clf.predict(test_X)
            acc = accuracy_score(test_y, preds)
            model_accuracies.append(acc)
            results[tag][test_model] = round(float(acc), 4)
            print(f"  Test Model: {test_model:<15} Accuracy: {acc:.4f}")

        results[tag]["average"] = round(float(np.mean(model_accuracies)), 4)
        print(f"  Average Accuracy: {results[tag]['average']:.4f}")

    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Extract features and load labels ───────────────────────────────────
    all_data = {}

    for model_name, slug in MODEL_SLUGS.items():
        labels = get_functional_labels(slug)
        if labels is None:
            print(f"[SKIP] Labels missing for {model_name} ({slug})")
            continue

        model_id = HF_IDS[model_name]
        print(f"\nLoading model {model_name} ({model_id}) on CPU to extract weight features...")
        # Load CPU only to save VRAM and handle different model sizes
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )

        features = extract_weight_features(model, model_name)

        model_data = {}
        for k in labels.keys():
            if k in features:
                model_data[k] = {
                    "label":    labels[k],
                    "features": features[k]
                }

        all_data[model_name] = model_data
        print(f"  Extracted features and labels for {len(model_data)} heads.")

        # Free memory
        del model

    # ── 2. Run Leave-One-Model-Out evaluation ─────────────────────────────────
    eval_results = run_cross_validation(all_data)

    # Convert keys in all_data to strings for json serialization
    serializable_data = {}
    for model_name, heads in all_data.items():
        serializable_data[model_name] = {
            f"{layer}_{head}": val
            for (layer, head), val in heads.items()
        }

    output = {
        "evaluation": eval_results,
        "models":     serializable_data,
    }

    # Save to outputs/phase3/weight_features.json
    out_json = os.path.join(OUT_DIR, "weight_features.json")
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved -> {out_json}")
    print("[DONE] Weight-based functional prediction complete.")


if __name__ == "__main__":
    main()
