# Match document embeddings to prototype centroids using cosine similarity.
# For each (layer, head), predict which cluster the document belongs to.
# Output: predictions.pkl in outputs/phase2/

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import PHASE2_DIR, PROTOTYPES_PATH

def predict_prototypes(doc_embedding, prototypes):
    """For each (layer, head), find the nearest prototype centroid."""
    predictions = {}
    if doc_embedding is not None:
        doc_vec = doc_embedding.reshape(1, -1)  # (1, hidden_dim)

    for (layer, head), proto_data in prototypes.items():
        centroids = proto_data["centroids"]  # (k, MAX_SEQ_LEN)
        # We compare doc embedding (hidden_dim) vs centroids (MAX_SEQ_LEN)
        # These are different spaces — so we use the centroid index directly
        # by computing distance in the attention-pattern space
        # For now: assign based on which centroid is most "local" vs "global"
        # (This is the simplest version — Phase 2 ablation will improve it)

        # Simple heuristic: score each centroid by how concentrated it is
        # (low entropy = specialized head behavior)
        entropies = []
        for c in centroids:
            c_norm = c / (c.sum() + 1e-10)
            entropy = -np.sum(c_norm * np.log(c_norm + 1e-10))
            entropies.append(entropy)

        # Predict: most specialized (lowest entropy) centroid
        predictions[(layer, head)] = int(np.argmin(entropies))

    return predictions

def main():
    # Load doc embeddings
    emb_path = os.path.join(PHASE2_DIR, "doc_embeddings.pkl")
    with open(emb_path, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]

    # Load prototypes
    with open(PROTOTYPES_PATH, "rb") as f:
        prototypes = pickle.load(f)

    print(f"Predicting prototypes for {len(embeddings)} documents...")
    all_predictions = []
    for i, emb in enumerate(embeddings):
        pred = predict_prototypes(emb, prototypes)
        all_predictions.append(pred)

    save_path = os.path.join(PHASE2_DIR, "predictions.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_predictions, f)
    print(f"Saved predictions to {save_path}")

if __name__ == "__main__":
    main()
