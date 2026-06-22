import os, json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

IN_DIR = 'outputs/phase1'
MODEL_SLUGS = {
    'GPT-2': 'gpt2-medium',
    'Qwen-0.5B': 'qwen2.5-0.5b',
    'Qwen-1.5B': 'qwen2.5-1.5b',
    'Llama-3.2-1B': 'llama-3.2-1b',
}
K_CLUSTERS = 4

for model_name, slug in MODEL_SLUGS.items():
    p = os.path.join(IN_DIR, slug + '_patterns_summary.json')
    if not os.path.exists(p):
        print('Missing: ' + p)
        continue
    with open(p) as f:
        data = json.load(f)
    heads = {k: np.array(v) for k, v in data['heads'].items()}
    keys = sorted(heads.keys())
    X = np.array([heads[k] for k in keys])
    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)

    stds  = [round(float(c.std()), 4) for c in km.cluster_centers_]
    sinks = [round(float(c[0:4].sum()), 4) for c in km.cluster_centers_]
    locs  = [round(float(c[1:10].sum()), 4) for c in km.cluster_centers_]
    sizes = [int((labels == i).sum()) for i in range(K_CLUSTERS)]

    print('=== ' + model_name + ' (silhouette=' + str(round(sil, 4)) + ') ===')
    print('  Centroid stds:       ' + str(stds))
    print('  Centroid sink_mass:  ' + str(sinks))
    print('  Centroid local_mass: ' + str(locs))
    print('  Cluster sizes:       ' + str(sizes))
    print()
    for i, c in enumerate(km.cluster_centers_):
        print('  Centroid ' + str(i) + ': std=' + str(round(float(c.std()), 4)) +
              ' sink=' + str(round(float(c[0:4].sum()), 4)) +
              ' local=' + str(round(float(c[1:10].sum()), 4)) +
              ' n=' + str(sizes[i]))
        print('    first 30: ' + str(c[:30].round(4)))
    print()
