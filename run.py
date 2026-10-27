#!/usr/bin/env python3
# Random-clusters baseline: each cell gets a random cluster (seeded -> reproducible)
import argparse, json, os
import anndata as ad
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--output_dir", "-o", required=True)
p.add_argument("--name", "-n", required=True)
p.add_argument("--data.h5ad", dest="data_h5ad", required=True)
p.add_argument("--data.clusters_truth_num", dest="truth_num", required=True)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--d_cluster", type=int, default=0)
p.add_argument("--n_comp", type=int, default=30)
args, _ = p.parse_known_args()

cells = np.asarray(ad.read_h5ad(args.data_h5ad, backed="r").obs_names)
k = max(1, int(open(args.truth_num).read().strip()) + args.d_cluster)

rng = np.random.default_rng(args.seed)
leiden = rng.integers(0, k, len(cells))
louvain = rng.integers(0, k, len(cells))
pcs = rng.standard_normal((len(cells), args.n_comp))

out = lambda ext: os.path.join(args.output_dir, f"{args.name}.{ext}")
pd.DataFrame({"cell_id": cells, "louvain": louvain, "leiden": leiden}).to_csv(out("clusters.tsv"), sep="\t", index=False)
pca = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(args.n_comp)])
pca.insert(0, "cell_id", cells)
pca.to_csv(out("pca.tsv"), sep="\t", index=False)
json.dump({"louvain": None, "leiden": None}, open(out("timings.json"), "w"))
json.dump({"resolutions": {"louvain": None, "leiden": None}, "num_runs": {"louvain": 1, "leiden": 1}},
          open(out("clustering_info.json"), "w"))
print(f"random clusters: {len(cells)} cells, k={k}, seed={args.seed}")
