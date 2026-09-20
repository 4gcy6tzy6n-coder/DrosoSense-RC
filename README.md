# DrosoSense-RC

**Drosophila Sensory Connectome Reservoir Computing for Lightweight AgriFood Sensing**

Initial repository bootstrap. The M1 deliverable — frozen protocol, leakage-audited
food benchmark, dataset manifests and baseline zoo — is under review in the
`data-2/m1-protocol-and-benchmark` branch.

## Data layout (M2 connectome engine)

The repository working tree holds **code and small manifests only**. All large
payloads live in a *data root* outside the repo, so the tree stays small and can
be replayed into an isolated task worktree.

```
<repo-parent>/
├── data-root/                      ← shared, NOT in Git
│   └── connectome/
│       ├── raw/                    ≈9.9 GB FlyWire v783 feather/npy inputs
│       └── metadata/
│           └── olfactory_v1_edge_meta.csv    566 MiB build output
└── repo/                           ← this repository
    └── connectome/
        ├── adjacency/olfactory_v1.npz         47 MB   (committed)
        ├── metadata/olfactory_v1_meta.json            (committed)
        ├── metadata/olfactory_v1_node_meta.csv 6.2 MB (committed)
        └── paths.py                                  data-root resolution
```

`connectome/paths.py` resolves the data root in this order:

1. `$DROSOSENSE_DATA`
2. `<repo-parent>/data-root` (default layout above)
3. `<repo>/connectome` (self-contained clone fallback)

```bash
# default layout needs no configuration; elsewhere:
export DROSOSENSE_DATA=/Volumes/big/flywire

python connectome/build_olfactory_connectome.py --report
python connectome/compute_topology_stats.py
python -m pytest connectome/tests/test_olfactory_connectome.py -v
```

Raw inputs are located by sha256 manifest in
`connectome/metadata/olfactory_v1_meta.json`; the edge CSV is gitignored but
tracked by the same manifest.
