"""Read-only: which design can satisfy C4.1-C4.7 TOGETHER?

Measured facts so far, on the C2 substrate (N=1000, 80,443 edges):

  A. rewire R0's NORMALIZED matrix, exact-weight partners only
     -> in-strength error 7.7e-08 (exact), overlap stalls at 0.858, 600 s budget spent
  B. rewire R0's NORMALIZED matrix, near-weight partners (+-25 %)
     -> overlap 0.351 after 311 s, in-strength median relative error 0.318

The reason A cannot mix: R0's matrix carries normalized+rescaled weights, and that
destroys the raw graph's quantization -- 80,443 edges spread over 17,213 distinct values
(mean class 4.7 edges), so an exact-weight swap has almost no partners and the reachable
target set per edge is tiny. The RAW synapse-count graph has 363 distinct values over the
same edges (mean class 221, 99.87 % of edges with an exact partner).

This measures design C: rewire the RAW quantized matrix (exact-weight partners, which are
plentiful there), then apply the SAME declared normalization and spectral rescale to both
graphs. C then satisfies C4.1/C4.2/C4.3 ON THE RAW OBJECT and mixes freely, but the
NORMALIZED matrices differ in weight multiset as a consequence of the wiring -- which is a
specification decision (does C4.2 mean "the reservoir's matrix keeps R0's multiset", or
"the wiring no longer destroys the weight structure at the source"?), not a code decision.

Nothing is written anywhere.
"""
import sys
import time
from pathlib import Path as _Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from connectome.cell_types import load_node_classes
from drososense.connectome_selection import expand_from_orns
from drososense.reservoir.connectome_reservoir import (
    ALLOWED_NORMALIZATIONS,
    load_reservoir_topology_from_npz,
    rescale_to_spectral_radius,
)
from drososense.reservoir.input_mapping import CellTypeAnnotation
from drososense.reservoir.r2_counterfactual import (
    counterfactual_quality,
    per_source_multiset_hash,
    weight_preserving_degree_rewire,
)

DATA = "/root/autodl-tmp/drososense/data-root"
NPZ = f"{DATA}/connectome/adjacency/olfactory_v1.npz"
META = "/root/autodl-tmp/drososense/repo/connectome/metadata/olfactory_v1_node_meta.csv"

with np.load(NPZ, allow_pickle=True) as payload:
    shape = tuple(int(v) for v in payload["adj_shape"])
    node_ids = np.asarray(payload["node_ids"])
    matrix = sp.csr_matrix(
        (np.asarray(payload["adj_data"], dtype=float),
         np.asarray(payload["adj_indices"]),
         np.asarray(payload["adj_indptr"])),
        shape=shape,
    )
classes = load_node_classes(META)
counts = {}
for value in classes.values():
    counts[value] = counts.get(value, 0) + 1
annotation = CellTypeAnnotation(classes=classes, source=META, per_class_counts=counts)
expansion = expand_from_orns(matrix, node_ids, annotation, 1000, din=5, seed=0)
idx = expansion.node_indices


def weight_classes(sub, label):
    w = sub.tocoo().data.astype(np.float64)
    values, counts_w = np.unique(w, return_counts=True)
    order = np.argsort(w)
    sw = w[order]
    window = np.searchsorted(sw, sw, side="right") - np.searchsorted(sw, sw, side="left")
    print(
        f"   {label}: edges={w.size:,} distinct={values.size:,} "
        f"mean class={w.size / values.size:.1f} "
        f"frac with an exact partner={float((window >= 2).mean()):.4f}"
    )


# --- the RAW quantized block -------------------------------------------------
raw = matrix[idx][:, idx].tocsr()
print("RAW substrate:")
weight_classes(raw, "raw synapse-count weights")

t0 = time.monotonic()
raw_rewired, raw_report = weight_preserving_degree_rewire(
    raw, seed=0, target_overlap=0.20, time_budget_s=300
)
print(
    f"   raw exact-weight rewire: accepted={raw_report.swaps_accepted:,} "
    f"overlap={raw_report.overlap_final:.4f} seconds={raw_report.seconds:.1f} "
    f"stopped={raw_report.stopped_because} "
    f"accepted_near={raw_report.swaps_accepted_near_weight:,}"
)

# --- normalize BOTH graphs identically --------------------------------------
def normalize_and_rescale(sub, normalization="n1_pre_l1", rho_target=0.9):
    if normalization == "n1_pre_l1":
        col = np.asarray(np.abs(sub).sum(axis=0)).ravel()
        col[col == 0] = 1.0
        scaled = sp.diags(1.0 / col) @ sub
    else:  # pragma: no cover
        raise SystemExit(f"unhandled normalization {normalization!r}")
    return rescale_to_spectral_radius(sp.csr_matrix(scaled), rho_target)


r0_norm = normalize_and_rescale(raw)
r2_norm = normalize_and_rescale(raw_rewired)
print("NORMALIZED objects (what the reservoir sees):")
weight_classes(r0_norm, "R0 normalized")
weight_classes(r2_norm, "R2 normalized")
print(
    f"   normalized multiset identical: "
    f"{np.array_equal(np.sort(r0_norm.tocoo().data), np.sort(r2_norm.tocoo().data))}"
)

# overlap of the normalized pair, and of the raw pair
def overlap(a, b):
    ka = set(zip(*a.tocoo().row.tolist() and (a.tocoo().row.tolist(), a.tocoo().col.tolist())))
    kb = set(zip(b.tocoo().row.tolist(), b.tocoo().col.tolist()))
    return len(ka & kb) / len(ka)


print(f"   raw overlap={overlap(raw, raw_rewired):.4f} "
      f"normalized overlap={overlap(r0_norm, r2_norm):.4f}")

q = counterfactual_quality(r0_norm, r2_norm, raw_report)
print("C4 measured on the NORMALIZED pair (design C):")
for name, value in q["verdict"]["lines"].items():
    shown = f"{value:.6g}" if isinstance(value, float) else str(value)
    print(f"   {name:26s} {shown:>12s}  {'PASS' if q['verdict']['checks'][name] else 'FAIL'}")

# --- what design C preserves on the RAW object ------------------------------
raw_coo, rew_coo = raw.tocoo(), raw_rewired.tocoo()
print("design C, conservation on the RAW object:")
print(
    "   degrees exact:",
    bool(
        np.array_equal(
            np.bincount(raw_coo.row.astype(np.int64), minlength=raw.shape[0]),
            np.bincount(rew_coo.row.astype(np.int64), minlength=raw.shape[0]),
        )
        and np.array_equal(
            np.bincount(raw_coo.col.astype(np.int64), minlength=raw.shape[0]),
            np.bincount(rew_coo.col.astype(np.int64), minlength=raw.shape[0]),
        )
    ),
)
print(
    "   raw weight multiset exact:",
    bool(np.array_equal(np.sort(raw_coo.data), np.sort(rew_coo.data))),
    "| per-source exact:",
    per_source_multiset_hash(raw_coo.row.astype(np.int64), raw_coo.data.astype(float))
    == per_source_multiset_hash(rew_coo.row.astype(np.int64), rew_coo.data.astype(float)),
)

# --- how far does the normalized multiset move, and why? --------------------
w0 = np.sort(r0_norm.tocoo().data.astype(np.float64))
w2 = np.sort(r2_norm.tocoo().data.astype(np.float64))
rel = np.abs(w2 - w0) / np.maximum(np.abs(w0), 1e-12)
print(
    f"   normalized weight values: median relative shift={float(np.median(rel)):.4f} "
    f"p90={float(np.quantile(rel, 0.9)):.4f} "
    f"(column L1 sums differ because the columns are wired differently)"
)
print(f"   in-strength median relative error on the normalized pair: "
      f"{q['in_strength']['median_relative_error']:.6g}")
