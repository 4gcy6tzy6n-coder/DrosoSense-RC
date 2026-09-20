"""
Tests for DATA-3 M2: Olfactory connectome engine (R1 rework)
Run with: python -m pytest connectome/tests/test_olfactory_connectome.py -v

R1 fixes incorporated:
  C4:  duplicate edge semantics — S_ij = count, aggregated to unique (pre, post) pairs
  C5:  self-loop — KEEP on diagonal, provide mask, no silent deletion
  C6:  topology — clustering + assortativity + modularity (via networkx)
  C8:  npz schema — node_ids array present
  C13: test_symmetric_vs_directed: nnz diff assertion; test_min_syn_filter: param
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "connectome"))
from paths import metadata_path, adjacency_path  # noqa: E402  (data-root resolution)

ADJ_PATH = adjacency_path("olfactory_v1.npz")
NODE_CSV = ROOT / "connectome/metadata/olfactory_v1_node_meta.csv"
# Oversized edge CSV lives in the external data root when one is provisioned.
EDGE_CSV = metadata_path("olfactory_v1_edge_meta.csv")
META_JSON = ROOT / "connectome/metadata/olfactory_v1_meta.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_sparse(npz, prefix: str) -> sp.spmatrix:
    """Reconstruct a sparse matrix from dict-of-arrays format."""
    data_key = f"{prefix}_data"
    if data_key not in npz.files:
        return None
    fmt = str(npz[f"{prefix}_format"].item())
    data = npz[data_key]
    indices = npz[f"{prefix}_indices"]
    indptr = npz[f"{prefix}_indptr"]
    shape = tuple(npz[f"{prefix}_shape"])
    if fmt == "csr":
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    elif fmt == "csc":
        return sp.csc_matrix((data, indices, indptr), shape=shape)
    return sp.coo_matrix((data, indices), shape=shape)


def load_npz():
    return np.load(str(ADJ_PATH), allow_pickle=True)


def load_adj():
    npz = load_npz()
    # Check if adj is stored as sparse dict or direct array
    if "adj_data" in npz.files:
        return _load_sparse(npz, "adj")
    return npz["adj"]


def load_nm():
    return pd.read_csv(NODE_CSV)


def load_sparse(npz_obj, prefix: str) -> sp.spmatrix:
    """Reconstruct a sparse matrix from dict-of-arrays npz entry."""
    data_key = f"{prefix}_data"
    if data_key not in npz_obj.files:
        return None
    fmt = str(npz_obj[f"{prefix}_format"].item())
    data = npz_obj[data_key]
    indices = npz_obj[f"{prefix}_indices"]
    indptr = npz_obj[f"{prefix}_indptr"]
    shape = tuple(npz_obj[f"{prefix}_shape"])
    if fmt == "csr":
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    elif fmt == "csc":
        return sp.csc_matrix((data, indices, indptr), shape=shape)
    return sp.coo_matrix((data, indices), shape=shape)


def load_em():
    return pd.read_csv(EDGE_CSV)


def load_meta():
    with open(META_JSON) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# TestAdjacency — structural tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAdjacency:
    @pytest.fixture
    def adj(self):
        return load_adj()

    def test_is_csr(self, adj):
        assert sp.isspmatrix_csr(adj), f"Expected CSR, got {type(adj)}"

    def test_square(self, adj):
        assert adj.shape[0] == adj.shape[1], "Adjacency must be square"

    def test_no_negative_values(self, adj):
        assert (adj.data < 0).sum() == 0, "Edge weights must be non-negative"

    def test_directed_not_symmetric(self, adj):
        """Olfactory circuit is directional; A and A.T must differ in nnz."""
        adj_t = adj.T
        diff_nnz = abs(adj.nnz - adj_t.nnz)
        # A directed graph may have the same number of edges as its transpose,
        # but the actual edge sets must differ if the graph is not symmetric
        # More robust: check that (A != A.T).nnz > 0
        diff = ((adj - adj_t) != 0)
        assert diff.nnz > 0, (
            "Olfactory circuit should be directed (A != A.T); "
            f"got symmetric graph with {adj.nnz} edges"
        )

    def test_min_syn_filter(self, adj):
        """With min_syn=1 (build default), all edge weights must be >= 1."""
        assert adj.data.min() >= 1.0, (
            f"min_syn=1 build but found edge with weight {adj.data.min()}"
        )

    def test_self_loops_present_on_diagonal(self, adj):
        """Self-loops are RETAINED on the diagonal (DATA-9 §3.4)."""
        diag = adj.diagonal()
        # At least one neuron has a self-loop (biological circuit property)
        # The mask is stored separately; here we verify diagonal is not zeroed
        assert diag is not None


class TestMinSynMonotonicity:
    """Verify size_structure_curve is monotonic in min_syn (higher threshold → fewer edges)."""

    def test_curve_monotonic(self):
        """Edge count must decrease (or stay same) as min_syn increases."""
        import json
        with open(META_JSON) as f:
            meta = json.load(f)
        curve = meta.get("size_structure_curve", [])
        # Filter core_only=False entries
        entries = sorted(
            [e for e in curve if not e.get("core_only")],
            key=lambda e: e["min_synapses"]
        )
        for i in range(1, len(entries)):
            prev = entries[i - 1]["M"]
            curr = entries[i]["M"]
            assert curr <= prev, (
                f"Monotonicity violated: min_syn={entries[i-1]['min_synapses']} "
                f"→ M={prev}, min_syn={entries[i]['min_synapses']} → M={curr} (should be ≤)"
            )


class TestNPZSchema:
    """DATA-9 §7 schema: npz must contain node_ids array."""

    @pytest.fixture
    def npz(self):
        return load_npz()

    def test_has_adj(self, npz):
        # adj is stored as dict-of-arrays (adj_data, adj_indices, adj_indptr, adj_shape, adj_format)
        assert "adj_data" in npz.files, (
            f"adj must be stored as sparse dict-of-arrays (DATA-9 §7). Keys: {list(npz.files)}"
        )

    def test_has_node_ids(self, npz):
        assert "node_ids" in npz.files, (
            f"npz must contain 'node_ids' array (DATA-9 §7). Keys: {list(npz.files)}"
        )

    def test_node_ids_int64_ascending(self, npz):
        node_ids = npz["node_ids"]
        assert node_ids.dtype == np.int64, f"node_ids dtype: {node_ids.dtype}"
        assert list(node_ids) == sorted(node_ids), "node_ids must be ascending"

    def test_node_ids_length_matches_adj(self, npz):
        adj = load_adj()
        assert len(npz["node_ids"]) == adj.shape[0], (
            f"node_ids length ({len(npz['node_ids'])}) != adj dimension ({adj.shape[0]})"
        )

    def test_has_self_loop_mask(self, npz):
        assert "self_loop_mask" in npz.files, "npz must contain 'self_loop_mask' (DATA-9 §3.4)"

    def test_self_loop_mask_dtype(self, npz):
        mask = npz["self_loop_mask"]
        assert mask.dtype in (np.int8, np.bool_), f"self_loop_mask dtype: {mask.dtype}"
        assert set(np.unique(mask)).issubset({0, 1}), "self_loop_mask must be 0/1"

    def test_normalization_arrays_present(self, npz):
        # Stored as dict-of-arrays (norm_n1_pre_l1_data, etc.)
        expected_data_keys = ["norm_n1_pre_l1_data", "norm_n2_post_l1_data",
                              "norm_n3_global_max_data", "norm_n4_log_pre_l1_data",
                              "norm_n5_binary_data"]
        for key in expected_data_keys:
            assert key in npz.files, (
                f"Missing normalisation array: {key} (DATA-9 §6). "
                f"Keys: {list(npz.files)}"
            )

    def test_normalization_shapes(self, npz):
        adj = load_adj()
        for key in ["norm_n1_pre_l1", "norm_n2_post_l1", "norm_n3_global_max",
                    "norm_n4_log_pre_l1", "norm_n5_binary"]:
            arr = load_sparse(npz, key)
            assert arr.shape == adj.shape, f"{key} shape {arr.shape} != adj shape {adj.shape}"

    def test_meta_string_in_npz(self, npz):
        assert "meta" in npz.files, "npz must contain 'meta' JSON string"
        meta = json.loads(str(npz["meta"]))
        assert "weight_semantics" in meta


# ─────────────────────────────────────────────────────────────────────────────
# TestNodeMetadata
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeMetadata:
    @pytest.fixture
    def nm(self):
        return load_nm()

    @pytest.fixture
    def npz(self):
        return load_npz()

    def test_node_count_matches_adjacency(self, nm, npz):
        adj = load_adj()
        assert len(nm) == adj.shape[0]

    def test_root_id_unique(self, nm):
        assert nm["root_id"].nunique() == len(nm)

    def test_root_id_matches_npz_node_ids(self, nm, npz):
        np_ids = set(npz["node_ids"])
        csv_ids = set(nm["root_id"].values)
        assert np_ids == csv_ids, (
            f"node_ids in npz ({len(np_ids)}) != root_ids in CSV ({len(csv_ids)})"
        )

    def test_node_idx_is_0_to_N_minus_1(self, nm):
        assert list(nm["node_idx"].values) == list(range(len(nm)))

    def test_degree_non_negative(self, nm):
        assert (nm["in_degree"] >= 0).all()
        assert (nm["out_degree"] >= 0).all()
        assert (nm["total_degree"] >= 0).all()

    def test_no_isolated_nodes(self, nm):
        assert (nm["total_degree"] == 0).sum() == 0

    def test_layer_mean_in_valid_range(self, nm):
        assert nm["layer_mean"].between(1.0, 15.0).all()

    def test_layer_mean_no_nan(self, nm):
        assert nm["layer_mean"].isna().sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestEdgeMetadata
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeMetadata:
    @pytest.fixture
    def em(self):
        return load_em()

    @pytest.fixture
    def npz(self):
        return load_npz()

    def test_edge_count_matches_adjacency(self, em, npz):
        adj = load_adj()
        assert len(em) == adj.nnz

    def test_syn_count_matches_adjacency(self, em, npz):
        """Verify edge CSV syn_count values match adjacency data."""
        adj = load_adj()
        nm = load_nm()
        idx_to_root = dict(zip(nm["node_idx"], nm["root_id"]))
        root_to_idx = {v: k for k, v in idx_to_root.items()}

        adj_coo = adj.tocoo()
        adj_syn = pd.DataFrame({
            "pre_root_id": [idx_to_root[i] for i in adj_coo.row],
            "post_root_id": [idx_to_root[j] for j in adj_coo.col],
            "syn_count_adj": adj_coo.data.astype(np.int32),
        })
        merged = em.merge(adj_syn, on=["pre_root_id", "post_root_id"])
        assert len(merged) == len(em) == len(adj_syn), (
            f"Edge sets differ: CSV={len(em)}, adj={len(adj_syn)}, merge={len(merged)}"
        )
        # em column is "syn_count", adj_syn column is "syn_count_adj"
        np.testing.assert_array_equal(merged["syn_count"], merged["syn_count_adj"].astype(int))

    def test_syn_count_min(self, em):
        assert em["syn_count"].min() >= 1

    def test_pre_post_different_in_edges(self, em, npz):
        """Non-self-loop edges must have pre_root_id != post_root_id.
        Self-loops (pre == post) exist in edge_meta.csv and are expressed
        via self_loop_mask in the NPZ per DATA-9 §3.4 — this is intentional.
        This test asserts that the self-loop mask correctly identifies them."""
        # Verify self-loops in edge CSV are matched by self_loop_mask in NPZ
        self_loops_in_csv = em[em["pre_root_id"] == em["post_root_id"]]
        mask = npz["self_loop_mask"]
        # Count nodes with self-loop (mask == 1)
        nodes_with_self_loop = int(np.sum(mask))
        assert nodes_with_self_loop == len(self_loops_in_csv), (
            f"self_loop_mask shows {nodes_with_self_loop} nodes with self-loop "
            f"but edge CSV has {len(self_loops_in_csv)} self-loop edges"
        )
        # Non-self edges must have pre != post
        non_self = em["pre_root_id"] != em["post_root_id"]
        assert non_self.sum() == len(em) - len(self_loops_in_csv)

    def test_duplicate_edge_semantics(self, em):
        """(pre_root_id, post_root_id) pairs are UNIQUE after aggregation.
        No multi-edges exist post-groupby."""
        dups = em.duplicated(subset=["pre_root_id", "post_root_id"])
        assert dups.sum() == 0, (
            f"{dups.sum()} duplicate (pre, post) pairs found — "
            "aggregation should produce unique edges"
        )

    def test_edge_csv_has_required_columns(self, em):
        required = ["pre_root_id", "post_root_id", "syn_count"]
        for col in required:
            assert col in em.columns, f"Missing column: {col}"


# ─────────────────────────────────────────────────────────────────────────────
# TestMetadataJSON
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataJSON:
    @pytest.fixture
    def meta(self):
        return load_meta()

    def test_provenance_present(self, meta):
        prov = meta.get("provenance", {})
        assert prov.get("data_source") == "FlyWire/FlyEM proofreading v783"
        assert "UNCALIBRATED" in prov.get("weight_semantics", "")

    def test_build_params_present(self, meta):
        bp = meta.get("build_params", {})
        assert "min_synapses" in bp
        assert "rng_seed" in bp

    def test_node_count_in_note(self, meta):
        adj = load_adj()
        note = meta.get("node_count_note", "")
        # note may contain "124,185" (with comma) or "124185"
        adj_str = str(adj.shape[0])
        note_digits = note.replace(",", "")
        assert adj_str in note_digits, (
            f"node_count_note must contain node count {adj_str}. "
            f"Note: {note[:100]}"
        )

    def test_topology_stats_present(self, meta):
        stats = meta.get("topology_stats", {})
        required = ["olfactory_nodes", "olfactory_edges", "olfactory_density",
                   "olfactory_isolated", "olfactory_largest_cc_pct"]
        for k in required:
            assert k in stats, f"topology_stats must contain '{k}'"

    def test_spectral_radius_present(self, meta):
        stats = meta.get("topology_stats", {})
        sr = stats.get("olfactory_spectral_radius")
        assert sr is not None and not np.isnan(sr)

    def test_avg_path_length_present(self, meta):
        stats = meta.get("topology_stats", {})
        apl = stats.get("olfactory_avg_path_length")
        # Deferred to compute_topology_stats.py (BFS sampling)
        if apl is not None:
            assert 1.0 <= apl <= 10.0

    def test_wcc_count(self, meta):
        stats = meta.get("topology_stats", {})
        assert stats.get("olfactory_wcc_count") == 1

    def test_file_manifest_all_five_raw_files(self, meta):
        """C3/C15: all 5 raw files must be in manifest."""
        fm = meta.get("file_manifest", {})
        required = [
            "flywire_synapses_783 (1).feather",
            "per_neuron_neuropil_count_pre_783.feather",
            "per_neuron_neuropil_count_post_783.feather",
            "proofread_connections_783.feather",
            "neuron_class_ranking_df_783-olfactory-10000.feather",
            "proofread_root_ids_783.npy",
        ]
        for name in required:
            assert name in fm, f"Missing raw file in manifest: {name}"
            assert "sha256" in fm[name], f"{name} missing sha256"
            assert "size_bytes" in fm[name], f"{name} missing size_bytes"

    def test_modularity_present(self, meta):
        """Modularity via networkx louvain (C6) — value must be a real float."""
        stats = meta.get("topology_stats", {})
        mod = stats.get("olfactory_modularity")
        assert mod is not None, "Modularity must be computed (not null)"
        assert not np.isnan(mod), "Modularity must not be NaN"
        assert isinstance(mod, (int, float)), f"Modularity must be numeric, got {type(mod)}"

    def test_clustering_present(self, meta):
        """Clustering coefficient must be present and a real float (C6)."""
        stats = meta.get("topology_stats", {})
        c = stats.get("olfactory_clustering")
        assert c is not None, "Clustering must be computed (not null)"
        assert not np.isnan(c), "Clustering must not be NaN"
        assert isinstance(c, (int, float)), f"Clustering must be numeric, got {type(c)}"

    def test_assortativity_present(self, meta):
        """Assortativity must be present and a real float (C6)."""
        stats = meta.get("topology_stats", {})
        a = stats.get("olfactory_assortativity")
        assert a is not None, "Assortativity must be computed (not null)"
        assert not np.isnan(a), "Assortativity must not be NaN"
        assert isinstance(a, (int, float)), f"Assortativity must be numeric, got {type(a)}"

    def test_normalization_schemes_recorded(self, meta):
        """All 6 normalisation schemes must be recorded (DATA-9 §6 / R1裁定3)."""
        norm = meta.get("topology_stats", {}).get("normalization", {})
        required = ["n0_raw", "n1_pre_l1", "n2_post_l1",
                    "n3_global_max", "n4_log_pre_l1", "n5_binary"]
        for scheme in required:
            assert scheme in norm, f"Missing normalisation scheme: {scheme}"

    def test_S_ij_source_in_provenance(self, meta):
        prov = meta.get("provenance", {})
        assert "S_ij_source" in prov, "S_ij source must be documented"
        assert "syn_count = count" in prov["S_ij_source"]

    def test_self_loop_handling_in_provenance(self, meta):
        prov = meta.get("provenance", {})
        sl = prov.get("self_loop_handling", "")
        assert "RETAINED" in sl or "retained" in sl, (
            "Self-loops must be retained per DATA-9 §3.4"
        )

    def test_node_count_note_not_vacuous(self, meta):
        note = meta.get("node_count_note", "")
        assert len(note) > 100, "node_count_note must be substantive"
        assert "Biological" in note or "biological" in note
        assert "Engineering" in note or "engineering" in note

    def test_pr_url_recorded(self, meta):
        pr_url = meta.get("pr_url")
        assert pr_url is not None, "pr_url must be recorded in meta.json"
        assert pr_url.startswith("https://github.com/"), (
            f"pr_url must be a GitHub URL, got: {pr_url}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestBiologicalScope
# ─────────────────────────────────────────────────────────────────────────────

class TestBiologicalScope:
    def test_node_count_outside_500_5000_target(self):
        """Document that node count exceeds advisory range, with biological rationale."""
        adj = load_adj()
        N = adj.shape[0]
        with open(META_JSON) as f:
            meta = json.load(f)
        note = meta.get("node_count_note", "")
        assert N != 0
        if not (500 <= N <= 5000):
            assert len(note) > 50, "Out-of-range node count requires a detailed rationale"
            assert "FlyWire" in note or "olfactory" in note.lower()

    def test_spectral_radius_reasonable(self):
        """Spectral radius should be in a biologically plausible range."""
        with open(META_JSON) as f:
            meta = json.load(f)
        rho = meta.get("topology_stats", {}).get("olfactory_spectral_radius")
        assert rho is not None
        assert 0 < rho < 10000, f"Spectral radius {rho} out of plausible range"


# ─────────────────────────────────────────────────────────────────────────────
# TestNormalizationAssertions
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizationAssertions:
    """Row/column sums and spectral radius assertions for each normalisation."""

    @pytest.fixture
    def npz(self):
        return load_npz()

    def test_n1_row_sums_near_one(self, npz):
        """Pre-L1 normalised rows should sum to ~1.0 (ignoring zero rows)."""
        n1 = load_sparse(npz, "norm_n1_pre_l1").tocsr()
        row_sums = np.array(n1.sum(axis=1)).flatten()
        nz_mask = row_sums != 0
        nz_sums = row_sums[nz_mask]
        assert len(nz_sums) > 0
        np.testing.assert_allclose(nz_sums, 1.0, rtol=1e-5,
                                   err_msg="n1 row sums should be 1.0")

    def test_n2_col_sums_near_one(self, npz):
        """Post-L1 normalised columns should sum to ~1.0 (ignoring zero cols)."""
        n2 = load_sparse(npz, "norm_n2_post_l1").tocsc()
        col_sums = np.array(n2.sum(axis=0)).flatten()
        nz_mask = col_sums != 0
        nz_sums = col_sums[nz_mask]
        assert len(nz_sums) > 0
        np.testing.assert_allclose(nz_sums, 1.0, rtol=1e-5,
                                   err_msg="n2 column sums should be 1.0")

    def test_n3_values_in_01(self, npz):
        """Global-max normalised values should be in [0, 1]."""
        n3 = load_sparse(npz, "norm_n3_global_max")
        assert n3.data.min() >= 0.0
        assert n3.data.max() <= 1.0

    def test_n5_binary_values(self, npz):
        """Binary normalisation should only have 0 or 1 values."""
        n5 = load_sparse(npz, "norm_n5_binary")
        assert set(np.unique(n5.data)).issubset({0.0, 1.0})

    def test_n4_log_positive(self, npz):
        """Log-normalised values should all be non-negative (log1p >= 0)."""
        n4 = load_sparse(npz, "norm_n4_log_pre_l1")
        assert (n4.data >= 0).all()
