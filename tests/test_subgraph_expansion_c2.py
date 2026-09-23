"""C2 — the grown, connected, edge-retaining olfactory subgraph (v2 D3/D4).

The v1 defect these tests are written against: at N=250 the selected subgraph was a
**126-edge residue** with **64.8 % isolated** nodes, largest weak component **14.8 %**
of N and mean in-subgraph out-degree **0.50** — the "layer-stratified
degree-proportional" sampler kept no more wiring than a random node set.

C2 is a property of the CONSTRUCTION, so every test here fixes a rule rather than a
score:

* **C2.1** the selected subgraph carries the typed populations (its ORN seeds are the
  declared ORN population, and no untyped node is ever added);
* **C2.2** eligible-edge retention, with the denominator the pre-registration fixed —
  the induced eligible edges of the selected node set, never the whole-brain count;
* **C2.3/C2.4/C2.5** component, isolation and degree, computed on the induced
  subgraph;
* **C2.6** the selection is deterministic, declared, and carries `target_n`, seed and
  sha256 on its provenance.

The real measurement runs on the GPU box (`ops/audit/v2_construct_check.py --only C2`);
these tests pin the rules with synthetic graphs and the committed vocabulary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drososense.connectome_selection import (  # noqa: E402
    LAYER_EXPANSION_GROUPS,
    METHOD_ORN_EXPANSION,
    ORN_DENSITY_LIMIT,
    expand_from_orns,
    layer_group_floors,
    orn_budget,
    subgraph_quality,
)
from drososense.reservoir.input_mapping import annotation_from_mapping  # noqa: E402

#: The delivered graph's ORN population, from the committed vocabulary.
DELIVERED_ORN = 2267


def _classes(n: int) -> dict[int, str]:
    """A typed graph layout that can satisfy C1.4 at Din <= 5.

    ORN 30, PN 60, KC 40, higher_order 20, MBON 20, DAN 15, rest other: the
    KENYON-cell floor (max(50, 4*Din) = 50 at Din=5) needs 50 of KC + higher_order,
    which is why the fixture carries 60. A fixture that cannot satisfy the criteria
    could only test the refusals.
    """
    layout = (
        ("ORN", 30),
        ("PN", 60),
        ("KC", 40),
        ("higher_order", 20),
        ("MBON", 20),
        ("DAN", 15),
        ("other", 40),
    )
    out: dict[int, str] = {}
    row = 0
    for cell_class, count in layout:
        for _ in range(count):
            out[row] = cell_class
            row += 1
    while row < n:
        out[row] = "other"
        row += 1
    return out


def _graph(n: int, *, density: float = 0.05, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = (rng.random((n, n)) < density).astype(float)
    np.fill_diagonal(dense, 0.0)
    return sp.csr_matrix(dense)


def _two_clusters(n: int = 200, *, seed: int = 1) -> sp.csr_matrix:
    """Two dense blocks with a single bridge — an enrichable structure."""
    rng = np.random.default_rng(seed)
    dense = np.zeros((n, n))
    half = n // 2
    for block in (slice(0, half), slice(half, n)):
        size = len(range(*block.indices(n)))
        dense[block, block] = (rng.random((size, size)) < 0.5).astype(float)
    dense[half - 1, half] = 1.0
    np.fill_diagonal(dense, 0.0)
    return sp.csr_matrix(dense)


@pytest.fixture()
def typed_graph():
    n = 240
    return _graph(n, density=0.06), np.arange(n), annotation_from_mapping(_classes(n))


# ---------------------------------------------------------------------------
# C2.1 / D3 -- what the expansion selects
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c2_1_the_seeds_are_the_declared_orn_population(typed_graph):
    """The ORN allocation is filled from the declared population, and nothing untyped."""
    matrix, roots, annotation = typed_graph
    selection = expand_from_orns(matrix, roots, annotation, 120, din=5)
    selected_classes = annotation.class_of(roots)[selection.node_indices]
    assert int((selected_classes == "ORN").sum()) == selection.detail["allocation"]["ORN"]
    assert int((selected_classes == "ORN").sum()) >= 20
    assert "other" not in set(selected_classes.tolist())
    assert set(selected_classes.tolist()) <= {"ORN", "PN", "KC", "higher_order", "MBON", "DAN"}


@pytest.mark.unit
def test_c2_1_every_declared_layer_gets_at_least_its_c1_4_floor(typed_graph):
    """The first implementation starved KC and MBON/DAN completely; this pins the fix.

    Measured on the delivered graph before the allocation existed: ``{ORN: 500,
    PN: 500}`` -- zero Kenyon cells and zero MBON/DAN, which FAILS C2.1.
    """
    matrix, roots, annotation = typed_graph
    selection = expand_from_orns(matrix, roots, annotation, 120, din=5)
    counts = selection.detail["layer_counts"]
    floors = selection.detail["group_floors_from_c1_4"]
    assert counts["ORN"] >= floors["ORN"] >= 20
    assert counts["PN"] >= floors["PN"] >= 20
    assert counts["KC"] + counts["higher_order"] >= floors["KC+higher_order"] == 50
    assert counts.get("MBON", 0) + counts.get("DAN", 0) >= floors["MBON+DAN"] == 20
    assert selection.detail["consumes_randomness"] is False
    assert selection.method == METHOD_ORN_EXPANSION
    assert [list(g) for g in LAYER_EXPANSION_GROUPS] == [
        ["PN"],
        ["KC", "higher_order"],
        ["MBON", "DAN"],
    ]


@pytest.mark.unit
def test_c2_1_the_orn_budget_is_the_c1_2_bound_not_a_choice():
    """One ORN one non-zero, so C1.2 caps the population at 0.10*Din*N."""
    assert orn_budget(1000, 5, DELIVERED_ORN) == 500
    assert orn_budget(1000, 3, DELIVERED_ORN) == 300
    assert orn_budget(250, 5, DELIVERED_ORN) == 125
    # the graph may offer fewer than the budget: then every ORN is a seed
    assert orn_budget(1000, 3, 100) == 100
    assert ORN_DENSITY_LIMIT == 0.10


@pytest.mark.unit
def test_c2_1_a_budget_below_c1_4_is_refused_not_under_filled():
    """A substrate that cannot satisfy its own input criterion is refused."""
    n = 40
    matrix = _graph(n)
    roots = np.arange(n)
    annotation = annotation_from_mapping({int(i): ("ORN" if i < 2 else "PN") for i in range(n)})
    with pytest.raises(ValueError, match="cannot satisfy the declared allocation"):
        # C1.4's floors sum to 110 at Din=1 and C1.2's ceiling needs N >= 200
        expand_from_orns(matrix, roots, annotation, 40, din=1)
    # and a graph that simply does not contain a population is refused by name
    thin = annotation_from_mapping(
        {**{int(i): "ORN" for i in range(30)}, **{int(i): "PN" for i in range(30, 240)}}
    )
    with pytest.raises(ValueError, match="needs 50 nodes for C1.4"):
        expand_from_orns(_graph(240), np.arange(240), thin, 200, din=5)


@pytest.mark.unit
def test_c2_1_a_graph_without_orns_is_refused():
    matrix = _graph(30)
    roots = np.arange(30)
    annotation = annotation_from_mapping({int(i): "PN" for i in range(30)})
    with pytest.raises(ValueError, match="no ORN row"):
        expand_from_orns(matrix, roots, annotation, 20, din=3)


@pytest.mark.unit
def test_c2_1_a_shortfall_is_reported_never_padded(typed_graph):
    """Typed populations exhausted → shortfall with its reason, not 'other' filler."""
    matrix, roots, annotation = typed_graph
    # typed nodes are 185 (30+60+40+20+20+15) of 240
    selection = expand_from_orns(matrix, roots, annotation, 200, din=5)
    assert selection.node_indices.size == 185
    assert selection.detail["shortfall"] == 15
    assert "other" in selection.detail["shortfall_reason"]
    assert selection.detail["covered_all_typed_layers"] is False
    assert "other" not in set(annotation.class_of(roots)[selection.node_indices].tolist())


# ---------------------------------------------------------------------------
# C2.6 -- determinism and declared provenance
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c2_6_the_selection_is_deterministic_and_declared(typed_graph):
    matrix, roots, annotation = typed_graph
    first = expand_from_orns(matrix, roots, annotation, 140, din=4)
    second = expand_from_orns(matrix, roots, annotation, 140, din=4)
    assert np.array_equal(first.node_indices, second.node_indices)
    assert first.sha256 == second.sha256

    described = first.describe()
    assert described["method"] == METHOD_ORN_EXPANSION
    assert described["target_n"] == 140
    assert described["n_selected"] == first.node_indices.size
    assert described["sha256_sorted_root_ids"] == first.sha256
    assert described["node_index_sha256"]
    detail = described["detail"]
    assert detail["orn_budget"] == orn_budget(140, 4, 30)
    assert detail["allocation_rule"].startswith("max(equal share of N")
    assert detail["frontier_order"].startswith("cumulative synapse mass")
    assert detail["consumes_randomness"] is False


@pytest.mark.unit
def test_c2_6_the_digest_uses_the_v1_convention(typed_graph):
    """A v2 digest must be comparable with the delivered v1 selection digests."""
    from connectome.select_neurons import sha256_of_array

    roots = np.arange(1000, dtype=np.int64) * 7919 + 720575940596125868
    from drososense.connectome_selection import _root_id_digest

    assert _root_id_digest(roots) == sha256_of_array(roots)


@pytest.mark.unit
def test_c2_6_the_frontier_prefers_the_stronger_connection(typed_graph):
    """Multi-source frontier: the best-connected candidate of the layer wins.

    Exactly one KC/HO candidate (row 60, a KC) is connected to the selected set, so
    it must be inside the slice of that group a 70-node substrate can hold.
    """
    matrix, roots, annotation = typed_graph
    dense = matrix.toarray()
    # give KC row 60 a large incident mass from the PN range (rows 20-59)
    dense[20:60, 60] = 4.0
    dense[60, 20:60] = 4.0
    np.fill_diagonal(dense, 0.0)
    weighted = sp.csr_matrix(dense)

    selection = expand_from_orns(weighted, roots, annotation, 120, din=5)
    counts = selection.detail["layer_counts"]
    assert counts["KC"] + counts["higher_order"] >= 50
    assert 60 in selection.node_indices.tolist(), "the connected KC must win the frontier"


@pytest.mark.unit
def test_c2_1_the_smallest_admissible_substrate_is_derived(typed_graph):
    """C1.2 and C1.4 together fix a minimum N, and the expansion refuses below it.

    Two derivations, not a hope: every layer group needs its C1.4 floor
    (``sum(floors)`` = 110 at Din=5), and C1.2's ceiling must be able to cover the
    ORN floor (``N >= max(20,2*Din)/(0.10*Din)`` = 200/Din). The admissible N is the
    larger of the two.
    """
    from drososense.connectome_selection import minimum_admissible_n

    for din in (3, 5, 7):
        floors = sum(layer_group_floors(din))
        from_ceiling = int(np.ceil(max(20, 2 * din) / (0.10 * din)))
        assert minimum_admissible_n(din) == max(floors, from_ceiling)
        assert minimum_admissible_n(din) >= 110

    matrix, roots, annotation = typed_graph
    below = minimum_admissible_n(5) - 1
    with pytest.raises(ValueError, match="cannot satisfy the declared allocation"):
        expand_from_orns(matrix, roots, annotation, below, din=5)


# ---------------------------------------------------------------------------
# C2.2 -- retention, with the signed denominator
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c2_2_retention_denominator_is_the_induced_eligible_set(typed_graph):
    matrix, roots, annotation = typed_graph
    selection = expand_from_orns(matrix, roots, annotation, 120, din=3)
    quality = subgraph_quality(matrix, selection.node_indices)
    # the denominator is the induced eligible set, NOT the whole graph's edge count
    assert quality["induced_eligible_edges"] < quality["n_full_edges"]
    assert quality["eligible_edge_retention_denominator"].startswith("induced eligible")
    # the construction keeps what biology provides
    assert quality["eligible_edge_retention"] == pytest.approx(1.0)
    assert quality["eligible_edges_kept"] == quality["induced_eligible_edges"]
    assert quality["n_nodes"] == 120


@pytest.mark.unit
def test_c2_2_self_loops_are_excluded_from_eligible_but_reported():
    n = 20
    matrix = sp.csr_matrix(np.eye(n) * 5.0 + np.roll(np.eye(n), 1) * 2.0)
    quality = subgraph_quality(matrix, np.arange(n))
    assert quality["self_loops_in_subgraph"] == n
    assert quality["induced_eligible_edges"] == quality["induced_edges_incl_self_loops"] - n
    assert quality["eligible_edge_retention"] == pytest.approx(1.0)


@pytest.mark.unit
def test_c2_2_the_enrichment_form_is_stated_and_informative():
    """Under the corrected denominator the retention ratio is ~1 for any node set,
    which is why Enrichment is defined on the connectome SHARE instead."""
    matrix = _two_clusters(200)
    # a selection inside one cluster: few nodes, most of the connectome's edges
    inside = np.arange(20)
    quality = subgraph_quality(matrix, inside, random_seed=3, n_random_draws=5)
    assert quality["eligible_edge_retention"] == pytest.approx(1.0)
    assert (
        quality["induced_share_of_connectome"]
        > quality["induced_share_of_connectome_random_same_size"]
    )
    assert quality["enrichment"] > 1.5, (
        f"a selection confined to one dense block must beat a random set of the same "
        f"size: enrichment={quality['enrichment']}"
    )
    assert "retention ratio is ~1" in quality["enrichment_form"]


# ---------------------------------------------------------------------------
# C2.3 / C2.4 / C2.5 -- component, isolation, degree
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c2_3_c2_4_c2_5_are_measured_on_the_induced_subgraph():
    """A ring of 10 plus 2 isolated nodes: known component, isolation, degree."""
    n = 12
    rows = list(range(10))
    cols = [(i + 1) % 10 for i in range(10)]
    matrix = sp.csr_matrix((np.ones(10), (rows, cols)), shape=(n, n))
    quality = subgraph_quality(matrix, np.arange(n))
    assert quality["mean_unweighted_out_degree"] == pytest.approx(10 / 12)
    assert quality["n_isolated"] == 2
    assert quality["isolated_fraction"] == pytest.approx(2 / 12)
    assert quality["largest_weak_component_fraction"] == pytest.approx(10 / 12)
    # and a disconnected pair of cliques is seen as two components
    block = sp.block_diag(
        [sp.csr_matrix(np.ones((5, 5)) - np.eye(5)), sp.csr_matrix(np.ones((5, 5)) - np.eye(5))]
    ).tocsr()
    split = subgraph_quality(block, np.arange(10))
    assert split["largest_weak_component_fraction"] == pytest.approx(0.5)


@pytest.mark.unit
def test_c2_metrics_refuse_to_divide_by_an_empty_subgraph():
    matrix = _graph(10)
    quality = subgraph_quality(matrix, np.array([], dtype=np.int64))
    assert quality["n_nodes"] == 0
    assert np.isnan(quality["eligible_edge_retention"])
    assert np.isnan(quality["largest_weak_component_fraction"])
