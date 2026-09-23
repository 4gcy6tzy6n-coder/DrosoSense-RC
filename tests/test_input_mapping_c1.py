"""C1 — the ORN-aligned sparse input mapping (v2 constraint 1).

Every test here is a property of the CONSTRUCTION, never of a model's score, and
each one names the criterion it pins:

* **C1.1** the set of nodes receiving external input is exactly the declared ORN
  population, and nothing else;
* **C1.2** input-map density ``nnz(W_in)/(N*Din) <= 0.10``, which (one non-zero per
  ORN) is exactly the ORN-fraction bound ``n_ORN/N <= 0.10*Din``;
* **C1.3** reassigning which nodes are ORNs, graph fixed, MUST move the states --
  the strict inverse of the v1 measurement (``max||dh|| = 0``, ``A6_input_mapping``);
* **C1.4** the population sizes present in the substrate, against the signed
  thresholds;
* **C1.5** no ``W_in`` support on any PN row, and the PNs that receive signal do so
  through real ORN->PN edges of the substrate;
* **C1.6** the mapping is a named, declared object with a reproducible digest.

The vocabulary tests read the COMMITTED ``olfactory_v1_node_meta.csv``, so they run
without the data root. Anything needing the 775 MB adjacency or the 858 MB
classified edge metadata is measured on the GPU box by
``ops/audit/v2_construct_check.py`` instead.
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

from drososense.reservoir.connectome_reservoir import (  # noqa: E402
    R0RealFlyReservoir,
    ReservoirTopology,
    make_shared,
    rescale_to_spectral_radius,
    spectral_radius,
)
from drososense.reservoir.input_mapping import (  # noqa: E402
    INPUT_MAPPING_DENSE_RANDOM,
    INPUT_MAPPING_ORN_ALIGNED,
    INPUT_MAPPING_PN_DIRECT_ABLATION,
    ORN_DENSITY_LIMIT,
    InputMappingError,
    annotation_from_mapping,
    build_orn_aligned_mapping,
    compare_annotations,
    dense_random_mapping,
    in_substrate_orn_to_pn_edges,
    load_declared_annotation,
    sparse_sha256,
)

NODE_META = PROJECT_ROOT / "connectome" / "metadata" / "olfactory_v1_node_meta.csv"


# ---------------------------------------------------------------------------
# Fixtures: a synthetic substrate typed from the real vocabulary
# ---------------------------------------------------------------------------
def _roots_by_class(want: dict[str, int]) -> list[int]:
    """Real root ids, drawn from the committed vocabulary, per class."""
    import pandas as pd

    from connectome.cell_types import layer_to_class

    frame = pd.read_csv(NODE_META)
    frame["cell_class"] = frame["layer_mean"].map(layer_to_class)
    picked: list[int] = []
    for cell_class, count in want.items():
        ids = frame.loc[frame["cell_class"] == cell_class, "root_id"].tolist()[:count]
        assert len(ids) == count, f"only {len(ids)} {cell_class} nodes available"
        picked.extend(int(i) for i in ids)
    return picked


def _synthetic_adjacency(n: int, *, seed: int = 5, density: float = 0.08) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    matrix = (rng.random((n, n)) < density).astype(float)
    np.fill_diagonal(matrix, 0.0)
    return sp.csr_matrix(matrix)


@pytest.fixture(scope="module")
def declared_annotation():
    return load_declared_annotation(NODE_META)


@pytest.fixture()
def typed_substrate(declared_annotation):
    """60 nodes, 18 ORN / 18 PN / 24 other, with an ORN->PN block present."""
    roots = _roots_by_class({"ORN": 18, "PN": 18, "other": 24})
    n = len(roots)
    classes = declared_annotation.class_of(roots)
    adjacency = _synthetic_adjacency(n, seed=11, density=0.0).tolil()
    orn = np.flatnonzero(classes == "ORN")
    pn = np.flatnonzero(classes == "PN")
    # every PN receives a real edge from the FIRST orn only: the other ORNs have no
    # PN target at all, which is what C1.5's second half has to notice.
    for row in orn[:1]:
        for column in pn:
            adjacency[row, column] = 1.0
    return sp.csr_matrix(adjacency), np.asarray(roots), declared_annotation


# ---------------------------------------------------------------------------
# C1.1 -- the receiving set is the ORN population, exactly
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c1_1_every_node_receiving_input_is_an_orn_and_nothing_else(typed_substrate):
    adjacency, roots, annotation = typed_substrate
    mapping = build_orn_aligned_mapping(
        adjacency, roots, annotation, 4, seed=3
    )
    assert mapping.name == INPUT_MAPPING_ORN_ALIGNED
    assert mapping.support_is_orn_only()
    assert mapping.support_rows.tolist() == mapping.orn_rows.tolist()
    # the receiving set is an exact equality against the declared population, so a
    # single wrongly-typed node has to be visible rather than absorbed
    assert set(mapping.support_rows.tolist()) == set(
        np.flatnonzero(annotation.class_of(roots) == "ORN").tolist()
    )
    # and no non-ORN row carries a single non-zero
    dense = mapping.w_in.toarray()
    non_orn = np.setdiff1d(np.arange(len(roots)), mapping.orn_rows)
    assert np.all(dense[non_orn] == 0.0)


@pytest.mark.unit
def test_c1_1_a_node_with_no_declared_type_can_never_become_an_orn():
    """An untyped node is not an ORN by omission."""
    annotation = annotation_from_mapping({1: "ORN", 2: "ORN", 3: "PN"})
    roots = [1, 2, 3, 4]  # node 4 is not named
    adjacency = _synthetic_adjacency(4, density=0.0)
    mapping = build_orn_aligned_mapping(adjacency, roots, annotation, 2, seed=0)
    assert mapping.orn_rows.tolist() == [0, 1]
    assert "4" not in mapping.annotation.classes
    assert mapping.support_rows.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# C1.2 -- density, and what it really constrains
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c1_2_density_is_the_orn_fraction_over_the_channel_count(typed_substrate):
    adjacency, roots, annotation = typed_substrate
    mapping = build_orn_aligned_mapping(adjacency, roots, annotation, 4, seed=1)
    assert mapping.w_in.nnz == mapping.orn_rows.size, "one non-zero per ORN"
    assert mapping.density() == pytest.approx(
        mapping.orn_rows.size / (mapping.n_nodes * 4)
    )
    assert mapping.density() == pytest.approx(mapping.orn_fraction() / 4)
    assert mapping.passes_density()
    assert mapping.orn_fraction() <= mapping.max_orn_fraction()


@pytest.mark.unit
def test_c1_2_fails_exactly_when_the_orn_fraction_exceeds_the_bound(typed_substrate):
    """The criterion is a constraint on the SUBSTRATE, not a tuning knob.

    18 ORN in 60 nodes is 30 %. At Din = 4 the bound is 40 %, so it passes; at
    Din = 3 the bound is 30 % and the density sits exactly ON the limit, which
    passes too (``<=``); at Din = 2 the bound is 20 % and it fails. The mapping
    reports that rather than thinning its own support.
    """
    adjacency, roots, annotation = typed_substrate
    passes = build_orn_aligned_mapping(adjacency, roots, annotation, 4, seed=1)
    assert passes.passes_density()
    assert passes.max_orn_fraction() == pytest.approx(0.4)
    assert passes.orn_fraction() == pytest.approx(0.3)

    # exactly at the bound: still admissible, and the equality is the boundary
    on_the_limit = build_orn_aligned_mapping(adjacency, roots, annotation, 3, seed=1)
    assert on_the_limit.density() == pytest.approx(ORN_DENSITY_LIMIT)
    assert on_the_limit.passes_density()

    fails = build_orn_aligned_mapping(adjacency, roots, annotation, 2, seed=1)
    assert fails.density() == pytest.approx(0.15)
    assert fails.density() > ORN_DENSITY_LIMIT
    assert not fails.passes_density()
    assert fails.orn_fraction() > fails.max_orn_fraction()
    # the support is NOT thinned to satisfy the criterion: the ORN population is
    # what the substrate has, and satisfying C1.2 is C2's job
    assert fails.support_rows.size == fails.orn_rows.size


@pytest.mark.unit
def test_c1_2_the_v1_dense_mapping_is_at_density_one():
    """The defect the criterion exists to forbid, stated numerically."""
    dense = dense_random_mapping(50, 4, seed=0)
    assert dense.density() == pytest.approx(1.0)
    assert not dense.passes_density()
    assert dense.name == INPUT_MAPPING_DENSE_RANDOM


# ---------------------------------------------------------------------------
# C1.3 -- the input population's identity must move the dynamics
# ---------------------------------------------------------------------------
def _drive_once(topology: ReservoirTopology, shared, x: np.ndarray) -> np.ndarray:
    model = R0RealFlyReservoir(
        task="classification",
        seed=0,
        params={"reservoir_size": topology.n_nodes, "washout": 0},
        topology=topology,
        shared=shared,
        n_channels=x.shape[-1],
    )
    return model._drive(x)


def _topology(matrix: sp.csr_matrix) -> ReservoirTopology:
    return ReservoirTopology(
        matrix=rescale_to_spectral_radius(matrix, 0.9),
        n_nodes=matrix.shape[0],
        n_edges=int(matrix.nnz),
        spectral_radius=spectral_radius(rescale_to_spectral_radius(matrix, 0.9), seed=0),
        density=matrix.nnz / (matrix.shape[0] ** 2),
        kind="R0_real_fly",
        normalization="synthetic_c1",
    )


@pytest.mark.unit
def test_c1_3_reassigning_the_input_population_moves_the_states(declared_annotation):
    """The strict inverse of the v1 measurement (``max||dh|| = 0``).

    Graph FIXED, substrate FIXED: only WHICH nodes are ORNs changes -- the same
    experiment A6 ran on v1 (``w[perm, :]``), where the dynamics came out
    bit-identical. Under the ORN-aligned mapping the states must differ.
    """
    roots = _roots_by_class({"ORN": 6, "PN": 6, "other": 6})
    n = len(roots)
    rng = np.random.default_rng(4)
    raw = sp.csr_matrix((rng.random((n, n)) < 0.15).astype(float))
    raw = sp.csr_matrix(raw - sp.diags(raw.diagonal()))
    x = rng.standard_normal((4, 6, 3))
    topology = _topology(raw)

    # Two declarations over the SAME nodes and the SAME graph. In the first the
    # first six nodes are the ORNs; in the second the LAST six are, and the first
    # six become PNs. Same substrate, different input population.
    first_population = annotation_from_mapping(
        {**{int(r): "ORN" for r in roots[:6]}, **{int(r): "PN" for r in roots[6:]}}
    )
    second_population = annotation_from_mapping(
        {**{int(r): "PN" for r in roots[:6]}, **{int(r): "ORN" for r in roots[6:]}}
    )

    states = []
    for annotation in (first_population, second_population):
        mapping = build_orn_aligned_mapping(raw, roots, annotation, 3, seed=0)
        shared = make_shared(n, 3, seed=0, mapping=mapping)
        states.append(_drive_once(topology, shared, x))
    delta = float(np.max(np.abs(states[0] - states[1])))
    assert delta > 0.0, "C1.3 FAILS: the input population does not move the states"

    # The v1 pathway over the same two declarations: the mapping does not look at
    # the annotation at all, so the states are bit-identical. That is the
    # measurement C1.3 exists to invert, reproduced here rather than quoted.
    v1 = [
        _drive_once(_topology(raw), make_shared(n, 3, seed=0), x)
        for _ in range(2)
    ]
    assert np.array_equal(v1[0], v1[1]), "the dense v1 map is annotation-blind"
    assert not np.array_equal(states[0], states[1])


# ---------------------------------------------------------------------------
# C1.5 -- PN receive only through real ORN->PN edges
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c1_5_no_w_in_support_on_pn_rows(typed_substrate):
    adjacency, roots, annotation = typed_substrate
    mapping = build_orn_aligned_mapping(adjacency, roots, annotation, 4, seed=2)
    assert mapping.n_support_rows_on_pn() == 0
    assert np.all(mapping.w_in.toarray()[mapping.pn_rows] == 0.0)


@pytest.mark.unit
def test_c1_5_the_receiving_pns_are_the_ones_a_real_orn_edge_reaches(typed_substrate):
    adjacency, roots, annotation = typed_substrate
    mapping = build_orn_aligned_mapping(adjacency, roots, annotation, 4, seed=2)
    # the fixture wires ONE ORN to every PN, so all PNs are reached
    assert mapping.orn_recipient_pn_rows.tolist() == mapping.pn_rows.tolist()
    assert mapping.n_orn_to_pn_edges == mapping.pn_rows.size

    # a substrate where the ORN->PN block is absent: no PN may be called receiving
    empty = sp.csr_matrix((mapping.n_nodes, mapping.n_nodes))
    starved = build_orn_aligned_mapping(empty, roots, annotation, 4, seed=2)
    assert starved.orn_recipient_pn_rows.size == 0
    assert starved.n_orn_to_pn_edges == 0
    assert starved.w_in.nnz == starved.orn_rows.size, "ORNs still receive input"


@pytest.mark.unit
def test_c1_5_the_pn_direct_ablation_targets_only_reached_pns(typed_substrate):
    adjacency, roots, annotation = typed_substrate
    ablation = build_orn_aligned_mapping(
        adjacency, roots, annotation, 4, seed=2, ablation=True
    )
    assert ablation.name == INPUT_MAPPING_PN_DIRECT_ABLATION
    assert ablation.ablation is True
    assert set(ablation.support_rows.tolist()) == set(
        ablation.orn_recipient_pn_rows.tolist()
    )
    assert set(ablation.orn_rows.tolist()) & set(ablation.support_rows.tolist()) == set()
    assert ablation.describe()["ablation"] is True

    # and it refuses a substrate the connectome does not feed
    empty = sp.csr_matrix((ablation.n_nodes, ablation.n_nodes))
    with pytest.raises(InputMappingError, match="none receives a real ORN edge"):
        build_orn_aligned_mapping(empty, roots, annotation, 4, seed=2, ablation=True)


@pytest.mark.unit
def test_c1_5_orn_to_pn_edge_measurement_matches_the_block():
    matrix = sp.csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float))
    n_edges, reached = in_substrate_orn_to_pn_edges(
        matrix, np.array([0]), np.array([1, 2])
    )
    assert n_edges == 1
    assert reached.tolist() == [1]
    assert in_substrate_orn_to_pn_edges(matrix, np.array([2]), np.array([1]))[0] == 0


# ---------------------------------------------------------------------------
# C1.6 -- the declared object, and its digest
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_c1_6_the_mapping_is_declared_on_the_shared_components(typed_substrate):
    adjacency, roots, annotation = typed_substrate
    mapping = build_orn_aligned_mapping(adjacency, roots, annotation, 4, seed=6)
    shared = make_shared(mapping.n_nodes, 4, seed=6, mapping=mapping)
    described = shared.describe()
    assert described["input_mapping"] == INPUT_MAPPING_ORN_ALIGNED
    population = described["input_population"]
    assert population["input_mapping"] == INPUT_MAPPING_ORN_ALIGNED
    assert population["nnz"] == int(mapping.w_in.nnz)
    assert population["density"] == pytest.approx(mapping.density())
    assert population["declared_input_population"] == "ORN"
    assert population["support_is_orn_only"] is True
    assert population["receptor_exact"] is False
    assert population["annotation"]["mapping_sha256"] == annotation.digest()


@pytest.mark.unit
def test_c1_6_the_v1_shared_block_keeps_its_exact_shape():
    """A v1 record must not gain fields it was not written with."""
    described = make_shared(20, 4, seed=1).describe()
    assert described["input_mapping"] == INPUT_MAPPING_DENSE_RANDOM
    assert "input_population" not in described


@pytest.mark.unit
def test_c1_6_a_sparse_w_in_digest_is_not_a_pointer_hash():
    """``np.ascontiguousarray(csr).tobytes()`` hashes 8 bytes of pointer."""
    left = dense_random_mapping(30, 3, seed=5).w_in
    right = dense_random_mapping(30, 3, seed=5).w_in
    other = dense_random_mapping(30, 3, seed=6).w_in
    assert sparse_sha256(left) == sparse_sha256(right)
    assert sparse_sha256(left) != sparse_sha256(other)
    assert len(np.ascontiguousarray(left).tobytes()) == 8, "the dense way is a pointer"

    # and the shared-components hash agrees with it
    assert make_shared(30, 3, seed=5, mapping=dense_random_mapping(30, 3, seed=5)).describe()[
        "w_in_sha256"
    ] == sparse_sha256(left)


@pytest.mark.unit
def test_c1_6_a_mapping_of_the_wrong_shape_is_refused(typed_substrate):
    adjacency, roots, annotation = typed_substrate
    mapping = build_orn_aligned_mapping(adjacency, roots, annotation, 4, seed=6)
    with pytest.raises(ValueError, match="has shape"):
        make_shared(mapping.n_nodes + 1, 4, seed=0, mapping=mapping)
    with pytest.raises(ValueError, match="has shape"):
        make_shared(mapping.n_nodes, 5, seed=0, mapping=mapping)


# ---------------------------------------------------------------------------
# The refusals: no silent fallback, each with its reason
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_substrate_without_orns_is_refused_not_densified(declared_annotation):
    roots = _roots_by_class({"PN": 10, "other": 10})
    adjacency = _synthetic_adjacency(len(roots), density=0.05)
    with pytest.raises(InputMappingError, match="no ORN row"):
        build_orn_aligned_mapping(adjacency, roots, declared_annotation, 3, seed=0)


@pytest.mark.unit
def test_a_substrate_with_fewer_orns_than_channels_is_refused(declared_annotation):
    roots = _roots_by_class({"ORN": 2, "PN": 10})
    adjacency = _synthetic_adjacency(len(roots), density=0.05)
    with pytest.raises(InputMappingError, match="one channel per ORN"):
        build_orn_aligned_mapping(adjacency, roots, declared_annotation, 3, seed=0)


@pytest.mark.unit
def test_a_substrate_without_pns_is_refused(declared_annotation):
    roots = _roots_by_class({"ORN": 10})
    adjacency = _synthetic_adjacency(len(roots), density=0.05)
    with pytest.raises(InputMappingError, match="no PN row"):
        build_orn_aligned_mapping(adjacency, roots, declared_annotation, 3, seed=0)


@pytest.mark.unit
def test_root_ids_that_do_not_match_the_graph_are_refused(typed_substrate):
    adjacency, roots, annotation = typed_substrate
    with pytest.raises(InputMappingError, match="cannot be attributed to substrate rows"):
        build_orn_aligned_mapping(adjacency, roots[:-1], annotation, 3, seed=0)


# ---------------------------------------------------------------------------
# The declared vocabulary itself
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_declared_vocabulary_covers_every_node_exactly_once(declared_annotation):
    counts = declared_annotation.per_class_counts
    assert sum(counts.values()) == 124_185, "the committed node metadata has one row per node"
    assert counts == {
        "other": 97558,
        "MBON": 9650,
        "PN": 5355,
        "KC": 3613,
        "higher_order": 3170,
        "DAN": 2572,
        "ORN": 2267,
    }


@pytest.mark.unit
def test_root_ids_survive_the_read_as_integers():
    """Root ids are 60-bit; a float round-trip collapses distinct neurons.

    Measured before this test existed: parsing through ``float`` merged 124,185
    neurons into 100,984 keys.
    """
    classes = load_declared_annotation(NODE_META)
    assert len(classes.classes) == 124_185
    sample = sorted(classes.classes)[:3]
    assert all(len(key) >= 17 and key.isdigit() for key in sample)
    # two ids that differ only beyond float64's 53-bit mantissa must stay distinct
    assert len({int(720575940596125868), int(720575940596125869)}) == 2


@pytest.mark.unit
def test_the_two_vocabularies_are_compared_rather_than_conflated(declared_annotation):
    """The delivered classified table disagrees with the declared tiers.

    The comparison is the artefact: the report publishes it, and a criterion that
    names "the declared ORN population" names this table.
    """
    other = annotation_from_mapping(
        {k: v for k, v in list(declared_annotation.classes.items())[:100]},
        source="fixture",
    )
    report = compare_annotations(declared_annotation, other)
    assert report["n_shared_nodes"] == 100
    assert report["left_per_class_counts"]["ORN"] == 2267
    assert report["left_mapping_sha256"] == declared_annotation.digest()
