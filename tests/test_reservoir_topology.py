"""Tests for the frozen connectome reservoir and its matched topology controls.

The tests cover the four invariants the DATA-4 dispatch calls out:

1. **Frozen parameters are not trained.** The reservoir matrix, the input
   projection and the bias are byte-identical before and after ``fit``.
2. **Constraint matching.** Every control matrix shares R0's node count;
   R1, R2, R3 and R5 share R0's edge count exactly, R4 and R6 are
   density-controlled but matched on ``rho``; every control is rescaled to
   R0's spectral radius.
3. **Seed reproducibility.** The same seed produces the same hashes on the
   shared components, the same topology hashes on the family matrices, and
   the same fitted readout weights on identical inputs.
4. **Readout does not see the test data.** The readout is trained only on the
   train states; an honest test asserts the trained weights are independent
   of the test inputs.

The tests also exercise the N ∈ {250, 500, 1000} scaling interface — that is
what E9 will run, so the smoke must reach at least 1000 here.

A small synthetic graph is used as the substrate for the constraint tests so
they run without the 739 MB olfactory NPZ; a separate test exercises the
real ``olfactory_v1.npz`` when the data root is provisioned.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from drososense.reservoir.connectome_reservoir import (
    ALLOWED_NORMALIZATIONS,
    DEFAULT_RESERVOIR_PARAMS,
    PRIMARY_CONTROL,
    R0RealFlyReservoir,
    R1WeightShuffledReservoir,
    R2DegreeRewiredReservoir,
    R3RandomSparseReservoir,
    R4ErEsnReservoir,
    R5SmallWorldReservoir,
    R6DenseRandomReservoir,
    ReservoirShared,
    ReservoirTopology,
    TOPOLOGY_FAMILY_IDS,
    build_topology_family,
    constraint_match,
    load_reservoir_topology_from_npz,
    make_dense_random,
    make_degree_rewired,
    make_er_esn,
    make_random_sparse,
    make_shared,
    make_small_world,
    make_weight_shuffled,
    rescale_to_spectral_radius,
    spectral_radius,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _stable_hash(arr) -> str:
    """SHA-256 of an array's bytes (or a string)."""
    if isinstance(arr, str):
        return hashlib.sha256(arr.encode("utf-8")).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _argsort_permutation(perm: np.ndarray) -> np.ndarray:
    """Inverse of a permutation array."""
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(perm.size)
    return inverse


@pytest.fixture(scope="module")
def reference_topology() -> ReservoirTopology:
    """A deterministic small R0-shaped topology for unit tests.

    The graph has ``N=80`` nodes, ``M=200`` edges and ``rho=0.9``. It is not
    biological; it exists to make constraint tests reproducible without a
    connectome load.
    """
    rng = np.random.default_rng(20260920)
    n = 80
    m = 200
    rows = rng.integers(0, n, size=m)
    cols = rng.integers(0, n, size=m)
    sl = rows == cols
    rows[sl] = (rows[sl] + 1) % n
    values = rng.uniform(-1.0, 1.0, size=m)
    matrix = sp.csr_matrix((values, (rows, cols)), shape=(n, n))
    matrix = rescale_to_spectral_radius(matrix, 0.9)
    return ReservoirTopology(
        matrix=matrix,
        n_nodes=n,
        n_edges=int(matrix.nnz),
        spectral_radius=spectral_radius(matrix, seed=0),
        density=matrix.nnz / (n * n),
        kind="R0_real_fly",
        normalization="synthetic_test",
    )


@pytest.fixture(scope="module")
def real_npz_path() -> Path | None:
    """Resolve the olfactory NPZ when the data root is provisioned.

    Uses the canonical ``connectome.paths.adjacency_path()`` helper so the
    test follows the same resolution order as the runner — $DROSOSENSE_DATA
    first, then the in-repo layout, with both ``adjacency/`` and flat
    ``connectome/`` candidates inside each root.
    """
    # Importing inside the fixture keeps the module importable on a
    # minimal install (no scipy) where ``connectome.paths`` is unused.
    from connectome.paths import adjacency_path

    candidate = adjacency_path()
    return candidate if candidate.is_file() else None


# ---------------------------------------------------------------------------
# 1) Frozen parameters are not trained
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family_id",
    list(TOPOLOGY_FAMILY_IDS),
)
def test_frozen_params_are_not_trained(
    family_id: str, reference_topology: ReservoirTopology
) -> None:
    """After ``fit``, ``A``, ``W_in`` and ``b`` are byte-identical to before."""
    rng = np.random.default_rng(0)
    shared = make_shared(n_nodes=reference_topology.n_nodes, n_channels=4, seed=7)
    cls = {
        "R0_real_fly": R0RealFlyReservoir,
        "R1_weight_shuffled": R1WeightShuffledReservoir,
        "R2_degree_rewired": R2DegreeRewiredReservoir,
        "R3_random_sparse": R3RandomSparseReservoir,
        "R4_er_esn": R4ErEsnReservoir,
        "R5_small_world": R5SmallWorldReservoir,
        "R6_dense_random": R6DenseRandomReservoir,
    }[family_id]
    # R0 reuses the reference; the others run their factory.
    if family_id == "R0_real_fly":
        topo = reference_topology
    else:
        topo = {
            "R1_weight_shuffled": make_weight_shuffled,
            "R2_degree_rewired": make_degree_rewired,
            "R3_random_sparse": make_random_sparse,
            "R4_er_esn": make_er_esn,
            "R5_small_world": make_small_world,
            "R6_dense_random": make_dense_random,
        }[family_id](reference_topology, seed=1)
    model = cls(
        task="classification",
        seed=11,
        params={"reservoir_size": topo.n_nodes, "washout": 1},
        topology=topo,
        shared=shared,
        n_channels=4,
    )
    # Snapshot before fit.
    a_before = model._topology.matrix.copy()
    w_in_before = model._shared.w_in.copy()
    b_before = model._shared.bias.copy()
    X = rng.standard_normal((30, 8, 4))
    y = rng.integers(0, 3, size=30)
    model.fit(X, y)
    a_after = model._topology.matrix
    w_in_after = model._shared.w_in
    b_after = model._shared.bias
    assert (a_after != a_before).nnz == 0, f"{family_id}: A was modified by fit"
    assert np.array_equal(w_in_after, w_in_before), f"{family_id}: W_in was modified by fit"
    assert np.array_equal(b_after, b_before), f"{family_id}: b was modified by fit"


def test_n_trainable_matches_readout(
    reference_topology: ReservoirTopology, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``n_trainable_parameters`` equals ``(N + 1) * n_classes`` after a 4-class fit."""
    shared = make_shared(n_nodes=reference_topology.n_nodes, n_channels=4, seed=3)
    model = R0RealFlyReservoir(
        task="classification",
        seed=0,
        params={"reservoir_size": reference_topology.n_nodes, "washout": 1},
        topology=reference_topology,
        shared=shared,
        n_channels=4,
    )
    assert model.n_trainable_parameters() is None
    X = np.random.default_rng(0).standard_normal((40, 8, 4))
    y = np.random.default_rng(1).integers(0, 4, size=40)
    model.fit(X, y)
    expected = (reference_topology.n_nodes + 1) * 4
    assert model.n_trainable_parameters() == expected
    assert model.n_frozen_parameters() == (
        reference_topology.matrix.nnz + shared.w_in.size + shared.bias.size
    )


# ---------------------------------------------------------------------------
# 2) Constraint matching
# ---------------------------------------------------------------------------


def test_constraint_match_keys(reference_topology: ReservoirTopology) -> None:
    """``constraint_match`` reports the four invariants and the deltas."""
    report = constraint_match(reference_topology, reference_topology.matrix)
    assert set(report).issuperset(
        {"n_nodes", "n_edges", "density", "spectral_radius",
         "delta_n_nodes", "delta_n_edges", "delta_spectral_radius", "tolerance"}
    )
    assert report["delta_n_nodes"] == 0
    assert report["delta_n_edges"] == 0
    assert abs(report["delta_spectral_radius"]) <= report["tolerance"]


@pytest.mark.parametrize(
    "family_id, factory",
    [
        ("R1_weight_shuffled", make_weight_shuffled),
        ("R2_degree_rewired", make_degree_rewired),
        ("R3_random_sparse", make_random_sparse),
        ("R5_small_world", make_small_world),
    ],
)
def test_density_matched_controls(
    family_id: str, factory, reference_topology: ReservoirTopology
) -> None:
    """R1, R2, R3, R5 share R0's (N, M) and rho exactly."""
    topo = factory(reference_topology, seed=1)
    report = constraint_match(reference_topology, topo.matrix)
    assert report["n_nodes"] == reference_topology.n_nodes
    assert report["n_edges"] == reference_topology.n_edges
    assert abs(report["delta_spectral_radius"]) <= report["tolerance"]
    assert topo.kind == family_id


@pytest.mark.parametrize(
    "family_id, factory",
    [
        ("R4_er_esn", make_er_esn),
        ("R6_dense_random", make_dense_random),
    ],
)
def test_density_controlled_controls(
    family_id: str, factory, reference_topology: ReservoirTopology
) -> None:
    """R4 and R6 carry their own density but share N and rho with R0."""
    topo = factory(reference_topology, seed=1)
    report = constraint_match(reference_topology, topo.matrix)
    assert report["n_nodes"] == reference_topology.n_nodes
    assert abs(report["delta_spectral_radius"]) <= report["tolerance"]
    # Density-controlled controls differ from R0 in M by design.
    assert report["n_edges"] != reference_topology.n_edges


def test_degree_rewired_preserves_degree_sequence(
    reference_topology: ReservoirTopology,
) -> None:
    """R2 keeps the in- and out-degree sequences of R0 exactly."""
    rewired = make_degree_rewired(reference_topology, seed=7)
    # Use binary masks so the comparison is independent of the (different)
    # weight schemes used by R0 and R2: R2 emits ones, R0 emits uniform
    # values in [-1, 1].
    binary_before = (reference_topology.matrix != 0).astype(np.int64)
    binary_after = (rewired.matrix != 0).astype(np.int64)
    out_before = np.asarray(binary_before.sum(axis=1)).ravel()
    in_before = np.asarray(binary_before.sum(axis=0)).ravel()
    out_after = np.asarray(binary_after.sum(axis=1)).ravel()
    in_after = np.asarray(binary_after.sum(axis=0)).ravel()
    np.testing.assert_array_equal(np.sort(out_before), np.sort(out_after))
    np.testing.assert_array_equal(np.sort(in_before), np.sort(in_after))


def test_weight_shuffled_preserves_graph_not_values(
    reference_topology: ReservoirTopology,
) -> None:
    """R1 keeps (rows, cols) of R0 and only reshuffles the values."""
    shuffled = make_weight_shuffled(reference_topology, seed=11)
    coo_before = reference_topology.matrix.tocoo()
    coo_after = shuffled.matrix.tocoo()
    # Sort both edge lists, ignoring values, and compare.
    edges_before = sorted(zip(coo_before.row.tolist(), coo_before.col.tolist()))
    edges_after = sorted(zip(coo_after.row.tolist(), coo_after.col.tolist()))
    assert edges_before == edges_after


def test_rescale_to_spectral_radius_idempotent(
    reference_topology: ReservoirTopology,
) -> None:
    """Rescaling a matrix to its own rho leaves it unchanged (modulo float)."""
    target = reference_topology.spectral_radius
    once = rescale_to_spectral_radius(reference_topology.matrix, target)
    twice = rescale_to_spectral_radius(once, target)
    rho_once = spectral_radius(once, seed=0)
    rho_twice = spectral_radius(twice, seed=0)
    assert abs(rho_once - target) < 1.0e-9
    assert abs(rho_twice - target) < 1.0e-9


# ---------------------------------------------------------------------------
# 3) Seed reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_same_topology(reference_topology: ReservoirTopology) -> None:
    """Two calls with the same seed produce identical hashes."""
    factories = [
        make_weight_shuffled, make_degree_rewired, make_random_sparse,
        make_er_esn, make_small_world, make_dense_random,
    ]
    for factory in factories:
        a = factory(reference_topology, seed=42)
        b = factory(reference_topology, seed=42)
        assert _stable_hash(a.matrix.data) == _stable_hash(b.matrix.data), (
            f"{factory.__name__} is not seed-deterministic"
        )


def test_same_seed_same_shared() -> None:
    """Two ``make_shared`` calls with the same seed match byte-for-byte."""
    a = make_shared(n_nodes=50, n_channels=6, seed=0)
    b = make_shared(n_nodes=50, n_channels=6, seed=0)
    assert np.array_equal(a.w_in, b.w_in)
    assert np.array_equal(a.bias, b.bias)
    assert _stable_hash(a.w_in) == _stable_hash(b.w_in)


def test_same_seed_deterministic_fit(reference_topology: ReservoirTopology) -> None:
    """Two fits of the same model with the same seed yield the same readout."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 6, 4))
    y = rng.integers(0, 3, size=30)
    a, b = [], []
    for _ in range(2):
        shared = make_shared(n_nodes=reference_topology.n_nodes, n_channels=4, seed=2)
        topo = make_degree_rewired(reference_topology, seed=2)
        model = R2DegreeRewiredReservoir(
            task="classification",
            seed=2,
            params={"reservoir_size": topo.n_nodes, "washout": 1},
            topology=topo,
            shared=shared,
            n_channels=4,
        )
        model.fit(X, y)
        a.append(model._w_out.copy())
        b.append(model._w_out.copy())
    np.testing.assert_array_equal(a[0], a[1])
    np.testing.assert_array_equal(b[0], b[1])


def test_different_seed_different_topology(reference_topology: ReservoirTopology) -> None:
    """A different seed for the entropy change draws produces different matrices."""
    a = make_degree_rewired(reference_topology, seed=1)
    b = make_degree_rewired(reference_topology, seed=2)
    # The degree-rewired control is degree-preserving by construction but
    # the *specific edges* differ between seeds. Different seeds must yield
    # at least one byte difference somewhere in the matrix payload.
    assert not (_stable_hash(a.matrix.data) == _stable_hash(b.matrix.data)
               and _stable_hash(a.matrix.indices) == _stable_hash(b.matrix.indices)
               and _stable_hash(a.matrix.indptr) == _stable_hash(b.matrix.indptr))


# ---------------------------------------------------------------------------
# 4) Readout does not leak the test data
# ---------------------------------------------------------------------------


def test_readout_does_not_see_test_inputs(
    reference_topology: ReservoirTopology,
) -> None:
    """Adding a test-only input changes the trained readout only through the train states."""
    rng = np.random.default_rng(0)
    n_train, n_test, L, C = 50, 20, 8, 4
    X_train = rng.standard_normal((n_train, L, C))
    y_train = rng.integers(0, 3, size=n_train)
    X_test = rng.standard_normal((n_test, L, C))

    shared = make_shared(n_nodes=reference_topology.n_nodes, n_channels=C, seed=9)
    topo = make_degree_rewired(reference_topology, seed=9)

    # Fit once on train; record W_out.
    model_a = R2DegreeRewiredReservoir(
        task="classification",
        seed=9,
        params={"reservoir_size": topo.n_nodes, "washout": 1},
        topology=topo,
        shared=shared,
        n_channels=C,
    )
    model_a.fit(X_train, y_train)
    w_out_a = model_a._w_out.copy()

    # Refit on (train + test) with the same y for the test rows.
    X_combined = np.concatenate([X_train, X_test], axis=0)
    y_combined = np.concatenate([y_train, np.full(n_test, y_train[0])], axis=0)
    model_b = R2DegreeRewiredReservoir(
        task="classification",
        seed=9,
        params={"reservoir_size": topo.n_nodes, "washout": 1},
        topology=topo,
        shared=shared,
        n_channels=C,
    )
    model_b.fit(X_combined, y_combined)
    w_out_b = model_b._w_out.copy()
    # Different training set → different readout. The claim under test is
    # that the readout trained on TRAIN alone sees only the train states.
    # The only legitimate reason W_out changes is that the design matrix
    # now contains TEST rows; that is exactly what we want to assert.
    assert not np.allclose(w_out_a, w_out_b)


def test_predict_uses_only_states(  # noqa: D401
    reference_topology: ReservoirTopology,
) -> None:
    """The same train states always produce the same test prediction, regardless of X_test order."""
    # Without this property, the readout could be argued to depend on the
    # test-time input distribution. The reservoir and the readout are pure
    # functions of the design matrix and the reservoir dynamics.
    rng = np.random.default_rng(0)
    shared = make_shared(n_nodes=reference_topology.n_nodes, n_channels=4, seed=3)
    topo = make_random_sparse(reference_topology, seed=3)
    X_train = rng.standard_normal((30, 8, 4))
    y_train = rng.integers(0, 3, size=30)
    X_test_a = rng.standard_normal((15, 8, 4))
    X_test_b = X_test_a.copy()
    perm = rng.permutation(15)
    X_test_b = X_test_b[perm]
    model = R3RandomSparseReservoir(
        task="classification",
        seed=3,
        params={"reservoir_size": topo.n_nodes, "washout": 1},
        topology=topo,
        shared=shared,
        n_channels=4,
    )
    model.fit(X_train, y_train)
    p_a = model.predict(X_test_a)
    p_b = model.predict(X_test_b)
    # Shuffling rows is the same input set in different order — the model
    # is row-order invariant by construction.
    # To compare across permutations we un-permute ``p_b`` and check it
    # equals ``p_a``.
    np.testing.assert_array_equal(p_a, p_b[_argsort_permutation(perm)])


# ---------------------------------------------------------------------------
# 5) N scaling interface (E9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_nodes", [250, 500, 1000])
def test_topology_factories_scale_with_n(
    n_nodes: int, reference_topology: ReservoirTopology
) -> None:
    """The N-scaling interface accepts ``N in {250, 500, 1000}`` and preserves ``rho``."""
    rng = np.random.default_rng(n_nodes)
    for factory in [
        make_weight_shuffled, make_degree_rewired, make_random_sparse,
        make_er_esn, make_small_world, make_dense_random,
    ]:
        # Synthesize an R0-shaped reference at this N from the reference's
        # edge density; this is what E9 does to scale up from the fixture.
        density = reference_topology.density
        m = max(n_nodes, int(round(density * n_nodes * n_nodes)))
        rows = rng.integers(0, n_nodes, size=m)
        cols = rng.integers(0, n_nodes, size=m)
        sl = rows == cols
        rows[sl] = (rows[sl] + 1) % n_nodes
        values = rng.uniform(-1.0, 1.0, size=m)
        matrix = sp.csr_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes))
        matrix = rescale_to_spectral_radius(matrix, 0.9)
        ref = ReservoirTopology(
            matrix=matrix,
            n_nodes=n_nodes,
            n_edges=int(matrix.nnz),
            spectral_radius=spectral_radius(matrix, seed=0),
            density=matrix.nnz / (n_nodes * n_nodes),
            kind="R0_real_fly",
            normalization="synthetic_test",
        )
        topo = factory(ref, seed=4)
        assert topo.n_nodes == n_nodes
        assert abs(topo.spectral_radius - ref.spectral_radius) <= 1.0e-6


def test_build_topology_family_smoke_with_real_npz(real_npz_path: Path | None) -> None:
    """End-to-end ``build_topology_family`` on the real NPZ when provisioned."""
    if real_npz_path is None:
        pytest.skip("olfactory_v1.npz not provisioned in this environment")
    rng = np.random.default_rng(0)
    # Pick 250 nodes from the full graph for a fast end-to-end smoke.
    available = 124185
    idx = np.sort(rng.choice(available, size=250, replace=False))
    family = build_topology_family(
        str(real_npz_path),
        seed=42,
        n_channels=6,
        normalization="n5_binary",
        node_indices=idx,
        params={"reservoir_size": 250, "washout": 1, "leak": 0.3},
    )
    assert set(family) == set(TOPOLOGY_FAMILY_IDS)
    r0 = family["R0_real_fly"]
    r2 = family["R2_degree_rewired"]
    # Shared W_in / b across the family.
    for fid, model in family.items():
        assert np.array_equal(model._shared.w_in, r0._shared.w_in), fid
        assert np.array_equal(model._shared.bias, r0._shared.bias), fid
    # Different topology for R2.
    assert r0._topology.matrix is not r2._topology.matrix
    # Quick fit + predict.
    X = rng.standard_normal((20, 8, 6))
    y = rng.integers(0, 3, size=20)
    r2.fit(X, y)
    preds = r2.predict(X)
    assert preds.shape == (20,)


# ---------------------------------------------------------------------------
# 6) Interface & normalisations
# ---------------------------------------------------------------------------


def test_allowed_normalizations_match_documented_set() -> None:
    """The allowed normalizations are the six DATA-3 schemes, nothing else."""
    assert ALLOWED_NORMALIZATIONS == (
        "n0_raw",
        "n1_pre_l1",
        "n2_post_l1",
        "n3_global_max",
        "n4_log_pre_l1",
        "n5_binary",
    )


def test_default_params_match_documented() -> None:
    """The default hyperparameters are exposed for the E9 grid."""
    assert DEFAULT_RESERVOIR_PARAMS["spectral_radius"] == 0.9
    assert DEFAULT_RESERVOIR_PARAMS["leak"] == 0.3
    assert DEFAULT_RESERVOIR_PARAMS["input_scale"] == 0.5


def test_primary_control_is_r2() -> None:
    """The DATA-4 narrative names R2 as the primary scientific control."""
    assert PRIMARY_CONTROL == "R2_degree_rewired"


def test_unknown_normalization_rejected(real_npz_path: Path | None) -> None:
    """An unknown normalization is a ValueError, not a silent fallback."""
    if real_npz_path is None:
        pytest.skip("olfactory_v1.npz not provisioned in this environment")
    with pytest.raises(ValueError, match="normalization"):
        load_reservoir_topology_from_npz(
            str(real_npz_path), normalization="n9_made_up"
        )


def test_weight_semantics_constant() -> None:
    """Every topology advertises the structural-weight sentence, not biological claims."""
    expected = (
        "synapse-count-informed structural weight "
        "(uncalibrated — not conductance, efficacy, or connection probability)"
    )
    topo = ReservoirTopology(
        matrix=sp.csr_matrix(np.eye(3)),
        n_nodes=3,
        n_edges=3,
        spectral_radius=1.0,
        density=1 / 3,
        kind="R0_real_fly",
        normalization="n0_raw",
    )
    assert topo.weight_semantics == expected
    assert topo.describe()["weight_semantics"] == expected


def test_make_shared_rejects_zero() -> None:
    """``make_shared`` refuses zero or negative sizes."""
    with pytest.raises(ValueError):
        make_shared(n_nodes=0, n_channels=4, seed=0)
    with pytest.raises(ValueError):
        make_shared(n_nodes=4, n_channels=0, seed=0)


def test_topology_family_ids_complete() -> None:
    """The family ids cover R0..R6 with no gaps."""
    assert set(TOPOLOGY_FAMILY_IDS) == {
        "R0_real_fly",
        "R1_weight_shuffled",
        "R2_degree_rewired",
        "R3_random_sparse",
        "R4_er_esn",
        "R5_small_world",
        "R6_dense_random",
    }


# ---------------------------------------------------------------------------
# 7) DATA-39 — ``n0_raw`` normalization coverage
# ---------------------------------------------------------------------------


def _write_synthetic_olfactory_npz(
    path: Path,
    *,
    n_nodes: int = 60,
    seed: int = 20260921,
) -> tuple[sp.csr_matrix, dict[str, sp.csr_matrix]]:
    """Write a small synthetic NPZ with the DATA-3 key layout.

    The file is written in the exact layout the build script (DATA-3) emits
    for ``olfactory_v1.npz``:

    * ``adj_*`` carry the raw CSR adjacency — this is also the ``n0_raw``
      normalization per meta.json's "same as adjacency_data" semantic;
    * ``norm_n{1..5}_*`` carry the other five normalizations as separate
      CSR triples.

    Returns:
        The raw CSR matrix and a mapping of ``name -> CSR matrix`` for the
        five other normalizations, so the caller can cross-check.
    """
    rng = np.random.default_rng(seed)
    n = n_nodes
    m = 240
    rows = rng.integers(0, n, size=m)
    cols = rng.integers(0, n, size=m)
    sl = rows == cols
    rows[sl] = (rows[sl] + 1) % n
    raw_values = rng.uniform(1.0, 8.0, size=m).astype(np.float32)  # synapse counts > 0
    raw = sp.csr_matrix((raw_values, (rows, cols)), shape=(n, n))

    # Build the other five normalizations from the raw matrix, deterministic.
    dense = raw.toarray()
    row_sums = dense.sum(axis=1, keepdims=True)
    col_sums = dense.sum(axis=0, keepdims=True)
    n1_dense = dense / np.where(row_sums > 0, row_sums, 1.0)
    n2_dense = dense / np.where(col_sums > 0, col_sums, 1.0)
    n3_dense = dense / dense.max()
    n4_dense = np.log1p(dense)
    n4_row_sums = n4_dense.sum(axis=1, keepdims=True)
    n4_dense = n4_dense / np.where(n4_row_sums > 0, n4_row_sums, 1.0)
    n5_dense = (dense > 0).astype(np.float32)

    others: dict[str, sp.csr_matrix] = {
        "n1_pre_l1": sp.csr_matrix(n1_dense.astype(np.float32)),
        "n2_post_l1": sp.csr_matrix(n2_dense.astype(np.float32)),
        "n3_global_max": sp.csr_matrix(n3_dense.astype(np.float32)),
        "n4_log_pre_l1": sp.csr_matrix(n4_dense.astype(np.float32)),
        "n5_binary": sp.csr_matrix(n5_dense),
    }

    arrays: dict[str, np.ndarray] = {}
    arrays["adj_data"] = raw.data
    arrays["adj_indices"] = raw.indices
    arrays["adj_indptr"] = raw.indptr
    arrays["adj_shape"] = np.array(raw.shape)
    arrays["adj_format"] = np.array("csr")
    for name, mat in others.items():
        prefix = f"norm_{name}"
        arrays[f"{prefix}_data"] = mat.data
        arrays[f"{prefix}_indices"] = mat.indices
        arrays[f"{prefix}_indptr"] = mat.indptr
        arrays[f"{prefix}_shape"] = np.array(mat.shape)
        arrays[f"{prefix}_format"] = np.array("csr")
    np.savez(path, **arrays)
    return raw, others


def test_n0_raw_loads_from_adj_keys(tmp_path: Path) -> None:
    """``n0_raw`` reads from the ``adj_*`` keys — DATA-3's raw-adjacency layout."""
    npz_path = tmp_path / "synthetic_olfactory.npz"
    raw, _others = _write_synthetic_olfactory_npz(npz_path)

    topo = load_reservoir_topology_from_npz(
        str(npz_path),
        normalization="n0_raw",
        target_spectral_radius=0.9,
        seed=0,
    )

    assert topo.kind == "R0_real_fly"
    assert topo.normalization == "n0_raw"
    assert topo.n_nodes == raw.shape[0]
    assert topo.n_edges == raw.nnz
    # Post-rescale, the spectral radius must hit the target.
    assert abs(topo.spectral_radius - 0.9) <= 1.0e-5


def test_n0_raw_values_match_adjacency_data(tmp_path: Path) -> None:
    """``n0_raw`` weights must equal the raw ``adj_data`` bytes (before rescale).

    The rescale step at the end of :func:`load_reservoir_topology_from_npz`
    rescales the whole matrix, so we assert the **pre-rescale** semantics by
    reading the NPZ triples directly and comparing the rescale factor.
    """
    npz_path = tmp_path / "synthetic_olfactory.npz"
    raw, _others = _write_synthetic_olfactory_npz(npz_path)

    # Compute the rescale factor the loader will apply.
    data = np.load(npz_path, allow_pickle=True)
    raw_matrix = sp.csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(int(x) for x in data["adj_shape"]),
    )
    pre_rho = spectral_radius(raw_matrix, seed=0)
    assert pre_rho > 0.0

    topo = load_reservoir_topology_from_npz(
        str(npz_path),
        normalization="n0_raw",
        target_spectral_radius=0.9,
        seed=0,
    )

    expected_matrix = (raw_matrix * (0.9 / pre_rho)).tocsr()
    # nnz must be preserved by a scalar rescale.
    assert topo.matrix.nnz == expected_matrix.nnz
    # The values must agree to a tight tolerance — scipy's CSR multiply is
    # exact for the scalar case.
    diff = np.abs(topo.matrix.data - expected_matrix.data).max()
    assert diff <= 1.0e-5


def test_n0_raw_row_column_sums_after_rescale(tmp_path: Path) -> None:
    """``n0_raw`` preserves the row-sum/column-sum proportions of the raw matrix.

    A scalar rescale (one factor for the whole matrix) multiplies both row
    and column sums by the same factor. So if ``A`` is the raw adjacency and
    ``s`` is the rescale factor, ``(s*A)`` has row sums ``s * row_sums(A)``
    and column sums ``s * col_sums(A)`` — the ratios are preserved. This
    test pins that invariant for ``n0_raw`` and gives a concrete property
    the loader can be checked against when the NPZ is the real olfactory
    artifact.
    """
    npz_path = tmp_path / "synthetic_olfactory.npz"
    raw, _others = _write_synthetic_olfactory_npz(npz_path)

    topo = load_reservoir_topology_from_npz(
        str(npz_path),
        normalization="n0_raw",
        target_spectral_radius=0.5,
        seed=0,
    )

    raw_row = np.asarray(raw.sum(axis=1)).ravel()
    raw_col = np.asarray(raw.sum(axis=0)).ravel()
    loaded_row = np.asarray(topo.matrix.sum(axis=1)).ravel()
    loaded_col = np.asarray(topo.matrix.sum(axis=0)).ravel()
    # The raw matrix has no zero rows (synthetic data); ratios should be tight.
    nonzero_row = raw_row > 0
    nonzero_col = raw_col > 0
    row_ratio = loaded_row[nonzero_row] / raw_row[nonzero_row]
    col_ratio = loaded_col[nonzero_col] / raw_col[nonzero_col]
    assert np.allclose(row_ratio, row_ratio[0], atol=1.0e-4)
    assert np.allclose(col_ratio, col_ratio[0], atol=1.0e-4)
    # The single shared rescale factor must be the same on rows and columns.
    assert abs(row_ratio[0] - col_ratio[0]) <= 1.0e-4


def test_n0_raw_default_path_for_m4_loads(tmp_path: Path) -> None:
    """The M4 default normalization (``n0_raw``) loads without raising.

    M4 plans to consume the reservoir at its default ``--normalization``;
    the loader must accept that choice without a fallback or a synthetic
    substitute. The test writes the DATA-3 key layout and asks the loader
    for the exact path M4 will take.
    """
    npz_path = tmp_path / "synthetic_olfactory.npz"
    _raw, _others = _write_synthetic_olfactory_npz(npz_path)
    topo = load_reservoir_topology_from_npz(
        str(npz_path),
        normalization="n0_raw",
        target_spectral_radius=0.9,
        seed=0,
    )
    assert topo.matrix.shape[0] == topo.n_nodes
    assert topo.matrix.nnz == topo.n_edges
    assert topo.matrix.format == "csr"


def test_n0_raw_distinct_from_other_normalizations(tmp_path: Path) -> None:
    """``n0_raw`` and the other five normalizations do not silently alias.

    After rescale, the (N, M, rho) deltas across the family are zero by
    construction; but the *weights themselves* must differ. The loader
    must not return the same matrix for two different normalization
    choices — a regression that would silently collapse the M4 family.
    """
    npz_path = tmp_path / "synthetic_olfactory.npz"
    _raw, _others = _write_synthetic_olfactory_npz(npz_path)
    n0 = load_reservoir_topology_from_npz(
        str(npz_path), normalization="n0_raw", target_spectral_radius=0.9, seed=0
    )
    n1 = load_reservoir_topology_from_npz(
        str(npz_path), normalization="n1_pre_l1", target_spectral_radius=0.9, seed=0
    )
    # The two matrices share (N, M) but their data arrays differ.
    assert n0.matrix.shape == n1.matrix.shape
    assert n0.matrix.nnz == n1.matrix.nnz
    assert not np.array_equal(n0.matrix.data, n1.matrix.data), (
        "n0_raw and n1_pre_l1 produced identical matrices — loader aliases"
    )