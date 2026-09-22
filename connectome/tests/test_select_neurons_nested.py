"""
Nested-sampling assertion for the protocol §19 size study (DATA-59).

The protocol (configs/protocol_v1.2.yaml, §19 size_study) declares:

    sizes: [250, 500, 1000, 2000, 4000]
    sampling: nested

i.e. the N=250 subgraph must be a subgraph of the N=500 subgraph, and so on,
so the Performance(N) curve is not contaminated by independent resampling
noise. This test pins that property on the frozen DATA-3 selection
(``connectome.select_neurons``, S0–S4: layer-stratified, seeded), which is
the exact function the reservoir runner (``drososense.connectome_selection.
select_nodes``) calls for every ``--reservoir-size``.

Scope: hermetic, real-NPZ-backed. It resolves the olfactory artifact through
``connectome.paths`` (data-root override, then in-repo fallback) — the same
resolution the runner uses — and needs no model, no folds, no server.

Run with:
  python -m pytest connectome/tests/test_select_neurons_nested.py -v
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "connectome"))

from paths import adjacency_path, metadata_path  # noqa: E402
from select_neurons import select_neurons  # noqa: E402

# §19 grid, in nested order.
SIZES = [250, 500, 1000, 2000, 4000]

# The DATA-3 selection's RNG seed (RNG_SEED in connectome/select_neurons.py);
# the reservoir runner pins the same constant (drososense/reservoir/runner.py).
SELECTION_SEED = 20260920


def _is_nested(a: np.ndarray, b: np.ndarray) -> bool:
    """True when sorted, deduplicated ``a`` is a subset of sorted ``b``.

    Uses ``searchsorted`` (numpy-version-stable) instead of ``in1d`` so the
    assertion is exact and fast on large N.
    """
    a = np.sort(np.unique(a))
    b = np.sort(np.unique(b))
    positions = np.searchsorted(b, a)
    positions = positions[positions < b.size]
    return bool(np.array_equal(b[positions], a))


@pytest.fixture(scope="module")
def nested_sets() -> dict[int, np.ndarray]:
    """One selection per §19 size, same seed — the exact sets compared."""
    adj_path = adjacency_path("olfactory_v1.npz")
    meta_path = metadata_path("olfactory_v1_node_meta.csv")
    assert adj_path.is_file(), (
        f"olfactory NPZ not found at {adj_path}; provision the data root "
        "(export DROSOSENSE_DATA=/path/to/data-root) and re-run."
    )
    sets = {}
    for size in SIZES:
        result = select_neurons(adj_path, meta_path, size, seed=SELECTION_SEED)
        assert result["N_selected"] == size, (
            f"select_neurons returned {result['N_selected']} nodes, "
            f"target_n={size}"
        )
        sets[size] = np.asarray(result["node_indices"], dtype=np.int64)
    return sets


@pytest.mark.parametrize("size", SIZES, ids=[f"N={s}" for s in SIZES])
def test_each_size_selects_exactly_target(nested_sets, size):
    """S3: the selection has exactly ``target_n`` distinct nodes."""
    selected = nested_sets[size]
    assert selected.size == size
    assert np.array_equal(selected, np.sort(np.unique(selected)))


def test_nested_inclusion(nested_sets):
    """§19: N=250 ⊂ N=500 ⊂ N=1000 ⊂ N=2000 ⊂ N=4000, same seed."""
    violations = []
    for small, large in zip(SIZES, SIZES[1:]):
        if not _is_nested(nested_sets[small], nested_sets[large]):
            leaked = nested_sets[small][
                ~np.isin(nested_sets[small], nested_sets[large])
            ]
            violations.append(
                f"N={small} ⊄ N={large}: {leaked.size}/{small} nodes "
                f"leaked (first: {leaked[:8]})"
            )
    assert not violations, "§19 nested sampling violated:\n" + "\n".join(violations)


def test_provenance_digests_differ_across_sizes(nested_sets):
    """Each size has its own node-index digest (the sets are not identical)."""
    digests = {
        size: hashlib.sha256(
            np.ascontiguousarray(np.sort(sets)).tobytes()
        ).hexdigest()
        for size, sets in nested_sets.items()
    }
    assert len(set(digests.values())) == len(SIZES), (
        "two §19 sizes produced the same node set — the nested chain "
        "degenerates; check the selection"
    )
