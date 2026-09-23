"""Canonical project paths.

Every module resolves paths through this file so that the project can be moved
or mounted read-only without editing code. No module hardcodes a path.
"""

from __future__ import annotations

from pathlib import Path

# drososense/utils/paths.py -> drososense/utils -> drososense -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DATA_SPLITS_DIR: Path = PROJECT_ROOT / "data" / "splits"
DATA_MANIFESTS_DIR: Path = PROJECT_ROOT / "data" / "manifests"
RESULTS_RAW_DIR: Path = PROJECT_ROOT / "results" / "raw"
RESULTS_TABLES_DIR: Path = PROJECT_ROOT / "results" / "tables"

# Each model's trainable-parameter count, committed. `results/raw/**` is
# git-ignored, so a gate term written `params(R0) < params(GRU)` is unevaluable
# for anyone without the author's working copy unless the counts travel with the
# repository. See drososense/evaluation/results.py:model_parameter_table.
MODEL_PARAMETERS_PATH: Path = RESULTS_TABLES_DIR / "model_parameters.json"
RESULTS_FIGURES_DIR: Path = PROJECT_ROOT / "results" / "figures"

# v1 is kept unchanged on disk: the freeze rule is that an amendment adds a new
# version file rather than editing the old one, so that results produced under an
# earlier protocol stay attributable to the text that produced them.
PROTOCOL_V1_PATH: Path = CONFIGS_DIR / "protocol_v1.yaml"
PROTOCOL_V1_1_PATH: Path = CONFIGS_DIR / "protocol_v1.1.yaml"
PROTOCOL_V1_2_PATH: Path = CONFIGS_DIR / "protocol_v1.2.yaml"
# v1.3 (DATA-25) is the amendment that changes D3's split to LOSO(62) per OD1
# ruling (a). The file is on disk with its own sidecar, but PROTOCOL_PATH below
# stays on v1.2 — the active protocol switch is the Experimental Statistician's
# independent review + Thinker gate, not part of this issue's deliverable.
PROTOCOL_V1_3_PATH: Path = CONFIGS_DIR / "protocol_v1.3.yaml"
# v1.4 (DATA-41/DATA-42) is the §17 clarification amendment (crashed runs do not
# consume the test-touch quota). Its file + sidecar travel byte-identical from
# the reviewed DATA-41 commit. It amends v1.3, which STAYS the ACTIVE
# protocol: the runner records PROTOCOL_VERSION = "1.4.0" on new runs (the
# clarified §17 behaviour), while dataset splits and every other field are
# still read from v1.3, so D3 remains LOSO(62). The active switch to the
# v1.4 FILE is the next active-switch version file's job (DATA-44) and needs
# its own independent review — v1.4 alone is a 6-key amendment record, not a
# loadable protocol.
PROTOCOL_V1_4_PATH: Path = CONFIGS_DIR / "protocol_v1.4.yaml"
# v1.5 (M4-Audit) is the evidence-identity amendment: §17's unit of identity is
# now a versioned named tuple that includes the substrate-defining fields
# (reservoir_size, normalization, input_mapping, topology_variant, rewire_seed),
# and the schema-1 fingerprint is retained as a lookup alias so a pre-v1.5 unit
# stays protected. Like v1.4 it is an amendment RECORD layered on the loadable
# v1.3 base: it changes the identity §17 enforces and nothing else — no model,
# dataset, split, endpoint or statistical decision rule moves.
PROTOCOL_V1_5_PATH: Path = CONFIGS_DIR / "protocol_v1.5.yaml"

# v1.5.1 (M4-Audit) declares the EVIDENCE SCOPE of a gate's `params(model)` term:
# the count must come from matched result rows inside a declared
# (experiment labels, condition, task, datasets) scope, with no fallback and no
# max/min/first/last selection, and every resolved count carries provenance.
# Another amendment record layered on the loadable v1.3 base.
PROTOCOL_V1_5_1_PATH: Path = CONFIGS_DIR / "protocol_v1.5.1.yaml"

# The active protocol. Everything that reads "the protocol" reads this.
#
# v1.2 supersedes v1.1 for the DECISION MACHINE only: v1.1's gate expressions
# could not resolve a dataset name and its decisive p-value was pseudoreplicated.
# v1.1 stays on disk, unchanged, with its own sidecar still matching, so every
# result produced under it remains attributable to the text that produced it.
# v1.3 stays active in turn: v1.4 on disk is the amendment record, and the
# runner's version label carries the clarified §17 semantics.
PROTOCOL_PATH: Path = PROTOCOL_V1_3_PATH

# A protocol cannot contain its own hash. The digest of the frozen YAML lives in
# a sidecar file next to it, which is what makes a post-freeze edit detectable:
# `python -m drososense.utils.protocol --check` recomputes it.
PROTOCOL_SHA256_PATH: Path = CONFIGS_DIR / "protocol_v1.3.sha256"
PROTOCOL_V1_1_SHA256_PATH: Path = CONFIGS_DIR / "protocol_v1.1.sha256"
PROTOCOL_V1_2_SHA256_PATH: Path = CONFIGS_DIR / "protocol_v1.2.sha256"
PROTOCOL_V1_3_SHA256_PATH: Path = CONFIGS_DIR / "protocol_v1.3.sha256"
PROTOCOL_V1_4_SHA256_PATH: Path = CONFIGS_DIR / "protocol_v1.4.sha256"
PROTOCOL_V1_5_SHA256_PATH: Path = CONFIGS_DIR / "protocol_v1.5.sha256"
PROTOCOL_V1_5_1_SHA256_PATH: Path = CONFIGS_DIR / "protocol_v1.5.1.sha256"


def protocol_version_paths(version: str) -> tuple[Path, Path]:
    """Return the (protocol_path, sidecar_path) for a frozen version.

    Args:
        version: The protocol version string, e.g. ``"1.3.0"``.

    Returns:
        ``(PROTOCOL_V{N+1}_PATH, PROTOCOL_V{N+1}_SHA256_PATH)``.

    Raises:
        ValueError: If ``version`` does not match any registered protocol file.

    Notes:
        v1.0.0 predates the sidecar mechanism and has no sidecar; the v1 entry
        points at a sidecar path that does not exist on disk, and any reader
        must check ``sidecar.is_file()`` before relying on it.
    """
    table = {
        "1.0.0": (PROTOCOL_V1_PATH, CONFIGS_DIR / "protocol_v1.sha256"),
        "1.1.0": (PROTOCOL_V1_1_PATH, PROTOCOL_V1_1_SHA256_PATH),
        "1.2.0": (PROTOCOL_V1_2_PATH, PROTOCOL_V1_2_SHA256_PATH),
        "1.3.0": (PROTOCOL_V1_3_PATH, PROTOCOL_V1_3_SHA256_PATH),
        "1.4.0": (PROTOCOL_V1_4_PATH, PROTOCOL_V1_4_SHA256_PATH),
    }
    if version not in table:
        raise ValueError(
            f"unknown protocol version {version!r}; known: {sorted(table)}"
        )
    return table[version]


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing and return it.

    Args:
        path: Directory to create.

    Returns:
        The same path, for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_raw_dir(dataset_id: str) -> Path:
    """Return the raw-data directory for one dataset, creating it if needed.

    Args:
        dataset_id: Dataset identifier, e.g. ``d1_beef_controlled``.

    Returns:
        ``data/raw/<dataset_id>``.
    """
    return ensure_dir(DATA_RAW_DIR / dataset_id)
