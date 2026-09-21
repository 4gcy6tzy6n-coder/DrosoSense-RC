"""
Tests for connectome/select_neurons.py (DATA-3 M2 / R1.2返工指令第3项).

Run with:
  python -m pytest connectome/tests/test_select_neurons.py -v

Requirements (R1.2 §3):
  - select_neurons.py must be end-to-end executable
  - --target-n 1000 --seed 42 must run successfully
  - Two calls with same seed produce identical node-index SHA-256
  - layer_distribution is non-empty and node count equals target
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "connectome"))
from paths import adjacency_path, metadata_path  # noqa: E402

# Point at data-root aware paths
ADJ_PATH = adjacency_path("olfactory_v1.npz")
NM_PATH = metadata_path("olfactory_v1_node_meta.csv")


def sha256_of_sorted_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.sort(arr).tobytes())
    return h.hexdigest()


class TestSelectNeurons:
    """Test select_neurons.py end-to-end executability."""

    def test_select_neurons_runs(self):
        """--target-n 1000 --seed 42 must complete without error."""
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "connectome" / "select_neurons.py"),
                "--target-n", "1000",
                "--seed", "42",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0, (
            f"select_neurons.py failed with:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_select_neurons_reproducible(self):
        """Two calls with same seed must produce identical SHA-256."""
        import subprocess
        cmd = [
            sys.executable, str(ROOT / "connectome" / "select_neurons.py"),
            "--target-n", "1000", "--seed", "42", "--output-json",
        ]
        out1 = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        out2 = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        assert out1.returncode == 0 and out2.returncode == 0

        prov1 = json.loads(out1.stdout.split("--- JSON provenance ---")[1].strip())
        prov2 = json.loads(out2.stdout.split("--- JSON provenance ---")[1].strip())
        assert prov1["sha256_sorted_root_ids"] == prov2["sha256_sorted_root_ids"], (
            "Same seed must produce identical node set SHA-256"
        )

    def test_select_neurons_layer_distribution_nonempty(self):
        """layer_distribution must be non-empty and node count must equal target."""
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "connectome" / "select_neurons.py"),
                "--target-n", "1000",
                "--seed", "42",
                "--output-json",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0
        prov = json.loads(result.stdout.split("--- JSON provenance ---")[1].strip())
        assert len(prov["layer_distribution"]) > 0, "layer_distribution must be non-empty"
        assert prov["N_selected"] == prov["target_n"], (
            f"N_selected={prov['N_selected']} != target_n={prov['target_n']}"
        )