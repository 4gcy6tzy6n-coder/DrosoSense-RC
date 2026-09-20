# GPU server environment (DATA-24)

This document pins how the GPU server (`connect.nmb2.seetacloud.com`,
container `autodl-container-lq8yfbsa9j-83f8ca6f`) was brought up for
DrosoSense-RC M4 work, with the actual commands and outputs that produced
the current state. The reproducible procedure is the body of this file; the
host facts are a snapshot of what was observed at bring-up and may drift.

## Host snapshot (2026-09-21 02:20 CST)

| Item | Value |
| --- | --- |
| OS | Linux 5.15.0-78-generic x86_64 (Ubuntu, AutoDL container) |
| CPU | 80 vCPU (Xeon Gold 6248 @ 2.5 GHz) |
| RAM | 125 GB total / 105 GB available (`free -h`: 106 GB `available`) |
| GPU | 1× NVIDIA RTX 3080 Ti 12 GB, driver 595.71.05, CUDA 13.2 |
| `/` (overlay) | 30 GB total, **1.3 GB used** (5%) — kept thin on purpose |
| `/root/autodl-tmp` | 50 GB total, **1.3 GB used** (3%) — work volume |
| `/autodl-pub/data` | 4 TB public read-only pool (unused) |
| Python | `/root/miniconda3/bin/python` = 3.12.3 (not on PATH in non-interactive SSH) |
| Pre-installed | numpy 2.1.3, torch 2.5.1+cu124 (`cuda.is_available()=True`), networkx 3.4.2, matplotlib 3.9.2 |
| Network | GitHub unreachable (curl 000); pypi / Mendeley / Zenodo 200 (but see note on Mendeley below) |

Everything bigger than a couple of MB lives under `/root/autodl-tmp` so the
30 GB root overlay never overflows. All commands below assume an SSH key that
Mika already published (`~/.ssh/id_ed25519`) and a non-interactive shell.

## 1. Directory layout on the server

```bash
mkdir -p /root/autodl-tmp/drososense/{repo,data/raw,data/connectome,results,logs}
cd /root/autodl-tmp/drososense
ls -la
```

Expected output:

```
drwxr-xr-x  data
drwxr-xr-x  logs
drwxr-xr-x  repo
drwxr-xr-x  results
```

All subsequent steps drop outputs into this tree. The `.git` of the
`repo/` checkout is included (the rsync preserves it) so subsequent
commits and `git log` work from the server side without an extra clone.

## 2. Repo sync (Mac → server)

The server cannot reach GitHub, so the only way to put code on it is
an `rsync` from this Mac. The source tree is checked out at
`origin/data-16/r0-protocol-and-data-binding-rework` before the rsync
so the working tree already carries `drososense/`, `configs/`,
`scripts/`, and `data/manifests/`.

```bash
# on the Mac
cd /Users/yyl/Desktop/workshop/DrosoSense-RC
git switch -f --detach origin/data-16/r0-protocol-and-data-binding-rework

rsync -a --exclude '.claude' --exclude '.multica' --exclude '.pytest_cache' \
      --exclude '__pycache__' --exclude '*.pyc' \
      repo/ \
      -e "ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651" \
      root@connect.nmb2.seetacloud.com:/root/autodl-tmp/drososense/repo/
```

Server-side sanity check after the rsync:

```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com \
    "cd /root/autodl-tmp/drososense/repo && \
     git config --global --add safe.directory /root/autodl-tmp/drososense/repo && \
     git log --oneline -1 && \
     ls scripts/ configs/ data/manifests/ | head -10"
```

Observed:

```
051943a docs(DATA-21): regenerate the inventory after the direction fix

scripts:
analyze.py  build_connectome.py  download_data.py  …
configs:
datasets  protocol_v1.1.sha256  protocol_v1.1.yaml  protocol_v1.2.sha256  protocol_v1.2.yaml  protocol_v1.yaml
data/manifests:
d1_beef_controlled.yaml  d2_beef_uncontrolled.yaml  d3_rainbow_trout.yaml  d3_sensor_files.yaml  synthetic_enose.yaml
```

`.git/` weighs about 559 MB on disk because the source clone carries every
branch's history — acceptable; trimming is not on M4's critical path.

## 3. Pip install (server)

The conda base had only torch / numpy / networkx / matplotlib; scipy,
scikit-learn, pandas, pyarrow, xgboost, PyYAML and requests were missing.
PyPI is reachable through the AutoDL mirror; pip ended up pulling

```
absl-py==2.1.0               pandas==3.0.6
cloudpickle==3.1.2           pyarrow==25.0.1
joblib==1.6.0               scipy==1.18.1
narwhals==2.26.0           scikit-learn==1.9.1
nvidia-nccl-cu13==2.31.2    threadpoolctl==3.7.0
```

alongside the existing torch + requests + PyYAML. `pip install
scikit-learn pandas` was the slowest leg (~30 s on a 25 kB/s Aliyun
mirror); `xgboost` brought `nvidia-nccl-cu13` in transit. The full
post-install pip freeze is committed at
`docs/server_env_pip.txt` (162 lines).

```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com \
    "/root/miniconda3/bin/pip install scipy scikit-learn pandas pyarrow xgboost requests PyYAML"
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com \
    "cd /root/autodl-tmp/drososense/repo && \
     /root/miniconda3/bin/pip freeze > docs/server_env_pip.txt"
```

Post-install import check:

```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com \
    "/root/miniconda3/bin/python -c 'import scipy, sklearn, pandas, pyarrow, xgboost, requests, yaml, torch; print(\"torch\", torch.__version__, \"cuda_avail\", torch.cuda.is_available())'"
# → torch 2.5.1+cu124 cuda_avail True
```

Sanity test of the project itself:

```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com \
    "cd /root/autodl-tmp/drososense/repo && \
     export DROSOSENSE_DATA=/root/autodl-tmp/drososense/data && \
     /root/miniconda3/bin/pip install pytest pytest-cov && \
     /root/miniconda3/bin/python -m pytest"
# → 383 passed in 24.58s
```

## 4. Food data acquisition (server, no Mac relay)

The data layer must be assembled *on the server* (issue text: "在服务器
上取，不要从本机传"); it is tiny in bytes — D1 + D2 + D3 = ~3.8 MB on
disk — even though D3 lives inside a 21.1 GB archive that we never
download whole.

```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com \
    "cd /root/autodl-tmp/drososense/repo && \
     export DROSOSENSE_DATA=/root/autodl-tmp/drososense/data && \
     /root/miniconda3/bin/python scripts/download_data.py --dataset d3_rainbow_trout"
# → [ok] d3_rainbow_trout: OK
#     210 archive member(s) extracted and verified by SHA-256
```

### 4a. Mendeley returns 403 to python-requests

`scripts/download_data.py` opens `requests.get(...)` with no override
of `User-Agent`, and Mendeley's Cloudflare bot filter rejects that
client with `403 Client Error: Forbidden for url: ...` (see
`logs/download_data.log` for the traceback). `curl` with no
`User-Agent` is *not* blocked — the same URL answers `302 → S3`.

Working procedure for D1 and D2:

```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com "
      mkdir -p /root/autodl-tmp/drososense/repo/data/raw/d1_beef_controlled
      mkdir -p /root/autodl-tmp/drososense/repo/data/raw/d2_beef_uncontrolled
      curl -sSL -o /root/autodl-tmp/drososense/repo/data/raw/d1_beef_controlled/d1_beef_controlled.csv \\
           https://data.mendeley.com/public-files/datasets/n8mc3nspfn/files/8b21e9d0-c805-40bd-814b-9778795c13b2/file_downloaded
      curl -sSL -o /root/autodl-tmp/drososense/repo/data/raw/d2_beef_uncontrolled/source_v3.zip \\
           https://data.mendeley.com/public-files/datasets/mwmhh766fc/files/2e65305f-5e53-4395-9b28-91dfc73ff490/file_downloaded
      cd /root/autodl-tmp/drososense/repo/data/raw/d2_beef_uncontrolled && \
      /root/miniconda3/bin/python -c 'import zipfile; zipfile.ZipFile(\"source_v3.zip\").extractall(\"./\")'
    "
```

D2's manifest declares `TS1.csv..TS5.csv` as
`local_archive_member` entries; extracting the zip into the dataset
directory produces exactly the filenames the manifest expects, so the
upstream `_extract_local_archive_members` step is a no-op.

### 4b. Verify — and a recovered synthetic fixture

Manifest checksum verification across all four manifests:

```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com \
    "cd /root/autodl-tmp/drososense/repo && \
     export DROSOSENSE_DATA=/root/autodl-tmp/drososense/data && \
     /root/miniconda3/bin/python scripts/download_data.py --verify-only --all"
# → [ok] d1_beef_controlled: verified (1 file(s))
#   [ok] d2_beef_uncontrolled: verified (6 file(s))
#   [ok] d3_rainbow_trout: verified (210 file(s))
#   [ok] synthetic_enose: verified (1 file(s))  ← .csv was regenerated by scripts/build_synthetic.py earlier in the project
```

Disk footprint on disk:

| Dataset | Bytes | Files |
| --- | ---: | ---: |
| d1_beef_controlled | 1 246 972 | 1 |
| d2_beef_uncontrolled | 1 064 410 | 6 (zip + 5 csv) |
| d3_rainbow_trout | 1 092 874 | 210 |
| synthetic_enose | 254 284 | 1 |
| **total** | **3 658 540** | **218** |

D3's 21.1 GB archive was **not** downloaded whole: only the 210 listed
CSV members were read by HTTP `Range` from `dataset.zip/content` and
written one-by-one into `data/raw/d3_rainbow_trout/`, with their
manifest SHA-256 checked on receipt.

## 5. Connectome artefacts (Mac → server)

The DATA-3 root moved `olfactory_v1.npz` (739 MB) out of the repo
tree into `data-root/connectome/adjacency/`. rsync from there:

```bash
rsync -av --progress \
      /Users/yyl/Desktop/workshop/DrosoSense-RC/data-root/connectome/adjacency/olfactory_v1.npz \
      /Users/yyl/Desktop/workshop/DrosoSense-RC/connectome/metadata/olfactory_v1_node_meta.csv \
      /Users/yyl/Desktop/workshop/DrosoSense-RC/connectome/metadata/olfactory_v1_meta.json \
      -e "ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651" \
      root@connect.nmb2.seetacloud.com:/root/autodl-tmp/drososense/data/connectome/
```

(The two metadata files still live in the older `connectome/metadata/`
path on the Mac; `data-root/connectome/metadata/` only carries the
566 MiB `edge_meta.csv`, which is not required for M4.)

Server-side check:

```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com \
    "ls -la /root/autodl-tmp/drososense/data/connectome/ && \
     /root/miniconda3/bin/python -c 'import numpy as np; \
       a = np.load(\"/root/autodl-tmp/drososense/data/connectome/olfactory_v1.npz\"); \
       print(\"keys:\", len(a.keys()), \"adj_nnz:\", int(a[\"adj_shape\"][1]))'"
# → keys: 33 adj_nnz: 124185
```

The npz carries 33 arrays: the raw CSR plus six pre-computed
normalizations (`norm_n1_pre_l1` … `norm_n5_binary`, `norm_n2_post_l1`)
and a JSON-encoded `meta` entry — same payload as on the Mac.

## 6. Protocol-freeze check

```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com \
    "cd /root/autodl-tmp/drososense/repo && \
     /root/miniconda3/bin/python -m drososense.utils.protocol --check"
# → freeze OK: 1.2.0 frozen at 2026-09-20T08:40:21Z, sha256 32c57f6efaa8…
#   first test evaluation: 2026-09-20T06:05:40.492008+00:00 (live log)
#     live log /root/autodl-tmp/drososense/repo/results/tables/data_contact_log.json: 24 entr(ies), datasets ['d2_beef_uncontrolled', 'd3_rainbow_trout']
```

The active protocol is `configs/protocol_v1.2.yaml`, frozen at
2026-09-20T08:40:21Z, sha-256 prefix `32c57f6efaa8` — the same
digest the Mac side records, i.e. the file was transferred bit-for-bit.

## 7. Smoke run (timing + GPU sampling)

Two frozen-protocol runs were made to give M4 a per-run budget. They
are *not* M4 themselves.

### 7a. SVM-RBF, D2, 1 seed, 1 fold, window 16

```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com \
    "cd /root/autodl-tmp/drososense/repo && \
     export DROSOSENSE_DATA=/root/autodl-tmp/drososense/data && \
     nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader && \
     date -Iseconds && \
     /root/miniconda3/bin/python scripts/run_baselines.py \
        --dataset d2_beef_uncontrolled --experiment smoke_d24 \
        --models svm_rbf --tasks classification --seeds 0 \
        --max-folds 1 --window-lengths 16 --smoke && \
     nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"
```

Captured in `logs/smoke_nv.log` and `logs/smoke_timing.json`:

```
dataset             model   task            window  macro_f1  bal_acc  auroc    prot_compl  n
d2_beef_uncontrolled svm_rbf classification  16      0.668     0.691    0.899     True         1

wall_clock_seconds=6.355  (Python start + data load + 1 fold + persist)

nvidia-smi before:  4 MiB used, 0% util
nvidia-smi after: 334 MiB used, 0% util   ← torch landed its runtime, no kernel was launched
```

The 334 MiB post-run footprint is `torch`'s lazy CUDA runtime loader,
not an actual GPU computation — `SVC(...).fit()` ran on CPU. The
experiment would have to call `model.to('cuda')` deliberately; none of
the M1 baselines do.

### 7b. ESN, D2, 1 seed, 1 fold, window 16

```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com \
    "cd /root/autodl-tmp/drososense/repo && \
     export DROSOSENSE_DATA=/root/autodl-tmp/drososense/data && \
     nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader && \
     /root/miniconda3/bin/python scripts/run_baselines.py \
        --dataset d2_beef_uncontrolled --experiment smoke_d24 \
        --models esn --tasks classification --seeds 0 \
        --max-folds 1 --window-lengths 16 --smoke && \
     nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"
```

Captured:

```
dataset             model  task            window  macro_f1  bal_acc  auroc    prot_compl  n
d2_beef_uncontrolled esn    classification  16      0.426     0.433    0.660     True         1

wall_clock_seconds=6.501
nvidia-smi before:  4 MiB used, 0% util
nvidia-smi after:   1 MiB used, 0% util   ← ESN is numpy-only, no torch runtime was loaded this time
```

The ESN baseline uses numpy throughout (`drososense/reservoir/esn.py`
has no `torch` import), so the GPU stays cold.

### 7c. Per-model wall-clock reference for M4 planning

| Model | D2 (LOSO×1) | GPU used | Notes |
| --- | ---: | --- | --- |
| `svm_rbf` | **6.4 s** | no (CPU only) | Includes Python + data load + 1 fold + JSON write |
| `esn` | **6.5 s** | no (CPU only) | Numpy ESN, no torch runtime even loaded |

Scaling hints for M4:

* 10 seeds × D2's LOSO(5) → about `10 × 5 × 6.4 s ≈ 320 s ≈ 5 min`
  per classical / ESN model on this box.
* 210 members × D3 will be larger; budget one full minute for the
  larger classical / deep models per (seed, fold).

## 8. Final disk state

```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -p 31651 \
    root@connect.nmb2.seetacloud.com \
    "df -h / /root/autodl-tmp && du -sh /root/autodl-tmp/drososense/*"
```

```
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G  1.3G   29G   5% /              ← not bloated
/dev/sdb         50G  1.3G   49G   3% /root/autodl-tmp

746M  /root/autodl-tmp/drososense/data
 32K  /root/autodl-tmp/drososense/logs
573M  /root/autodl-tmp/drososense/repo     ← 559M of that is .git/
 60K  /root/autodl-tmp/drososense/results
```

The split that the issue calls for held: the 30 GB `/` only grew to
1.3 GB (5%), and the npz + datasets + repo all sit on `/root/autodl-tmp`.

## 9. Known limits and blockers

* **Mendeley vs python-requests** — see §4a. The fix is a one-line
  `headers={"User-Agent": ...}` inside `download_file()` in
  `scripts/download_data.py` (and a matching default for
  `drososense/data/remote_zip.py`); both files only need a small
  default-user-agent shim, not new logic. Patching the shared
  code is OUT of scope for this issue (DATA-24 only establishes the
  environment) and is being left for whoever owns DATA-2.x.
* **No GitHub access from the server.** Subsequent server-side
  commits/pushes must be planned, not ad-hoc: the server's `.git/`
  is real but `origin` cannot be pushed from there.
* **`.git/` is 559 MB.** Acceptable for a single-room work volume;
  `git gc --aggressive` would halve it but pulls CPU time we don't
  yet need.
* **`docs/server_env_pip.txt` is large (162 lines).** It is committed
  intentionally: it's the only way to recover the exact pip state on
  a fresh container, and full reproducibility is the whole point of
  this issue.
