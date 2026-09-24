#!/usr/bin/env python
"""FD3 -- the evaluation-leakage audit on the wine dataset.

Authority: docs/resaudit_food_preregistration_amendment_7.md section 4.

The question is NOT "which substrate is better". It is: how much does an evaluation inflate when
repeated acquisitions of one physical bottle are placed on both sides of the train/test boundary?
No reservoir, no family and no A3 qualification is involved.

Pre-declared design (fixed before measurement):
  * 235 wine sequences, 3 classes (AQ/HQ/LQ), 6 gas channels; RH and temperature excluded
    (columns 0-1), matching the declared Din=6. The 65-file ethanol folder is not used.
  * grouped (primary): leave-one-bottle-out, 22 folds -- all 9-13 repetitions of a bottle stay on
    one side.
  * naive (control): 22 folds by random partition of sequences, 5 independent seeds.
  * three plain, fixed, non-novel models, identical across schemes, no per-scheme tuning:
      ridge  -- multinomial logistic regression on standardised per-channel summary features
      rf     -- RandomForestClassifier at fixed hyper-parameters
      temp   -- ridge classification on the downsampled series (20 x 6)
  * macro-F1 primary, accuracy secondary; reported per fold, then aggregated.
  * Delta_leak = mean over seeds of naive - grouped, with per-bottle and per-seed spread. NO
    significance claim is made: 22 bottles, of which only 4 are AQ.
  * folds whose test side lacks a class are reported, not silently scored 0.

Usage:
    PYTHONPATH=. python3 ops/audit/fd3_leakage.py [--data ../data1/raw/wine] [--out ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

#: The frozen design constants (amendment 7 section 4).
N_FOLDS = 22
NAIVE_SEEDS = (0, 1, 2, 3, 4)
SENSOR_COLS = slice(2, 8)          # 6 gas channels; columns 0-1 are RH and temperature
DOWNSAMPLE_POINTS = 20
N_ESTIMATORS = 300
RF_SEED = 20260923
NAME_RE = re.compile(r"^(?P<class>AQ|HQ|LQ)_Wine(?P<wine>\d+)-B(?P<bottle>\d+)_R(?P<rep>\d+)\.txt$")


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_wine(root: Path):
    """Load the 235 wine acquisitions with their bottle identity."""
    rows = []
    for path in sorted(root.glob("*/*.txt")):
        m = NAME_RE.match(path.name)
        if not m:
            continue  # the Ethanol folder -- a separate experiment, excluded by design
        seq = np.genfromtxt(path, delimiter=None)[:, SENSOR_COLS]
        rows.append({
            "path": path, "label": m.group("class"),
            "wine": f"{m.group('class')}_Wine{m.group('wine')}",
            "bottle": f"{m.group('class')}_Wine{m.group('wine')}-B{m.group('bottle')}",
            "rep": int(m.group("rep")), "seq": seq,
        })
    return rows


def features_summary(rows) -> np.ndarray:
    """Per-channel summary statistics: mean, sd, min, max, and first-to-last slope."""
    out = []
    for r in rows:
        s = r["seq"]
        f = [s.mean(0), s.std(0), s.min(0), s.max(0), (s[-1] - s[0])]
        out.append(np.concatenate(f))
    return np.asarray(out, dtype=np.float64)


def features_temporal(rows) -> np.ndarray:
    """A temporal baseline representation: the series downsampled to fixed time points."""
    idx = np.linspace(0, rows[0]["seq"].shape[0] - 1, DOWNSAMPLE_POINTS).astype(int)
    return np.asarray([r["seq"][idx].ravel() for r in rows], dtype=np.float64)


def make_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        # "plain, fixed, non-novel" -- declared hyper-parameters, no tuning anywhere.
        # sklearn >= 1.8 dropped multi_class; multinomial is the default for lbfgs.
        "ridge_logreg": lambda: LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
        "rf": lambda: RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RF_SEED, n_jobs=1
        ),
    }


def _fit_eval(make_model, Xtr, ytr, Xte, yte, classes):
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    model = make_model().fit(sc.transform(Xtr), ytr)
    pred = model.predict(sc.transform(Xte))
    present = sorted(set(yte))
    # macro-F1 with the full label set is DEGENERATE when the fold's test side is single-class:
    # the two absent classes score 0, so the maximum attainable value is 1/len(classes). Both
    # variants are reported so the reader can see the degeneracy instead of inheriting it.
    macro_all = f1_score(yte, pred, labels=classes, average="macro", zero_division=0)
    macro_present = f1_score(yte, pred, labels=present, average="macro", zero_division=0)
    return {
        "macro_f1": float(macro_all),
        "macro_f1_present_classes": float(macro_present),
        "accuracy": float(accuracy_score(yte, pred)),
        "test_n": int(len(yte)),
        "test_classes": present,
        "test_missing_classes": [c for c in classes if c not in present],
    }


def grouped_folds(rows):
    """Leave-one-bottle-out: 22 folds, one per bottle."""
    bottles = sorted({r["bottle"] for r in rows})
    for b in bottles:
        test = [i for i, r in enumerate(rows) if r["bottle"] == b]
        train = [i for i, r in enumerate(rows) if r["bottle"] != b]
        yield b, train, test


def grouped_kfold(rows, k: int):
    """k folds by bottle: every fold's test side is a whole set of bottles, so it spans classes."""
    bottles = sorted({r["bottle"] for r in rows})
    for i in range(k):
        test_b = set(bottles[i::k])
        test = [j for j, r in enumerate(rows) if r["bottle"] in test_b]
        train = [j for j, r in enumerate(rows) if r["bottle"] not in test_b]
        yield f"gfold{i}", train, test


def naive_kfold(rows, k: int, seed: int):
    """k folds by stratified random assignment of sequences (class balance preserved)."""
    y = np.asarray([r["label"] for r in rows])
    rng = np.random.default_rng(seed)
    assign = np.empty(len(rows), dtype=int)
    for c in sorted(set(y.tolist())):
        idx = np.where(y == c)[0]
        assign[rng.permutation(idx)] = np.arange(len(idx)) % k
    for i in range(k):
        test = [j for j in range(len(rows)) if assign[j] == i]
        train = [j for j in range(len(rows)) if assign[j] != i]
        yield f"nfold{i}", train, test


def naive_folds(rows, seed: int):
    """22 folds by random partition of sequences, matched to the grouped fold count."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    assignment = np.array_split(order, N_FOLDS)
    for k, test in enumerate(assignment):
        tset = set(int(i) for i in test)
        train = [i for i in range(len(rows)) if i not in tset]
        yield f"fold{k}", train, sorted(tset)


def run_scheme(rows, y, classes, folds, feat, make_model):
    per_fold = []
    for key, tr, te in folds:
        if not tr or not te:
            continue
        res = _fit_eval(make_model, feat[tr], y[tr], feat[te], y[te], classes)
        res["fold"] = key
        per_fold.append(res)
    macro = np.array([f["macro_f1"] for f in per_fold])
    mpres = np.array([f["macro_f1_present_classes"] for f in per_fold])
    acc = np.array([f["accuracy"] for f in per_fold])
    single = sum(1 for f in per_fold if len(f["test_classes"]) == 1)
    return per_fold, {
        "n_folds": len(per_fold),
        "macro_f1_mean": float(macro.mean()),
        "macro_f1_sd": float(macro.std(ddof=1)) if len(macro) > 1 else 0.0,
        "macro_f1_min": float(macro.min()),
        "macro_f1_max": float(macro.max()),
        "macro_f1_present_classes_mean": float(mpres.mean()),
        "accuracy_mean": float(acc.mean()),
        "accuracy_sd": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
        "folds_missing_a_class": int(sum(1 for f in per_fold if f["test_missing_classes"])),
        "folds_with_single_class_test": int(single),
        "macro_f1_attainable_ceiling": float(1.0 / len(classes)) if single else 1.0,
        "degenerate_macro_f1": bool(single == len(per_fold)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "data1/raw/wine"))
    ap.add_argument("--out", default=str(REPO / "results/audit/resaudit_stage1/fd3_leakage.json"))
    args = ap.parse_args(argv)

    root, out_path = Path(args.data), Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_wine(root)
    y = np.asarray([r["label"] for r in rows])
    classes = sorted(set(y.tolist()))
    bottles = sorted({r["bottle"] for r in rows})
    reps = {}
    for r in rows:
        reps[r["bottle"]] = reps.get(r["bottle"], 0) + 1
    print(f"loaded {len(rows)} wine acquisitions | classes {classes} "
          f"| bottles {len(bottles)} | reps/bottle {min(reps.values())}-{max(reps.values())}")
    print(f"class counts (sequences): { {c: int((y == c).sum()) for c in classes} }")
    print(f"class counts (bottles)  : "
          f"{ {c: sum(1 for b in bottles if b.startswith(c + '_')) for c in classes} }")

    feats = {"summary": features_summary(rows), "temporal": features_temporal(rows)}
    models = make_models()
    results: dict = {"comparison_matched_kfold": {}, "comparison_loo_accuracy": {},
                     "degeneracy_note": {
                         "finding": ("leave-one-bottle-out gives every test fold a SINGLE class, "
                                     "so macro-F1 over the full label set is bounded above by "
                                     "1/len(classes) and is not a valid performance measure"),
                         "ceiling_when_single_class": 1.0 / len(classes),
                         "consequence": ("Delta_leak computed on the LOO scheme using full-label "
                                         "macro-F1 is an artifact of that degeneracy; the matched "
                                         "k-fold comparison is the primary result"),
                     }}

    # ---- primary: matched k-fold, both sides multi-class, macro-F1 non-degenerate ----
    for fname, X in feats.items():
        for mname, make in models.items():
            _, g5 = run_scheme(rows, y, classes, grouped_kfold(rows, 5), X, make)
            seeds5 = [run_scheme(rows, y, classes, naive_kfold(rows, 5, s), X, make)[1]
                      for s in NAIVE_SEEDS]
            nmean5 = float(np.mean([v["macro_f1_mean"] for v in seeds5]))
            nrange5 = [min(v["macro_f1_mean"] for v in seeds5),
                       max(v["macro_f1_mean"] for v in seeds5)]
            key = f"{fname}/{mname}"
            results["comparison_matched_kfold"][key] = {
                "grouped_5fold": g5, "naive_5fold_seeds": seeds5,
                "naive_5fold_macro_f1_mean": nmean5, "naive_5fold_macro_f1_range": nrange5,
                "delta_macro_f1": nmean5 - g5["macro_f1_mean"],
                "grouped_folds_degenerate": g5["degenerate_macro_f1"],
            }
            print(f"  [matched 5-fold] {fname:9s} {mname:12s} "
                  f"grouped={g5['macro_f1_mean']:.4f} (deg={g5['degenerate_macro_f1']}) "
                  f"naive={nmean5:.4f} [{nrange5[0]:.4f},{nrange5[1]:.4f}]  "
                  f"Delta={nmean5 - g5['macro_f1_mean']:+.4f}")

    # ---- secondary: LOO-bottle vs 22-fold naive, measured on ACCURACY (well defined) ----
    for fname, X in feats.items():
        for mname, make in models.items():
            _, glo = run_scheme(rows, y, classes, grouped_folds(rows), X, make)
            seeds22 = [run_scheme(rows, y, classes, naive_folds(rows, s), X, make)[1]
                       for s in NAIVE_SEEDS]
            nacc = float(np.mean([v["accuracy_mean"] for v in seeds22]))
            naccc = float(np.mean([v["macro_f1_mean"] for v in seeds22]))
            key = f"{fname}/{mname}"
            results["comparison_loo_accuracy"][key] = {
                "grouped_loo": glo, "naive_22fold_seeds": seeds22,
                "naive_22fold_accuracy_mean": nacc,
                "naive_22fold_macro_f1_mean_degenerate_side": naccc,
                "delta_accuracy": nacc - glo["accuracy_mean"],
                "delta_macro_f1_full_labels_ARTIFACT": naccc - glo["macro_f1_mean"],
            }
            print(f"  [LOO vs 22-fold, accuracy] {fname:9s} {mname:12s} "
                  f"grouped_acc={glo['accuracy_mean']:.4f} naive_acc={nacc:.4f}  "
                  f"Delta_acc={nacc - glo['accuracy_mean']:+.4f}")

    legacy_results = {"grouped": {}, "naive": {}, "delta_leak": {}}

    for fname, X in feats.items():
        for mname, make in models.items():
            # grouped: deterministic
            gfold, gsum = run_scheme(rows, y, classes, grouped_folds(rows), X, make)
            legacy_results["grouped"][f"{fname}/{mname}"] = {"summary": gsum, "per_fold": gfold}
            # naive: 5 seeds
            seeds = []
            for s in NAIVE_SEEDS:
                nfold, nsum = run_scheme(rows, y, classes, naive_folds(rows, s), X, make)
                seeds.append(nsum)
            nmean = float(np.mean([s["macro_f1_mean"] for s in seeds]))
            nrng = (min(s["macro_f1_mean"] for s in seeds),
                    max(s["macro_f1_mean"] for s in seeds))
            legacy_results["naive"][f"{fname}/{mname}"] = {"seeds": seeds,
                                                           "macro_f1_mean": nmean,
                                                           "macro_f1_range": list(nrng)}
            legacy_results["delta_leak"][f"{fname}/{mname}"] = {
                "delta_macro_f1": nmean - gsum["macro_f1_mean"],
                "naive_mean": nmean, "naive_range": list(nrng),
                "grouped_mean": gsum["macro_f1_mean"],
                "grouped_sd_across_bottles": gsum["macro_f1_sd"],
            }
            print(f"  {fname:9s} {mname:12s} grouped={gsum['macro_f1_mean']:.4f} "
                  f"naive={nmean:.4f} [{nrng[0]:.4f},{nrng[1]:.4f}]  "
                  f"Delta_leak={nmean - gsum['macro_f1_mean']:+.4f}")

    report = {
        "experiment": "FD3 evaluation-leakage audit (wine)",
        "authority": "docs/resaudit_food_preregistration_amendment_7.md section 4",
        "design": {
            "n_sequences": len(rows), "classes": classes, "n_bottles": len(bottles),
            "n_folds": N_FOLDS, "naive_seeds": list(NAIVE_SEEDS),
            "grouped": "leave-one-bottle-out (22 folds)",
            "naive": "22-fold random partition of sequences",
            "features": {"summary": "mean/sd/min/max/slope per channel (30 dims)",
                         "temporal": f"downsampled series {DOWNSAMPLE_POINTS}x6 (120 dims)"},
            "models": {"ridge_logreg": "LogisticRegression(C=1.0, lbfgs; multinomial default)",
                       "rf": f"RandomForestClassifier(n_estimators={N_ESTIMATORS}, "
                             f"random_state={RF_SEED})"},
            "metric_primary": "macro_f1",
            "sensor_columns": "2..7 (6 gas channels); RH and temperature excluded",
            "ethanol_folder": "excluded (separate experiment)",
            "no_significance_claim": ("22 bottles, 4 of them AQ; Delta_leak is reported with "
                                      "per-bottle and per-seed spread only"),
        },
        "repetitions_per_bottle": reps,
        "results": {**results, "legacy_full_label_macro_f1": legacy_results},
        "git_head": git_head(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
