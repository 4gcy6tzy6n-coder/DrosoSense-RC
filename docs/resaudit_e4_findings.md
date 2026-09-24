# E4 -- pork: why a grouped-vs-naive classification cannot be defined

**Headline: the pork dataset defines labels as the only independent experimental variable (the
adulteration percentage), so any grouped split on a putative "mixture identifier" would reduce to
"one group = one label", which is not a CV design. We do not run the experiment, and we report
the design-level reason for that decision.**

Authority: the operator's E4 spec (2026-09-23): "不要为了迎合 reviewer 强行做无效 CV ... 一个小型 protocol check：证明 mixture_id == class-defining preparation，列出 7 个 mixture、对应文件数、label，并明确说明为什么不存在 label-independent grouped split。"

## 1. The seven "preparations" and their file counts

The pork deposit names its 420 files as `S{level}_{index}.csv`, where

```
level in {0, 10, 25, 50, 75, 90, 100}   -- 7 adulteration percentages (label-defining)
index in {1..N_level}                    -- a campaign-wide trial number
```

The depositor's own description states (recorded in the v3_substrate_selection-equivalent audit notes
on the project's frozen record): "seven sample compositions (each 250 g) were prepared ... 250 g beef
(100% beef), 225 g beef + 25 g pork (90% / 10%), ..., 250 g pork (100% pork)." Seven compositions, each
defining one label. The 60 files per level are *trials* on that composition.

Measured counts from the deposit:

| level label | label meaning                  | files |
|---|---|---:|
| S000 | 0 % pork (100 % beef)          |  60 |
| S010 | 10 % pork (90 % / 10 %)        |  60 |
| S025 | 30 % pork (70 % / 30 %) -- level name is 025, content 30 % |  60 |
| S050 | 50 % pork (50 % / 50 %)        |  60 |
| S075 | 70 % pork (30 % / 70 %)        |  60 |
| S090 | 90 % pork (10 % / 90 %)        |  60 |
| S100 | 100 % pork                    |  60 |
| **total** |                              | **420** |

## 2. The trial numbering is not a mixture identifier

The second index runs from 1 to a level-dependent maximum (S000 spans 1..64 with gaps; S090 spans 37..99;
S050 spans 1..90 with gaps; etc.). Trial indices are non-contiguous per level and overlap across levels,
which is consistent with a **campaign-wide trial counter** rather than a per-mixture sequence number.
The deposit contains:

```
mixture_id field:           NONE
date field:                 NONE
animal / specimen field:    NONE
session / batch field:      NONE
preparation / sample field: NONE
trial index (file-suffix):  present, but not reproducible per mixture
```

Therefore the deposit does not provide an independent experimental unit other than `level`. Any
"mixture id" inferred from the deposit would have to equal "the level that defines the label", by
construction, because no other discriminator is present.

## 3. Why a grouped split cannot be defined

A grouped split requires a grouping variable that is

```
(a) not the label,
(b) independent across groups for the same label, and
(c) recovered from the deposit.
```

For pork only (a) is recoverable, and it implies (a) is *false* -- every group would carry one label.
The two consequences the reviewer should know about:

- **There is no label-independent grouping variable.** Every proposed mixture id collapses to the
  level label, so a "grouped" CV with one mixture per group is "leave-one-class-out", not k-fold CV.
- **The 60 files within a level are not 60 independent preparations**, and the deposit does not say
  they are. They could be 60 separate preparations, 60 successive measurements on one preparation, or
  any mix in between; the files are byte-distinct but their physical independence is unrecoverable.

Therefore, a pork grouped-vs-naive classification experiment is not a missing-data experiment; it is
an *indefinable* experiment. Running it would require inventing a mixture definition (e.g., "the
trial index modulo some_k") that the deposit does not declare and that the user did not authorise.
That is exactly the failure mode the operator's spec warns against ("强行做无效 CV").

## 4. The audit-paper framing

This E4 note IS the answer to the reviewer's request that "Wine/Tea/Pork grouped-vs-naive"
experiments all be run. The deposit's structure makes the pork variant undefined by design:

- wine defines a real bottle identity (9-13 trials per bottle, bottle-id recoverable);
- tea defines a real chopping identity (3 trials per chop, chop-id recoverable);
- pork defines only the label, not the preparation.

The governance contribution (C1) is therefore: *even when a deposition uses the words "trial" or
"sample", the presence of a grouping variable is a positive empirical fact, not a default
assumption*. Pork fails that fact, so the inference that "420 files = 420 trials" is a 60x unit
inflation that is exposed by a 1-line filename audit, and it cannot be repaired by experiment design
because there is no experiment-design variable to repair it with.

## 5. What is recorded, not invented

- **420 files in 7 levels of 60.** Verified by enumerating `Data_Pork_Adulteration/S{level}_{index}.csv`.
- **No field other than level and campaign-wide trial index exists in any filename.** Verified by
  filename regex.
- **A grouped-vs-naive CV cannot be defined without manufacturing a mixture identifier**, and the
  operator explicitly refused that.
- **No model is run on pork.** The dataset's role is the inflation exemplar (the 420 -> 7 headline),
  not a benchmark.

## Artifacts

This document, plus the existing `data1/reports/dataset_registry.csv` line for pork that records
420 files / 7 units / 60x inflation.
