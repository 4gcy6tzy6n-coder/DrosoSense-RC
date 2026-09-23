# ResAudit-Food — pre-registration amendment 3: the probe's input realization

**Status: amendment, 2026-09-23. Written after the audit runner was built and its invariant
tests passed, and BEFORE any real family (F1–F5) was measured.**

**The original pre-registration is immutable**, and so is every earlier amendment. This file
edits nothing; per this repository's amendment discipline a new file is added instead.

**Scope, stated narrowly because a wider version of this amendment would be redundant.** The
operator's pre-Stage-1 condition was: *if A4/A5 are driven by a task-blind probe, freeze the probe
generator before measuring.* That condition is **already met** — amendment 1 §0.1 freezes the drive
protocol, with an explicit "may it change after seeing F scores? **No**" against each value and the
statement that *"if a later stage wants a different drive, that is a new amendment with its own
justification, not a fallback when a family fails."* This amendment does **not** restate, replace or
renegotiate any of that.

What it closes is one demonstrable gap. Amendment 1 fixes the protocol's *hyper-parameters* but
does not define the *realization* the protocol produces. Searching that amendment confirms it:

```
distribution   0 hits     autocorr   0 hits     input_scale   0 hits
W_in           0 hits     amplitude  0 hits     prefix        0 hits
```

Those five properties exist in code (`resaudit/battery.py`) and are pinned by tests, so no
measurement was ever exposed to a free choice. They are transcribed here so the written record
alone is sufficient to reproduce the drive — which is what "frozen before measurement" is supposed
to mean to a reader who does not read the source.

## 1. What amendment 1 already fixed, and what this amendment adds

| Property | Where frozen | Value |
|---|---|---|
| `rho_target` | amendment 1 §0.1 | 0.95 |
| `leak`, `gain` | amendment 1 §0.1 | 1.0, 1.0 |
| windows / length / seed / washout | amendment 1 §0.1 | 48 / 256 / 20260923 / 16 |
| `K` | pre-registration | 16 |
| **distribution** | **this amendment** | **i.i.d. Uniform(−1.0, +1.0) per channel** |
| **input scale** | **this amendment** | **`PROBE_INPUT_SCALE = 1.0`, one constant for every family** |
| **`W_in` realization** | **this amendment** | **channel `c` drawn from `default_rng(SEED + c)`** |
| **prefix property** | **this amendment** | **the `Din = k` probe is a strict prefix of the `Din = k+1` probe** |
| **autocorrelation** | **this amendment** | **zero by construction (white)** |
| A3/A4/A5 thresholds | unchanged | `2·Din`, `0.20`, `1.5·Din` |

## 2. The frozen realization, transcribed from code

From `resaudit/battery.py` (SHA-256 `518ddc6f411689a7fb2f3d21731b94d3ecad9b0130a0b15e481e7437de34dbe8`):

```python
out = np.empty((PROBE_WINDOWS, PROBE_LENGTH, din))          # (48, 256, Din)
for c in range(din):
    rng = np.random.default_rng(PROBE_SEED + c)             # per-channel generator
    out[:, :, c] = rng.uniform(-1.0, 1.0, size=(PROBE_WINDOWS, PROBE_LENGTH))
return np.ascontiguousarray(out * PROBE_INPUT_SCALE)        # scale 1.0
```

The per-channel seeding is the load-bearing part, and its purpose is stated in the code: *"channel
`c` is drawn from its own generator seeded `seed + c`, so the probe for `din = k` is a strict prefix
of the probe for `din = k + 1`: widening `Din` does not change any existing channel, so comparisons
across families cannot be moved by an accidental redraw."*

This matters directly for the Stage 1 design, because `Din` varies by dataset (FD1/FD3 = 6,
FD2 = 8). Under this rule, a difference between the `Din = 6` and `Din = 8` audit results **cannot**
be attributed to a different input realization — the six shared channels are byte-identical.

## 3. Verification performed for this amendment

The realization was exercised, not merely read:

```
shape                 (48, 256, 8)
range                 min -0.999998   max 0.999994       (consistent with Uniform(-1, 1))
mean / sd             -0.001513 / 0.576501               (1/sqrt(3) = 0.577350)
autocorrelation       lag 1:  0.001355     lag 2:  0.002736
                      lag 4: -0.002182     lag 8:  0.000216     lag 16: 0.000219
determinism           probe_input(8) == probe_input(8)
prefix property       probe_input(6) == probe_input(8)[:, :, :6]
```

Pinned by the engine's own tests, so this is the enforced behaviour and not an observation of a
moment: `test_probe_protocol_is_constant_and_task_free`,
`test_probe_input_is_deterministic_and_family_independent`, and a provenance test asserting the
emitted report carries `probe_protocol.krylov_K`.

## 4. Autocorrelation: declared, with its consequence

The operator's checklist named autocorrelation, and it is **not** a free constant — it is implied by
the distribution. An i.i.d. uniform draw is white, confirmed above at |r| < 0.003 for every lag
tested. Declaring it explicitly turns an inference into a rule, and it is the right rule rather than
a default that happened:

> A white drive means any memory the audit measures must come from the reservoir's recurrence and
> **cannot** come from structure in the input — which is precisely what A4 claims to test. A
> correlated probe would confound input memory with substrate memory.

**Declared limitation, recorded now rather than at review time.** Food sensor inputs are smooth and
strongly autocorrelated, so A4/A5 are measured under a drive whose spectral character differs from
the task drive. Two consequences, both already bounded elsewhere and neither needing action here:

- A4 was retired as a gate before this amendment (see below), so the mismatch cannot affect any
  qualification verdict;
- A4/A5 must therefore **not** be presented as predictions of task performance. If a
  drive-sensitivity check is wanted it belongs in Stage 2 as a declared robustness probe on the task
  drive, outside the audit, and it may not change any A3/A5 verdict.

## 5. Relationship to the A4 decision

A4's stimulus dependence would be alarming if A4 were still a gate. It is not. Amendment 2 withdrew
the memory-contribution claim, and the A4 construct-validity audit
(`docs/resaudit_a4_construct_validity.md`, SHA-256
`cd1dc95d75969bb79b9812f379f3f0516f2845941e7367d17e9db74d3749a7c7`) retired it, with
`BLOCKING_CRITERIA = ("A1", "A2", "A3", "A5")` — A4 absent. Amendment 1 §2.2 records the measurement
that forced it: on the frozen probe the A4 observable is O(0.01), its signed contribution is
negative for graphs with genuine recurrence, and **more** recurrent coupling scores **lower**. So the
operator's two candidate explanations — "the reservoir is inadmissible" versus "the stimulus failed
to excite it" — were already separated, and the recorded answer is a third one: the observable does
not track recurrence strength in the direction its own definition requires. A4 is reported and
explained, never blocking, and the Stage 1 table prints it as `(descriptive only)`.

## 6. Effect

| Item | Effect |
|---|---|
| A3/A4/A5 thresholds | **none** — unchanged |
| Drive protocol (amendment 1 §0.1) | **none** — not restated, not changed |
| Input realization | **documented** — transcribed from code into the record |
| Autocorrelation | **declared** — zero by construction, with its limitation |
| Candidates admitted or excluded | **none** |

Nothing in the battery changes. The written record is now sufficient on its own to reproduce the
drive, which is the standard the operator set for proceeding.
