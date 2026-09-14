# Design notes

## The contract

A consumer supplies three things:

1. **Cases** — an input, optionally a reference, and arbitrary metadata.
2. **A task** — `case -> observations`, where observations is a plain dict of
   whatever that project wants scored.
3. **Metrics** — `observations -> float | None`, each declaring a direction.

The harness owns everything between: seeded repetition, latency capture,
aggregation, variance across seeds, serialisation, and the comparison against a
baseline.

The observation dict is intentionally untyped. A classifier records
`{"correct": bool}`, an agent records `{"groundedness": float, "retries": int}`.
Forcing those into one schema would mean either a union type nobody can satisfy
or a lowest common denominator that measures nothing interesting.

## Why direction is registered, not tabulated

The extracted version held a module-level `DIRECTIONS` dict naming one
project's metrics. That has two failure modes:

- Every new metric in any consumer requires an edit **here**, which defeats the
  point of a shared package.
- A metric missing from the table silently defaults to one direction, so a
  lower-is-better metric can regress in a way the gate reads as an improvement.

Registering direction with the function removes both. Writing it onto the report
removes a third: comparison no longer needs the registry at all, so a baseline
from an older version still compares correctly.

## Why the gate skips rather than fails on a missing metric

Three states are possible for a metric name: present on both sides, only in the
run, or only in the baseline. Only the first is comparable.

Failing on the others would mean a commit that adds a metric breaks CI, and a
commit that retires one breaks it too — which trains people to bypass the gate.
Skipping silently is worse, because a gate that quietly compares nothing still
prints "passed". So: skip, and name what was skipped.

## Why seeds

A single run over a small dataset cannot distinguish a real improvement from
noise. `MetricSummary` carries mean, standard deviation and the per-seed values,
and a single-seed report says so in its own summary rather than presenting one
number as settled.

The runner can pass the seed into the task (`pass_seed=True`) for systems whose
own sampling has to be controlled. For a deterministic system the seeds still
capture run-to-run variance from the model provider.

## What was deliberately left out

Reference-based scorers — BERTScore, ROUGE, F1 — are not here. Each pulls in a
large dependency tree, and a harness that installs PyTorch to be imported will
not be adopted by the project that only wanted the regression gate. They are one
`register_metric` call away in the project that needs them.
