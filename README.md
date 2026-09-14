# eval-harness

A task-agnostic evaluation harness for LLM systems: datasets, a metric registry,
multi-seed runs, and a **direction-aware regression gate** that CI can fail a
build on.

Nothing in this package knows what an agent, a classifier or a summarizer is. It
takes a callable, a dataset and a set of metrics, and it answers one question:
**did this change make the system better or worse, and by more than noise?**

This is a personal project. It is not derived from any employer's systems, data
or code.

## Why it exists

It was extracted from a retrieval agent, where the evaluation code had grown two
couplings that a shared harness cannot have:

1. The case object called its input a "question", which is meaningless for a
   classifier whose input is a complaint narrative.
2. Metric direction lived in a hardcoded table naming that one project's
   metrics, so no other project could use the gate without editing it.

Both are fixed here: the input is just an input, and **direction is declared
when a metric is registered** and travels on the report. Three projects now
share this one harness — a retrieval agent, a text classifier and a summarizer —
rather than each carrying its own copy of the same 200 lines.

## Install

```bash
pip install "eval-harness @ git+https://github.com/mukundjaiswal/eval-harness.git"
```

Or, working on it side by side with a consumer:

```bash
pip install -e ../eval-harness
```

One runtime dependency (`typer`, for the CLI). A harness that drags a framework
in with it is a harness nobody adopts.

## Use

```python
from eval_harness import (
    Direction,
    EvaluationRunner,
    RegressionCheck,
    load_cases,
    mean_of,
    rate_of,
    register_metric,
)

# 1. Register what your project measures. Direction is part of the metric.
register_metric("accuracy", mean_of("correct"), direction=Direction.HIGHER_IS_BETTER)
register_metric("retry_rate", rate_of("retried"), direction=Direction.LOWER_IS_BETTER)


# 2. Say how one case is handled. Return whatever you want scored.
def task(case):
    prediction = my_system(case.input)
    return {"correct": prediction == case.reference, "retried": False}


# 3. Run it. More than one seed, or you cannot tell a gain from noise.
report = EvaluationRunner(
    task,
    metrics=["accuracy", "retry_rate", "p95_latency_seconds"],
    seeds=[0, 1, 2],
).run(load_cases("datasets/v1.jsonl"), dataset="datasets/v1.jsonl")

print(report.summary())
report.write("results/report.json")
```

Gate a build on it, from Python or from the command line:

```bash
eval-harness compare results/report.json --baseline baselines/v1.json --tolerance 0.02
```

Exits non-zero when a metric moves the wrong way past the tolerance. That exit
code is the whole point: **quality is only defended if a bad change fails the
build.**

## Design decisions worth the words

**Direction travels on the report, not in a lookup table.** A baseline outlives
the code that produced it. A report that describes its own metrics can be
compared a year later by a process that has never imported the project that
wrote it. `RegressionCheck` reads direction off the report.

**A malformed dataset line fails the load; it is never skipped.** Silently
dropping a case changes the denominator of every metric computed afterwards, and
the run still prints a confident number.

**`None` is not zero.** A metric that could not be computed for a case is
excluded from the average, not counted as a failure. Zero reads as "scored
badly"; absent means "not measured", and conflating them quietly drags every
average down.

**Duplicate registration is refused.** Several projects share one module-level
registry. Without the check, one could redefine another's metric and produce
reports that no longer compare to their own baselines. Pass `replace=True` when
you mean it.

**Latency is recorded whether or not the task reports it**, so every consumer
gets the cost side of the trade without asking. `p50` and `p95` are the only
built-in metrics, because they are the only ones every task shares. Domain
metrics belong to the project that defines them.

**A metric on only one side of a comparison is skipped, and the skip is
reported.** Adding a metric must not fail the build on the commit that
introduces it, and a silent skip is how a gate stops gating without anyone
noticing.

## What it does not do

- **No reference-based scorers built in.** BERTScore, ROUGE and F1 pull in heavy
  dependencies; they belong in the consuming project, registered through
  `register_metric`.
- **No slice-level breakdown yet.** `EvalCase.slice()` reads the metadata a
  dataset carries for this, but the runner does not group by it.
- **No token or cost accounting.** Latency is measured; spend is not.
- **No statistical significance testing.** Mean and standard deviation across
  seeds, which is enough to see that a difference is inside the noise, not
  enough to put a p-value on it.

## Consumers

- [agentic-rag-assistant-nutrition](https://github.com/mukundjaiswal/agentic-rag-assistant-nutrition)
  — grounded question answering with LLM-as-judge quality gates

## Development

```bash
make install-dev
make check          # ruff, mypy --strict, pytest
```

## License

MIT. See [LICENSE](LICENSE).
