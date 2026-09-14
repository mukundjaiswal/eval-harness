"""The evaluation runner.

Takes any callable that handles a case and returns observations, runs it over a
dataset once per seed, and summarises the result.

Multiple seeds are supported because a single run over a small dataset cannot
distinguish a real improvement from noise. Reporting a mean without a spread is
how a two-point difference gets presented as progress.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

from eval_harness.case import EvalCase
from eval_harness.metrics import MetricSpec, Observation, get_metric
from eval_harness.report import EvaluationReport, MetricSummary

TaskFunction = Callable[[EvalCase], Observation]
SeededTaskFunction = Callable[[EvalCase, int], Observation]


class EvaluationRunner:
    """Runs a task over a dataset and summarises the result."""

    def __init__(
        self,
        task: TaskFunction | SeededTaskFunction,
        *,
        metrics: Sequence[str | MetricSpec],
        seeds: Sequence[int] = (0,),
        pass_seed: bool = False,
    ) -> None:
        """Configure the run.

        Args:
            task: Handles one case and returns the observations to score.
            metrics: Registered metric names, or specs built at runtime.
            seeds: One full pass per seed.
            pass_seed: Call ``task(case, seed)`` instead of ``task(case)``, for
                a task whose own randomness has to be seeded.

        Raises:
            ValueError: No metrics or no seeds were given.
        """
        if not metrics:
            message = "At least one metric is required"
            raise ValueError(message)
        if not seeds:
            message = "At least one seed is required"
            raise ValueError(message)

        self._task = task
        self._specs = [
            m if isinstance(m, MetricSpec) else get_metric(m) for m in metrics
        ]
        self._seeds = list(seeds)
        self._pass_seed = pass_seed

    def run(
        self, cases: Sequence[EvalCase], *, dataset: Path | str
    ) -> EvaluationReport:
        """Score ``cases`` and return the report."""
        per_seed: list[list[Observation]] = []

        for seed in self._seeds:
            observations: list[Observation] = []
            for index, case in enumerate(cases, start=1):
                started = time.perf_counter()
                raw = (
                    self._task(case, seed)  # type: ignore[call-arg]
                    if self._pass_seed
                    else self._task(case)  # type: ignore[call-arg]
                )
                observation = dict(raw)
                # Latency is recorded whether or not the task reports it, so
                # every project gets the cost side of the trade for free.
                observation.setdefault("seconds", time.perf_counter() - started)
                observation["case_index"] = index
                observation["seed"] = seed
                observations.append(observation)
            per_seed.append(observations)

        summaries = [
            MetricSummary.from_runs(
                spec.name, spec.direction, [spec(run) for run in per_seed]
            )
            for spec in self._specs
        ]

        return EvaluationReport(
            dataset=str(dataset),
            n_cases=len(cases),
            seeds=list(self._seeds),
            metrics=summaries,
            observations=[o for run in per_seed for o in run],
        )
