"""Reports and the regression gate.

A report carries each metric's **direction** with it. That is deliberate: the
alternative is looking direction up in the registry at comparison time, which
breaks the moment a baseline outlives the code that produced it. A report that
describes itself can be compared a year later by a process that has never
imported the project that wrote it.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eval_harness.metrics import Direction


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """One metric across every seed of a run."""

    name: str
    direction: Direction
    mean: float | None
    stdev: float | None
    values: list[float]

    @classmethod
    def from_runs(
        cls,
        name: str,
        direction: Direction,
        values: Sequence[float | None],
        *,
        precision: int = 4,
    ) -> MetricSummary:
        """Summarise per-seed values, tolerating seeds that produced nothing."""
        usable = [v for v in values if v is not None]
        if not usable:
            return cls(name=name, direction=direction, mean=None, stdev=None, values=[])
        return cls(
            name=name,
            direction=direction,
            mean=round(statistics.fmean(usable), precision),
            stdev=(
                round(statistics.stdev(usable), precision) if len(usable) > 1 else None
            ),
            values=[round(v, precision) for v in usable],
        )

    def display(self) -> str:
        """Render for a terminal."""
        if self.mean is None:
            return f"{self.name}: n/a"
        if self.stdev is None:
            return f"{self.name}: {self.mean} (single seed, no variance estimate)"
        return f"{self.name}: {self.mean} +/- {self.stdev}"


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """The result of one evaluation run."""

    dataset: str
    n_cases: int
    seeds: list[int]
    metrics: list[MetricSummary]
    observations: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def metric(self, name: str) -> MetricSummary | None:
        """Look up one metric summary."""
        return next((m for m in self.metrics if m.name == name), None)

    def to_dict(self, *, include_observations: bool = True) -> dict[str, Any]:
        """Serialise the report."""
        payload = asdict(self)
        if not include_observations:
            payload.pop("observations")
        return payload

    def write(self, path: Path | str, *, include_observations: bool = True) -> None:
        """Write the report as JSON, creating parent directories."""
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(
                self.to_dict(include_observations=include_observations),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"dataset: {self.dataset}",
            f"cases:   {self.n_cases}",
            f"seeds:   {', '.join(str(s) for s in self.seeds)}",
            "",
            *(f"  {m.display()}" for m in self.metrics),
        ]
        if len(self.seeds) == 1:
            lines += [
                "",
                "Single seed: a difference smaller than run-to-run noise is not "
                "distinguishable. Pass more seeds for a variance estimate.",
            ]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Comparison:
    """What changed for one metric between a baseline and a run."""

    name: str
    baseline: float
    current: float
    direction: Direction
    drift: float
    regressed: bool

    def describe(self) -> str:
        """One line, readable in CI output."""
        verdict = "REGRESSED" if self.regressed else "ok"
        return (
            f"{self.name}: {self.baseline} -> {self.current} "
            f"({self.direction}, drift {self.drift:+.4f}) {verdict}"
        )


@dataclass(frozen=True, slots=True)
class GateResult:
    """The outcome of comparing a run against a baseline."""

    comparisons: list[Comparison]
    skipped: list[str]

    @property
    def regressions(self) -> list[Comparison]:
        """Only the metrics that moved the wrong way past tolerance."""
        return [c for c in self.comparisons if c.regressed]

    @property
    def passed(self) -> bool:
        """Whether the run is clear to merge."""
        return not self.regressions

    def report(self) -> str:
        """Full comparison, readable in CI output."""
        lines = [c.describe() for c in self.comparisons]
        if self.skipped:
            lines.append(
                "skipped (absent from one side): " + ", ".join(sorted(self.skipped))
            )
        return "\n".join(lines) or "no comparable metrics"


@dataclass(frozen=True, slots=True)
class RegressionCheck:
    """Compares a report against a committed baseline."""

    tolerance: float = 0.02

    def compare(
        self, current: EvaluationReport, baseline: dict[str, Any]
    ) -> GateResult:
        """Compare ``current`` against a serialised ``baseline``.

        A metric present on only one side is skipped rather than failed, so
        adding a metric does not break the build on the commit that introduces
        it, and removing one does not either. Skips are reported, not hidden.
        """
        baseline_metrics = {
            m["name"]: m for m in baseline.get("metrics", []) if isinstance(m, dict)
        }
        comparisons: list[Comparison] = []
        skipped: list[str] = []

        for summary in current.metrics:
            before_entry = baseline_metrics.pop(summary.name, None)
            before = before_entry.get("mean") if before_entry else None
            after = summary.mean

            if before is None or after is None:
                skipped.append(summary.name)
                continue

            drift = (
                float(before) - after
                if summary.direction is Direction.HIGHER_IS_BETTER
                else after - float(before)
            )
            comparisons.append(
                Comparison(
                    name=summary.name,
                    baseline=float(before),
                    current=after,
                    direction=summary.direction,
                    drift=drift,
                    regressed=drift > self.tolerance,
                )
            )

        skipped.extend(baseline_metrics)
        return GateResult(comparisons=comparisons, skipped=skipped)


def load_baseline(path: Path | str) -> dict[str, Any]:
    """Load a committed baseline report."""
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data
