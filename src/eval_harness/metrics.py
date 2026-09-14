"""The metric registry.

A metric turns a list of per-case observations into one number, and declares
which direction is an improvement. Direction is registered with the metric
rather than held in a lookup table somewhere else, because a shared harness
cannot know its consumers' metrics: a table here would have to be edited every
time any project adds one.

The factories at the bottom exist so that a project's typical metric is one
line at registration rather than a function body.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from eval_harness.exceptions import MetricError

Observation = dict[str, Any]
MetricFn = Callable[[Sequence[Observation]], float | None]


class Direction(StrEnum):
    """Which way is an improvement."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """A registered metric."""

    name: str
    fn: MetricFn
    direction: Direction
    description: str = ""

    def __call__(self, observations: Sequence[Observation]) -> float | None:
        """Compute the metric."""
        return self.fn(observations)


_REGISTRY: dict[str, MetricSpec] = {}


def register(
    name: str,
    *,
    direction: Direction,
    description: str = "",
    replace: bool = False,
) -> Callable[[MetricFn], MetricFn]:
    """Register a metric function under ``name``.

    Raises:
        MetricError: ``name`` is already registered and ``replace`` is False.
            Silent replacement is the failure mode this guards against: three
            projects sharing one registry could otherwise redefine each other's
            metrics and produce reports that are not comparable to their own
            baselines.
    """

    def decorator(function: MetricFn) -> MetricFn:
        register_metric(
            name,
            function,
            direction=direction,
            description=description or (function.__doc__ or "").strip(),
            replace=replace,
        )
        return function

    return decorator


def register_metric(
    name: str,
    function: MetricFn,
    *,
    direction: Direction,
    description: str = "",
    replace: bool = False,
) -> MetricSpec:
    """Register a metric built at runtime, such as one from a factory."""
    if name in _REGISTRY and not replace:
        message = (
            f"Metric {name!r} is already registered. Pass replace=True if that "
            "is intended."
        )
        raise MetricError(message)
    spec = MetricSpec(
        name=name, fn=function, direction=direction, description=description
    )
    _REGISTRY[name] = spec
    return spec


def get_metric(name: str) -> MetricSpec:
    """Look up a registered metric."""
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        message = f"Unknown metric {name!r}. Registered: {known}"
        raise MetricError(message)
    return _REGISTRY[name]


def registered_metrics() -> list[str]:
    """Every registered metric name, sorted."""
    return sorted(_REGISTRY)


def unregister(name: str) -> None:
    """Remove a metric. Intended for test isolation."""
    _REGISTRY.pop(name, None)


# --- value extraction -------------------------------------------------------


def values_of(observations: Sequence[Observation], key: str) -> list[float]:
    """Numeric values of ``key``, skipping observations where it is absent.

    ``None`` is skipped rather than coerced to zero. Zero would read as "scored
    badly"; absent means "not measured", and conflating the two silently drags
    every average down.
    """
    out: list[float] = []
    for observation in observations:
        value = observation.get(key)
        if value is None:
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


# --- factories --------------------------------------------------------------


def mean_of(key: str) -> MetricFn:
    """Mean of a numeric field across cases that have it."""

    def metric(observations: Sequence[Observation]) -> float | None:
        values = values_of(observations, key)
        return statistics.fmean(values) if values else None

    return metric


def rate_of(key: str) -> MetricFn:
    """Share of cases where a field is truthy."""

    def metric(observations: Sequence[Observation]) -> float | None:
        if not observations:
            return None
        return sum(bool(o.get(key)) for o in observations) / len(observations)

    return metric


def none_rate_of(key: str) -> MetricFn:
    """Share of cases where a field is missing or ``None``.

    A harness-health metric rather than a quality one: if it is non-zero, every
    quality number computed over the same field is measured on a smaller sample
    than the case count suggests.
    """

    def metric(observations: Sequence[Observation]) -> float | None:
        if not observations:
            return None
        return sum(o.get(key) is None for o in observations) / len(observations)

    return metric


def percentile_of(key: str, quantile: float) -> MetricFn:
    """Percentile of a numeric field, by nearest rank."""
    if not 0.0 <= quantile <= 1.0:
        message = "quantile must be between 0 and 1"
        raise ValueError(message)

    def metric(observations: Sequence[Observation]) -> float | None:
        values = sorted(values_of(observations, key))
        if not values:
            return None
        index = min(round(quantile * (len(values) - 1)), len(values) - 1)
        return values[index]

    return metric


# --- built-ins --------------------------------------------------------------
#
# Only metrics every task shares. Domain metrics belong to the project that
# defines them, registered from that project's own module.

register_metric(
    "p50_latency_seconds",
    percentile_of("seconds", 0.50),
    direction=Direction.LOWER_IS_BETTER,
    description="Median wall-clock seconds per case.",
)

register_metric(
    "p95_latency_seconds",
    percentile_of("seconds", 0.95),
    direction=Direction.LOWER_IS_BETTER,
    description=(
        "95th-percentile wall-clock seconds per case. The tail is what users "
        "feel; a mean hides the cases that retried."
    ),
)
