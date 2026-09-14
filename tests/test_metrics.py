import pytest

from eval_harness.exceptions import MetricError
from eval_harness.metrics import (
    Direction,
    get_metric,
    mean_of,
    none_rate_of,
    percentile_of,
    rate_of,
    register,
    register_metric,
    registered_metrics,
    unregister,
    values_of,
)


def test_built_in_latency_metrics_are_registered():
    assert "p50_latency_seconds" in registered_metrics()
    assert get_metric("p95_latency_seconds").direction is Direction.LOWER_IS_BETTER


def test_unknown_metric_lists_what_is_available():
    with pytest.raises(MetricError, match="Registered"):
        get_metric("does_not_exist")


def test_duplicate_registration_is_refused():
    """Three projects share this registry; silent replacement is the hazard."""
    register_metric("dupe", mean_of("x"), direction=Direction.HIGHER_IS_BETTER)
    try:
        with pytest.raises(MetricError, match="already registered"):
            register_metric("dupe", mean_of("y"), direction=Direction.HIGHER_IS_BETTER)
    finally:
        unregister("dupe")


def test_duplicate_registration_is_allowed_when_explicit():
    register_metric("dupe2", mean_of("x"), direction=Direction.HIGHER_IS_BETTER)
    try:
        register_metric(
            "dupe2", mean_of("y"), direction=Direction.LOWER_IS_BETTER, replace=True
        )
        assert get_metric("dupe2").direction is Direction.LOWER_IS_BETTER
    finally:
        unregister("dupe2")


def test_decorator_registers_with_a_direction():
    @register("decorated", direction=Direction.LOWER_IS_BETTER)
    def _decorated(observations):
        return float(len(observations))

    try:
        assert get_metric("decorated").direction is Direction.LOWER_IS_BETTER
        assert get_metric("decorated")([{}, {}]) == 2.0
    finally:
        unregister("decorated")


def test_decorator_takes_its_description_from_the_docstring():
    @register("described", direction=Direction.HIGHER_IS_BETTER)
    def _described(observations):
        """What this measures."""
        return 1.0

    try:
        assert get_metric("described").description == "What this measures."
    finally:
        unregister("described")


def test_values_of_skips_none_rather_than_counting_it_as_zero():
    """Zero reads as 'scored badly'; absent means 'not measured'."""
    assert values_of([{"x": 1.0}, {"x": None}, {"x": 3.0}], "x") == [1.0, 3.0]


def test_values_of_skips_unparseable_entries():
    assert values_of([{"x": "abc"}, {"x": 2}], "x") == [2.0]


def test_mean_of_ignores_missing_values():
    assert mean_of("x")([{"x": 1.0}, {"x": None}, {"x": 0.0}]) == pytest.approx(0.5)


def test_mean_of_nothing_is_none_not_zero():
    assert mean_of("x")([{"x": None}]) is None


def test_rate_of_counts_truthy_values():
    assert rate_of("flag")([{"flag": True}, {"flag": False}]) == pytest.approx(0.5)


def test_none_rate_of_measures_missing_values():
    assert none_rate_of("x")([{"x": None}, {"x": 0.9}]) == pytest.approx(0.5)


def test_percentile_is_monotonic():
    observations = [{"s": float(i)} for i in range(1, 21)]
    assert percentile_of("s", 0.95)(observations) >= percentile_of("s", 0.5)(
        observations
    )


def test_percentile_of_empty_is_none():
    assert percentile_of("s", 0.5)([]) is None


@pytest.mark.parametrize("quantile", [-0.1, 1.1])
def test_percentile_quantile_must_be_a_fraction(quantile):
    with pytest.raises(ValueError, match="quantile"):
        percentile_of("s", quantile)
