import json

from eval_harness.metrics import Direction
from eval_harness.report import (
    EvaluationReport,
    MetricSummary,
    RegressionCheck,
    load_baseline,
)

UP = Direction.HIGHER_IS_BETTER
DOWN = Direction.LOWER_IS_BETTER


def report(**metrics) -> EvaluationReport:
    """Build a report from ``name=(mean, direction)`` pairs."""
    return EvaluationReport(
        dataset="v1.jsonl",
        n_cases=10,
        seeds=[0],
        metrics=[
            MetricSummary(
                name=name, direction=direction, mean=mean, stdev=None, values=[mean]
            )
            for name, (mean, direction) in metrics.items()
        ],
    )


def serialised(**metrics) -> dict:
    return report(**metrics).to_dict(include_observations=False)


def test_multi_seed_summary_reports_a_spread():
    summary = MetricSummary.from_runs("m", UP, [0.8, 0.9, 0.85])
    assert summary.stdev is not None
    assert "+/-" in summary.display()


def test_metric_with_no_usable_values_is_unavailable_not_zero():
    summary = MetricSummary.from_runs("m", UP, [None, None])
    assert summary.mean is None
    assert summary.display().endswith("n/a")


def test_a_drop_in_a_higher_is_better_metric_regresses():
    result = RegressionCheck(tolerance=0.02).compare(
        report(score=(0.80, UP)), serialised(score=(0.90, UP))
    )
    assert not result.passed
    assert result.regressions[0].name == "score"


def test_a_drop_within_tolerance_does_not_regress():
    result = RegressionCheck(tolerance=0.02).compare(
        report(score=(0.89, UP)), serialised(score=(0.90, UP))
    )
    assert result.passed


def test_an_improvement_is_never_a_regression():
    result = RegressionCheck().compare(
        report(score=(0.95, UP)), serialised(score=(0.70, UP))
    )
    assert result.passed
    assert result.comparisons[0].drift < 0


def test_a_rise_in_a_lower_is_better_metric_regresses():
    """More retries per case is worse, even though the number went up."""
    result = RegressionCheck(tolerance=0.02).compare(
        report(retries=(1.4, DOWN)), serialised(retries=(0.5, DOWN))
    )
    assert not result.passed
    assert result.regressions[0].name == "retries"


def test_direction_comes_from_the_current_report_not_a_lookup_table():
    """A baseline outliving the code that wrote it must still be comparable."""
    baseline = {"metrics": [{"name": "retries", "mean": 0.5}]}
    result = RegressionCheck(tolerance=0.02).compare(
        report(retries=(1.4, DOWN)), baseline
    )
    assert not result.passed


def test_a_metric_absent_from_the_baseline_is_skipped_not_failed():
    result = RegressionCheck().compare(
        report(score=(0.9, UP), brand_new=(0.1, UP)), serialised(score=(0.9, UP))
    )
    assert result.passed
    assert "brand_new" in result.skipped


def test_a_metric_dropped_from_the_run_is_reported_as_skipped():
    result = RegressionCheck().compare(
        report(score=(0.9, UP)), serialised(score=(0.9, UP), retired=(0.1, UP))
    )
    assert result.passed
    assert "retired" in result.skipped


def test_skips_are_visible_in_the_gate_report():
    result = RegressionCheck().compare(
        report(score=(0.9, UP), brand_new=(0.1, UP)), serialised(score=(0.9, UP))
    )
    assert "skipped" in result.report()


def test_report_round_trips_through_disk(tmp_path):
    path = tmp_path / "nested" / "report.json"
    report(score=(0.9, UP)).write(path)
    payload = json.loads(path.read_text())
    assert payload["dataset"] == "v1.jsonl"
    assert payload["metrics"][0]["direction"] == "higher_is_better"
    assert load_baseline(path)["n_cases"] == 10
