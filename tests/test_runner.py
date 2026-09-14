import pytest

from eval_harness.case import EvalCase
from eval_harness.metrics import Direction, MetricSpec, mean_of
from eval_harness.runner import EvaluationRunner

SCORE = MetricSpec(
    name="mean_score", fn=mean_of("score"), direction=Direction.HIGHER_IS_BETTER
)


def test_runs_every_case_once_per_seed(cases):
    seen = []
    EvaluationRunner(
        lambda case: seen.append(case.input) or {"score": 1.0},
        metrics=[SCORE],
        seeds=[0, 1],
    ).run(cases, dataset="v1.jsonl")
    assert seen == ["a", "b", "a", "b"]


def test_multiple_seeds_produce_a_variance_estimate(cases):
    scores = iter([0.8, 0.8, 0.6, 0.6])
    report = EvaluationRunner(
        lambda case: {"score": next(scores)}, metrics=[SCORE], seeds=[0, 1]
    ).run(cases, dataset="v1.jsonl")

    summary = report.metric("mean_score")
    assert summary.values == [0.8, 0.6]
    assert summary.stdev is not None


def test_single_seed_reports_no_spread(cases):
    report = EvaluationRunner(lambda case: {"score": 0.9}, metrics=[SCORE]).run(
        cases, dataset="v1.jsonl"
    )
    assert report.metric("mean_score").stdev is None
    assert "no variance estimate" in report.summary()


def test_latency_is_recorded_without_the_task_reporting_it(cases):
    report = EvaluationRunner(
        lambda case: {"score": 0.9}, metrics=["p50_latency_seconds"]
    ).run(cases, dataset="v1.jsonl")
    assert report.metric("p50_latency_seconds").mean is not None


def test_a_task_may_report_its_own_latency(cases):
    report = EvaluationRunner(
        lambda case: {"score": 0.9, "seconds": 5.0}, metrics=["p50_latency_seconds"]
    ).run(cases, dataset="v1.jsonl")
    assert report.metric("p50_latency_seconds").mean == pytest.approx(5.0)


def test_seed_can_be_passed_to_the_task(cases):
    seeds_seen = []
    EvaluationRunner(
        lambda case, seed: seeds_seen.append(seed) or {"score": 1.0},
        metrics=[SCORE],
        seeds=[3, 7],
        pass_seed=True,
    ).run(cases, dataset="v1.jsonl")
    assert set(seeds_seen) == {3, 7}


def test_observations_carry_their_seed_and_index(cases):
    report = EvaluationRunner(
        lambda case: {"score": 0.9}, metrics=[SCORE], seeds=[7]
    ).run(cases, dataset="v1.jsonl")
    assert {o["seed"] for o in report.observations} == {7}
    assert [o["case_index"] for o in report.observations] == [1, 2]


def test_metrics_may_be_named(cases, scratch_metric):
    report = EvaluationRunner(lambda case: {}, metrics=[scratch_metric]).run(
        cases, dataset="v1.jsonl"
    )
    assert report.metric(scratch_metric).mean == 1.0


def test_at_least_one_metric_is_required():
    with pytest.raises(ValueError, match="metric"):
        EvaluationRunner(lambda case: {}, metrics=[])


def test_at_least_one_seed_is_required():
    with pytest.raises(ValueError, match="seed"):
        EvaluationRunner(lambda case: {}, metrics=[SCORE], seeds=[])


def test_dataset_name_is_recorded_on_the_report(cases):
    report = EvaluationRunner(lambda case: {"score": 1.0}, metrics=[SCORE]).run(
        cases, dataset="datasets/v3.jsonl"
    )
    assert report.dataset == "datasets/v3.jsonl"
    assert isinstance(cases[0], EvalCase)
