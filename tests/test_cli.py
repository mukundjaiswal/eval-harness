import json

from typer.testing import CliRunner

from eval_harness.cli import app

runner = CliRunner()


def baseline_file(tmp_path, mean, direction="higher_is_better", name="baseline.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "dataset": "v1.jsonl",
                "n_cases": 10,
                "seeds": [0],
                "metrics": [
                    {
                        "name": "score",
                        "direction": direction,
                        "mean": mean,
                        "values": [mean],
                    }
                ],
            }
        )
    )
    return path


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0


def test_metrics_lists_direction():
    result = runner.invoke(app, ["metrics"])
    assert result.exit_code == 0
    assert "p95_latency_seconds" in result.stdout
    assert "lower_is_better" in result.stdout


def test_compare_passes_on_an_improvement(tmp_path):
    current = baseline_file(tmp_path, 0.95, name="current.json")
    base = baseline_file(tmp_path, 0.90)
    result = runner.invoke(app, ["compare", str(current), "--baseline", str(base)])
    assert result.exit_code == 0
    assert "No regression" in result.stdout


def test_compare_exits_non_zero_on_a_regression(tmp_path):
    """This exit code is what lets CI gate on quality."""
    current = baseline_file(tmp_path, 0.70, name="current.json")
    base = baseline_file(tmp_path, 0.90)
    result = runner.invoke(app, ["compare", str(current), "--baseline", str(base)])
    assert result.exit_code == 1


def test_compare_honours_direction_from_the_report(tmp_path):
    current = baseline_file(tmp_path, 1.4, direction="lower_is_better", name="c.json")
    base = baseline_file(tmp_path, 0.5, direction="lower_is_better")
    result = runner.invoke(app, ["compare", str(current), "--baseline", str(base)])
    assert result.exit_code == 1


def test_tolerance_is_configurable(tmp_path):
    current = baseline_file(tmp_path, 0.80, name="current.json")
    base = baseline_file(tmp_path, 0.90)
    assert (
        runner.invoke(
            app,
            ["compare", str(current), "--baseline", str(base), "--tolerance", "0.5"],
        ).exit_code
        == 0
    )
