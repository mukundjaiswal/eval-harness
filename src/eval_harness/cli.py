"""Command-line interface.

The gate is exposed as a command so that CI in any consuming project can run it
without importing that project's code:

    eval-harness compare report.json --baseline baselines/v1.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from eval_harness import __version__
from eval_harness.metrics import Direction, get_metric, registered_metrics
from eval_harness.report import (
    EvaluationReport,
    MetricSummary,
    RegressionCheck,
    load_baseline,
)

app = typer.Typer(
    name="eval-harness",
    help="Evaluation datasets, metrics and a direction-aware regression gate.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def metrics() -> None:
    """List every registered metric with its direction."""
    names = registered_metrics()
    if not names:
        typer.echo("No metrics registered.")
        return
    for name in names:
        spec = get_metric(name)
        typer.echo(f"{name}  [{spec.direction}]")
        if spec.description:
            typer.echo(f"    {spec.description}")


@app.command()
def compare(
    report: Annotated[Path, typer.Argument(help="Report produced by a run.")],
    baseline: Annotated[Path, typer.Option(help="Committed baseline report.")],
    tolerance: Annotated[
        float, typer.Option(help="Allowed drift before a metric counts as regressed.")
    ] = 0.02,
) -> None:
    """Compare a report against a baseline, exiting non-zero on a regression."""
    current_raw = load_baseline(report)
    current = EvaluationReport(
        dataset=str(current_raw.get("dataset", str(report))),
        n_cases=int(current_raw.get("n_cases", 0)),
        seeds=list(current_raw.get("seeds", [])),
        metrics=[
            MetricSummary(
                name=str(m["name"]),
                direction=Direction(m.get("direction", Direction.HIGHER_IS_BETTER)),
                mean=m.get("mean"),
                stdev=m.get("stdev"),
                values=list(m.get("values", [])),
            )
            for m in current_raw.get("metrics", [])
        ],
    )

    result = RegressionCheck(tolerance=tolerance).compare(
        current, load_baseline(baseline)
    )
    typer.echo(result.report())

    if not result.passed:
        typer.echo(
            f"\n{len(result.regressions)} metric(s) regressed past "
            f"tolerance {tolerance}.",
            err=True,
        )
        sys.exit(1)
    typer.echo("\nNo regression against baseline.")


if __name__ == "__main__":
    app()
