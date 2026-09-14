"""A task-agnostic evaluation harness for LLM systems.

Nothing here knows what an agent, a classifier or a summarizer is. The harness
takes a callable, a dataset and a set of metrics; it runs them, summarises the
result across seeds, and compares that result against a committed baseline.

That generality is the point. The same harness drives a retrieval agent, a
text classifier and a summarizer without being rewritten, which is what makes
it a shared harness rather than three scripts that happen to look alike.
"""

from eval_harness.case import EvalCase, load_cases
from eval_harness.exceptions import DatasetError, EvalHarnessError, MetricError
from eval_harness.metrics import (
    Direction,
    MetricSpec,
    get_metric,
    mean_of,
    none_rate_of,
    percentile_of,
    rate_of,
    register,
    registered_metrics,
)
from eval_harness.report import EvaluationReport, MetricSummary, RegressionCheck
from eval_harness.runner import EvaluationRunner

__all__ = [
    "DatasetError",
    "Direction",
    "EvalCase",
    "EvalHarnessError",
    "EvaluationReport",
    "EvaluationRunner",
    "MetricError",
    "MetricSpec",
    "MetricSummary",
    "RegressionCheck",
    "__version__",
    "get_metric",
    "load_cases",
    "mean_of",
    "none_rate_of",
    "percentile_of",
    "rate_of",
    "register",
    "registered_metrics",
]
__version__ = "0.1.0"
