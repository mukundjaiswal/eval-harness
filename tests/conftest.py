"""Shared fixtures."""

from __future__ import annotations

import pytest

from eval_harness.case import EvalCase
from eval_harness.metrics import Direction, register_metric, unregister


@pytest.fixture
def cases() -> list[EvalCase]:
    """Two trivial cases."""
    return [EvalCase(input="a"), EvalCase(input="b")]


@pytest.fixture
def scratch_metric() -> str:
    """Register a metric for one test and remove it afterwards.

    The registry is module-global, so a test that leaves one behind changes what
    later tests see.
    """
    name = "_scratch_score"
    register_metric(
        name,
        lambda observations: 1.0 if observations else None,
        direction=Direction.HIGHER_IS_BETTER,
        replace=True,
    )
    yield name
    unregister(name)
