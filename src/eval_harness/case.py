"""Evaluation cases and dataset loading.

A case is an input, optionally a reference answer, and arbitrary metadata. The
input is deliberately not called a "question": the same loader serves a
retrieval agent, a classifier and a summarizer, and each calls its input
something different.

``input_key`` exists so a project that already documents its dataset with a
domain-specific key ("question", "narrative", "review") keeps that format
instead of rewriting its files to suit this package.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_harness.exceptions import DatasetError


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One unit of work to evaluate.

    Attributes:
        input: What the task under evaluation receives.
        reference: The gold answer, where one exists. Reference-free metrics
            such as an LLM judge do not need it.
        metadata: Everything else on the record, preserved verbatim. Use it for
            slice labels so that a dataset can carry them without a schema
            change here.
    """

    input: str
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def slice(self, key: str, default: str = "all") -> str:
        """Return a metadata value, for grouping results by slice."""
        return str(self.metadata.get(key, default))


def load_cases(
    path: Path | str,
    *,
    input_key: str = "input",
    reference_key: str = "reference",
) -> list[EvalCase]:
    """Load cases from a JSON Lines file.

    Args:
        path: The dataset file. Version it by filename and never edit a version
            in place; a report records the dataset it ran against, and two
            reports are comparable only if that string matches.
        input_key: Record key holding the input.
        reference_key: Record key holding the gold answer, if any.

    Raises:
        DatasetError: The file is missing, a line is malformed, a line lacks the
            input key, or the file holds no cases. A malformed line fails the
            load rather than being skipped: silently dropping cases changes the
            denominator of every metric computed afterwards.
    """
    resolved = Path(path)
    if not resolved.exists():
        message = f"Evaluation dataset not found: {resolved}"
        raise DatasetError(message)

    cases: list[EvalCase] = []
    for number, line in enumerate(_nonempty_lines(resolved), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            message = f"{resolved}:{number} is not valid JSON"
            raise DatasetError(message) from exc

        if not isinstance(record, dict):
            message = f"{resolved}:{number} is not a JSON object"
            raise DatasetError(message)

        value = record.pop(input_key, None)
        if not value:
            message = f"{resolved}:{number} has no {input_key!r} field"
            raise DatasetError(message)

        cases.append(
            EvalCase(
                input=str(value),
                reference=record.pop(reference_key, None),
                metadata=record,
            )
        )

    if not cases:
        message = f"{resolved} contains no cases"
        raise DatasetError(message)
    return cases


def _nonempty_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line
