"""Exception hierarchy."""


class EvalHarnessError(Exception):
    """Base class for every error raised by this package."""


class DatasetError(EvalHarnessError):
    """A dataset is missing, malformed or empty."""


class MetricError(EvalHarnessError):
    """A metric is unknown, or registered twice under the same name."""
