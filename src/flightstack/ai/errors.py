"""Clear failure modes for optional FlightStack training facilities."""

from __future__ import annotations

TRAIN_EXTRA_COMMAND = 'python -m pip install -e ".[train]"'


class OptionalTrainingDependencyError(RuntimeError):
    """Raised when a Gymnasium/SB3-only path is requested without its extra."""


class PolicyNotTrainedError(RuntimeError):
    """Raised instead of silently substituting a non-learned pilot."""


class PolicySchemaError(RuntimeError):
    """Raised when a checkpoint is incompatible with this FlightStack seam."""
