"""Small dependency-free space objects matching the Gymnasium Box surface."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class BoxSpace:
    """A finite Box used by the core environment before Gymnasium is installed.

    It intentionally exposes the small `low`/`high`/`shape`/`contains` surface
    that callers need.  :func:`flightstack.ai.environment.make_gymnasium_env`
    replaces it with a real ``gymnasium.spaces.Box`` when the optional training
    extra is installed.
    """

    def __init__(
        self,
        low: float,
        high: float,
        shape: Sequence[int],
        *,
        dtype: type[np.float32] = np.float32,
    ) -> None:
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise ValueError("Box bounds must be finite and ordered")
        normalized_shape = tuple(int(item) for item in shape)
        if not normalized_shape or any(item <= 0 for item in normalized_shape):
            raise ValueError("Box shape must contain positive dimensions")
        self.low: NDArray[np.float32] = np.full(normalized_shape, low, dtype=dtype)
        self.high: NDArray[np.float32] = np.full(normalized_shape, high, dtype=dtype)
        self.shape = normalized_shape
        self.dtype = dtype

    def contains(self, value: object) -> bool:
        """Return whether ``value`` has the correct shape and finite bounds."""
        try:
            array = np.asarray(value, dtype=self.dtype)
        except (TypeError, ValueError):
            return False
        return bool(
            array.shape == self.shape
            and np.all(np.isfinite(array))
            and np.all(array >= self.low)
            and np.all(array <= self.high)
        )

    def sample(self, rng: np.random.Generator | None = None) -> NDArray[np.float32]:
        """Sample a bounded value without creating an implicit global RNG."""
        generator = np.random.default_rng() if rng is None else rng
        return np.asarray(generator.uniform(self.low, self.high), dtype=self.dtype)
