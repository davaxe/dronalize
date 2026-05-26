from __future__ import annotations

import math
from collections import Counter

import pytest

from dronalize.processing.loading.assigner import StatelessWeightedAssigner


def test_weighted_assigner_is_deterministic() -> None:
    assigner_a = StatelessWeightedAssigner(groups=["A", "B", "C"], weights=[0.5, 0.3, 0.2], seed=42)
    assigner_b = StatelessWeightedAssigner(groups=["A", "B", "C"], weights=[0.5, 0.3, 0.2], seed=42)

    values = list(range(200))
    assert [assigner_a.assign(v) for v in values] == [assigner_b.assign(v) for v in values]


def test_weighted_assigner_matches_weights() -> None:
    assigner = StatelessWeightedAssigner(groups=["A", "B", "C"], weights=[0.5, 0.3, 0.2], seed=100)

    counts = Counter(assigner.assign(v) for v in range(10_000))
    total = sum(counts.values())

    assert math.isclose(counts["A"] / total, 0.5, abs_tol=0.02)
    assert math.isclose(counts["B"] / total, 0.3, abs_tol=0.02)
    assert math.isclose(counts["C"] / total, 0.2, abs_tol=0.02)


def test_weighted_assigner_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _ = StatelessWeightedAssigner(groups=["A", "B"], weights=[1.0, -1.0], seed=0)

    with pytest.raises(ValueError, match="At least one weight"):
        _ = StatelessWeightedAssigner(groups=["A", "B"], weights=[0.0, 0.0], seed=0)
