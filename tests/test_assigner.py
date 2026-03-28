# pyright: standard

from __future__ import annotations

import random
from collections import Counter

import pytest

from dronalize.processing.ingest.assigner import StatelessWeightedAssigner


def test_deterministic_assigner_distribution_matches_weights() -> None:
    """Verify that the stateless assigner produces the expected distribution."""
    groups = ["A", "B", "C"]
    weights = [0.5, 0.3, 0.2]
    assigner = StatelessWeightedAssigner(groups=groups, weights=weights, seed=100)

    assignments = [assigner.assign(i) for i in range(10000)]
    counts = Counter(assignments)
    total = sum(counts.values())

    assert pytest.approx(counts["A"] / total, abs=0.02) == 0.5
    assert pytest.approx(counts["B"] / total, abs=0.02) == 0.3
    assert pytest.approx(counts["C"] / total, abs=0.02) == 0.2


def test_deterministic_assigner_multiple_keys() -> None:
    """Verify that different keys produce different assignments."""
    groups = ["A", "B", "C"]
    weights = [0.5, 0.3, 0.2]
    assigner = StatelessWeightedAssigner(groups=groups, weights=weights, seed=100)
    rng = random.Random(0)
    assignments = [assigner.assign(i, rng.randint(0, 1000)) for i in range(10000)]

    counts = Counter(assignments)
    total = sum(counts.values())
    assert pytest.approx(counts["A"] / total, abs=0.01) == 0.5
    assert pytest.approx(counts["B"] / total, abs=0.01) == 0.3
    assert pytest.approx(counts["C"] / total, abs=0.01) == 0.2


def test_deterministic_assigner_same_assignments() -> None:
    """Verify that the same key produces the same assignment across same seed."""
    groups = ["A", "B", "C"]
    weights = [0.5, 0.3, 0.2]

    assigner1 = StatelessWeightedAssigner(groups=groups, weights=weights, seed=42)
    assigner2 = StatelessWeightedAssigner(groups=groups, weights=weights, seed=42)
    rng = random.Random(0)
    values = [rng.randint(0, 10000000) for _ in range(100)]
    assignments1 = [assigner1.assign(value) for value in values]
    assignments2 = [assigner2.assign(value) for value in values]

    assert assignments1 == assignments2


def test_deterministic_assigner_different_assignments() -> None:
    """Verify that different seeds produce different assignments for the same values."""
    groups = ["A", "B", "C"]
    weights = [0.5, 0.3, 0.2]

    assigner1 = StatelessWeightedAssigner(groups=groups, weights=weights, seed=42)
    assigner2 = StatelessWeightedAssigner(groups=groups, weights=weights, seed=43)
    rng = random.Random(0)
    values = [rng.randint(0, 10000000) for _ in range(100)]
    assignments1 = [assigner1.assign(value) for value in values]
    assignments2 = [assigner2.assign(value) for value in values]

    assert assignments1 != assignments2
