# pyright: standard

from __future__ import annotations

import random
from collections import Counter

import pytest

from dronalize.execution.assigner import DeckWeightedAssigner, StatelessWeightedAssigner


def test_exact_distribution_for_round_multiples() -> None:
    """Verify that group distribution exactly matches weights for multiples of round_size."""
    groups = ["A", "B", "C"]
    weights = [0.25, 0.25, 0.50]
    round_size = 12
    # Expectation per round: 3 'A', 3 'B', 6 'C'

    splitter = DeckWeightedAssigner(groups=groups, weights=weights, round_size=round_size, seed=42)

    # Pull exactly 3 rounds worth of data (36 samples)
    samples = list(splitter.take(round_size * 3))
    counts = Counter(samples)

    assert counts["A"] == 9
    assert counts["B"] == 9
    assert counts["C"] == 18


def test_deterministic_seed() -> None:
    """Verify determinism across identical instances using the same seed."""
    groups = ["X", "Y", "Z"]
    weights = [0.1, 0.3, 0.6]

    splitter_1 = DeckWeightedAssigner(groups, weights, seed=123)
    splitter_2 = DeckWeightedAssigner(groups, weights, seed=123)

    samples_1 = list(splitter_1.take(50))
    samples_2 = list(splitter_2.take(50))

    assert samples_1 == samples_2


def test_different_seeds_produce_different_streams() -> None:
    """Ensure different seeds result in different output streams."""
    groups = [1, 2, 3, 4]

    splitter_1 = DeckWeightedAssigner(groups, seed=111)
    splitter_2 = DeckWeightedAssigner(groups, seed=222)

    samples_1 = list(splitter_1.take(50))
    samples_2 = list(splitter_2.take(50))

    assert samples_1 != samples_2


def test_uniform_distribution_without_weights() -> None:
    """Test that omitting weights defaults to a uniform distribution."""
    groups = ["Up", "Down"]
    round_size = 10

    splitter = DeckWeightedAssigner(groups, round_size=round_size)
    samples = list(splitter.take(round_size * 5))
    counts = Counter(samples)

    # Expected exactly 25 of each
    assert counts["Up"] == 25
    assert counts["Down"] == 25


def test_single_group_repeats_indefinitely() -> None:
    """Ensure a single group is yielded continuously."""
    splitter = DeckWeightedAssigner(["A"], round_size=5)
    samples = list(splitter.take(15))

    assert len(samples) == 15
    assert all(s == "A" for s in samples)


def test_negative_weights_raise_error() -> None:
    """Ensure a ValueError is raised when any weight is negative."""
    with pytest.raises(ValueError, match="All weights must be non-negative"):
        _ = DeckWeightedAssigner(["A", "B"], weights=[1.0, -0.5])


def test_zero_sum_weights_raise_error() -> None:
    """Ensure a ValueError is raised when all weights evaluate to zero."""
    # Using a partial match for the error string since the numpy array string representation
    # gets appended to the end of the error message.
    with pytest.raises(ValueError, match="At least one weight must be greater than zero"):
        _ = DeckWeightedAssigner(["A", "B", "C"], weights=[0.0, 0.0, 0.0])


def test_zero_weight_excludes_group() -> None:
    """Ensure a group with a zero weight is never yielded."""
    groups = ["A", "B", "C"]
    weights = [0.5, 0.0, 0.5]
    round_size = 10

    splitter = DeckWeightedAssigner(groups=groups, weights=weights, round_size=round_size, seed=42)

    # Take multiple rounds of samples to be certain
    samples = list(splitter.take(round_size * 5))
    counts = Counter(samples)

    assert "B" not in counts
    assert counts["A"] == 25
    assert counts["C"] == 25


def test_iterator_protocol() -> None:
    """Test that the class instances can be used as infinite iterators."""
    splitter = DeckWeightedAssigner(["A", "B"])

    iterator = iter(splitter)
    samples = [next(iterator) for _ in range(5)]

    assert len(samples) == 5
    assert all(s in {"A", "B"} for s in samples)


def test_non_unique_groups_raise_error() -> None:
    """Ensure a ValueError is raised when initialized with duplicate groups."""
    with pytest.raises(ValueError, match="Groups must be unique"):
        _ = DeckWeightedAssigner(["A", "A", "B"])


def test_weight_length_mismatch_raises_error() -> None:
    """Ensure a ValueError is raised if the number of weights does not match the groups."""
    with pytest.raises(ValueError, match="Number of weights must match number of groups"):
        _ = DeckWeightedAssigner(["A", "B"], weights=[0.5, 0.3, 0.2])


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
