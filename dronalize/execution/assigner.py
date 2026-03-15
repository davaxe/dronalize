from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from dronalize.categories import DatasetSplit

T_co = TypeVar("T_co", covariant=True)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class Assigner(Protocol, Generic[T_co]):
    """Common interface for group assigners."""

    def assign(self, *values: int | str) -> T_co:
        """Get assignment for the provided values.

        Parameters
        ----------
        *values : int or str
            A set of integer or string values that will be used to compute the
            assignment (implementation dependent).

        Returns
        -------
        T
            The assigned group for the provided values.
        """
        ...


SplitAssigner = Assigner[DatasetSplit]
"""A group assigner for dataset splits (train/val/test)."""


class StatelessWeightedAssigner(Assigner[T_co]):
    """Stateless group assigner based on hashing.

    Given a set of groups and optional weights, this class provides a
    deterministic assignment of any combination of integer values to one of the
    groups. The same seed and the same set of values will always yield the same
    group assignment. The weights determine the probability distribution over
    the groups.

    The advantage of this approach is that it does not require maintaining any
    internal, which allows for deterministic group assignment for the same set
    of input values and seed. However, the downside is that it will only
    converge to the specified distribution in the limit of infinite assignment,
    and could differ significantly from the specified distribution for small
    numbers of assignments.

    In contrast, the `WeightedAssigner` class maintains internal state to ensure
    that the distrubtion is always close, at the cost of requirng the order of
    assignment to be identical to get deterministic results across independent
    runs.

    """

    def __init__(
        self,
        groups: Sequence[T_co],
        weights: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize and construct the StatelessWeightedAssigner.

        Parameters
        ----------
        groups : Sequence[T]
            A sequence of unique groups to assign the stream into.
        weights : Sequence[float], optional
            A sequence of weights corresponding to each group. If None,
            groups are treated as equally weighted. The weights will be
            normalized to sum to 1.
        seed : int, optional
            An optional random seed for deterministic group assignment. If None,
            a random seed will be generated internally.

        Raises
        ------
        ValueError
            If groups are not unique, if the number of weights does not
            match the number of groups, if any weight is negative, or if all
            weights are zero.
        """
        self._seed: int = (
            seed if seed is not None else np.random.default_rng().integers(0, 2**31 - 1)
        )

        self._groups: list[T_co] = list(dict.fromkeys(groups))
        if len(groups) != len(self._groups):
            msg = (
                f"Groups must be unique. Found {len(groups)} elements "
                f"but only {len(self._groups)} unique groups."
            )
            raise ValueError(msg)
        if weights is not None and len(weights) != len(self._groups):
            msg = (
                f"Number of weights must match number of groups. "
                f"Found {len(weights)} weights but {len(self._groups)} groups."
            )
            raise ValueError(msg)

        self._weights: npt.NDArray[np.float64] = (
            np.array(weights, dtype=np.float64)
            if weights is not None
            else np.ones(len(self._groups), dtype=np.float64)
        )

        if np.any(self._weights < 0):
            msg = "All weights must be non-negative."
            raise ValueError(msg)

        weights_sum = float(self._weights.sum())

        if weights_sum == 0:
            msg = f"At least one weight must be greater than zero got {self._weights}."
            raise ValueError(msg)

        self._weights /= weights_sum
        self._cumulative_weights: npt.NDArray[np.float64] = np.cumsum(self._weights)
        self._cumulative_weights[-1] = 1.0

    @override
    def assign(self, *values: int | str) -> T_co:
        """Assign a group based on the provided values.

        Given the same seed and the same set of values, this method will always
        return the same assignment.

        Parameters
        ----------
        *values : int
            A set of integer values that will be used to compute a hash for
            group assignment.

        """
        hash_float = _uniform_u64(self._seed, *values)
        idx = int(np.searchsorted(self._cumulative_weights, hash_float, side="right"))
        return self._groups[idx]


def _uniform_u64(seed: int, *values: int | str) -> float:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(seed).encode("utf-8"))
    h.update(b"|")
    for v in values:
        h.update(repr(v).encode("utf-8"))
        h.update(b"|")
    digest = h.digest()
    x = int.from_bytes(digest[:8], "big", signed=False)
    return x / 2**64


class DeckWeightedAssigner(Assigner[T_co]):
    """A class for assigning a stream of data into groups.

    The `DeckWeightedAssigner takes a sequence of unique groups and optional
    weights, and provides an infinite stream of group assignments. The groups
    are shuffled in rounds, where each round consists of a fixed number of
    samples (`round_size`) before reshuffling. The number of pre-generated
    rounds can be specified with `rounds` parameter.

    It works in the following steps:
    1. Create a 2D array (deck) where each row is a shuffled sequence of
    group entries where the number of entries for each group is determined
    by the provided weights and the specified `round_size`.
    2. Keep an index for the current position in the round.
    3. Get an entry from the round and increment the index. If the index
    exceed the round size, reset the index and randomly select a new row from
    the deck for the next round.
    4. Repeat step 3 indefinitely to provide an infinite stream of group
    assignments.

    Notes
    -----
    This implementation will guarantee that the distribution of groups will
    exactly follow the specified weights if:
    1. The number of groups assigned is a multiple of the `round_size`.
    2. The `round_size` is sufficiently large to allow for a exact integer count
    of each group based on the weights. For example if `round_size` is 4 only
    the weights that are multiples of 0,25 can be represented exactly. If the
    `round_size` is 100, then weights that are multiples of 0.01 can be
    represented exactly.

    In all other cases the implementation will be as close as possible to the
    specified weights and in the limit when number of distributed groups goes to
    infinity the distribution will converge to the uniform distribution defined
    by the weights.

    """

    def __init__(
        self,
        groups: Sequence[T_co],
        weights: Sequence[float] | None = None,
        round_size: int = 100,
        rounds: int = 10,
        seed: int | None = None,
    ) -> None:
        """Initialize and construct the WeightedAssigner.

        Parameters
        ----------
        groups : Sequence[T]
            A sequence of unique groups to assign the stream into.
        weights : Sequence[float], optional
            A sequence of weights corresponding to each group. If None,
            groups are treated as equally weighted. The weights will be
            normalized to sum to 1.
        round_size : int, optional
            The number of samples in each round before reshuffling.
        rounds : int, optional
            The number of pre-generated rounds of shuffled group indices.

        Raises
        ------
        ValueError
            If groups are not unique, if the number of weights does not
            match the number of groups, if any weight is negative, or if all
            weights are zero.

        """
        # fromkeys preserves order and removes duplicates
        self._groups: list[T_co] = list(dict.fromkeys(groups))
        if len(groups) != len(self._groups):
            msg = (
                f"Groups must be unique. Found {len(groups)} elements "
                f"but only {len(self._groups)} unique groups."
            )
            raise ValueError(msg)
        if weights is not None and len(weights) != len(self._groups):
            msg = (
                f"Number of weights must match number of groups. "
                f"Found {len(weights)} weights but {len(self._groups)} groups."
            )
            raise ValueError(msg)

        self._index_to_group: list[T_co] = list(self._groups)
        self._weights: npt.NDArray[np.float64] = (
            np.array(weights, dtype=np.float64)
            if weights is not None
            else np.ones(len(self._groups), dtype=np.float64)
        )

        if np.any(self._weights < 0):
            msg = "All weights must be non-negative."
            raise ValueError(msg)

        weights_sum = float(self._weights.sum())

        if weights_sum == 0:
            msg = f"At least one weight must be greater than zero got {self._weights}."
            raise ValueError(msg)

        self._weights /= weights_sum

        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._deck: npt.NDArray[np.int32] = _generate_shuffled_deck(
            self._weights, round_size, rounds, self._rng
        )
        self._index: int = 0
        self._round_index: int = 0

    @override
    def assign(self, *values: int | str) -> T_co:
        """Assign a group based on the provided values.

        The `values` are ignored in this implementation since the assignment is
        based on the shuffled deck and internal state. The same seed and the
        same order of calls to `assign` will yield the same sequence of group
        assignments.

        Parameters
        ----------
        *values : int or str
            A set of integer or string values that are ignored in this implementation.

        """
        _ = values  # Unused since assignment is based on internal state
        return self.next()

    def next(self) -> T_co:
        """Next group in the stream.

        Returns
        -------
            The next randomly selected group based on the current round's
            shuffled deck.

        Examples
        --------
        >>> assigner = DeckWeightedAssigner([1, 2], seed=42, round_size=2)
        >>> assigner.next() # First is 2 by chance
        2
        >>> assigner.next() # Second is guaranteed 1, since `round_size` is 2
        1
        """
        index = self._index
        value = int(self._deck[self._round_index, index])
        self._update()
        return self._index_to_group[value]

    def take(self, n: int) -> Iterator[T_co]:
        """Take the next `n` groups as an iterator.

        Parameters
        ----------
        n : int
            The number of groups to yield from the stream.

        Returns
        -------
        Iterator[T]
            An iterator that yields the next `n` groups based on the current
            round's shuffled deck.

        """
        return (self.next() for _ in range(n))

    def __iter__(self) -> Iterator[T_co]:
        """Return an infinite iterator that yields groups from the stream.

        Returns
        -------
        Iterator[T]
            An infinite iterator that yields groups based on the current round's
            shuffled deck.

        """

        def infinite_generator() -> Iterator[T_co]:
            while True:
                yield self.next()

        return infinite_generator()

    def _update(self) -> None:
        self._index += 1
        if self._index >= self._deck.shape[1]:
            self._index = 0
            # Random row selection for the next round
            self._round_index = int(self._rng.integers(0, int(self._deck.shape[0])))


def _generate_shuffled_deck(
    weights: npt.NDArray[np.float64],
    round_size: int,
    rounds: int,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.int32]:
    # Calculate exact integer counts for one round
    float_counts = weights * round_size
    counts = np.floor(float_counts).astype(np.int32)

    # Handle rounding remainders to strictly enforce round_size
    remainder = round_size - int(counts.sum())
    if remainder > 0:
        # Distribute remainder to the groups with the largest fractional parts
        fractional_parts = float_counts - counts
        top_indices = np.argsort(fractional_parts)[-remainder:]
        counts[top_indices] += 1

    # Create the base 1D array for a single round
    n_groups = len(weights)
    base_round = np.repeat(np.arange(n_groups, dtype=np.int32), counts)

    # Duplicate the base round to create a 2D array
    deck = np.tile(base_round, (rounds, 1))

    # Independently shuffle each row
    rng = rng or np.random.default_rng()
    return rng.permuted(deck, axis=1).astype(np.int32)


class ConstantAssigner(Assigner[T_co]):
    """A simple assigner that always assigns the same constant group."""

    def __init__(self, group: T_co) -> None:
        self._group: T_co = group

    @override
    def assign(self, *values: int | str) -> T_co:
        """Assign the constant group, ignoring the input values."""
        return self._group


ConstantSplitAssigner = ConstantAssigner[DatasetSplit]
