from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class StreamSplitter(Generic[T]):
    """A class for splitting a stream of data into groups.

    The StreamSplitter takes a sequence of unique groups and optional weights,
    and provides an infinite stream of group assignments. The groups are
    shuffled in rounds, where each round consists of a fixed number of samples
    (`round_size`) before reshuffling. The number of pre-generated rounds can be
    specified with `rounds` parameter.

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
        groups: Sequence[T],
        weights: Sequence[float] | None = None,
        round_size: int = 100,
        rounds: int = 10,
        seed: int | None = None,
    ) -> None:
        """Initialize the StreamSplitter.

        Parameters
        ----------
        groups : Sequence[T]
            A sequence of unique groups to split the stream into.
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
        self._groups = list(dict.fromkeys(groups))
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

        self._index_to_group = list(self._groups)

        self._weights = np.array(weights) if weights is not None else np.ones(len(self._groups))

        if np.any(self._weights < 0):
            msg = "All weights must be non-negative."
            raise ValueError(msg)

        weights_sum = self._weights.sum()

        if weights_sum == 0:
            msg = f"At least one weight must be greater than zero got {self._weights}."
            raise ValueError(msg)

        self._weights /= self._weights.sum()

        self._rng = np.random.default_rng(seed)
        self._deck = _generate_shuffled_deck(self._weights, round_size, rounds, self._rng)
        self._index = 0
        self._round_index = 0

    def next(self) -> T:
        """Next group in the stream.

        Returns
        -------
            The next randomly selected group based on the current round's
            shuffled deck.

        Examples
        --------
        >>> splitter = StreamSplitter([1, 2], seed=42, round_size=2)
        >>> splitter.next() # First is 2 by chance
        2
        >>> splitter.next() # Second is guarenteed 1, since `round_size` is 2
        1
        """
        index = self._index
        value = self._deck[self._round_index, index]
        self._update()
        return self._index_to_group[value]

    def take(self, n: int) -> Iterator[T]:
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

    def __iter__(self) -> Iterator[T]:
        """Return an infinite iterator that yields groups from the stream.

        Returns
        -------
        Iterator[T]
            An infinite iterator that yields groups based on the current round's
            shuffled deck.

        """

        def infinite_generator() -> Iterator[T]:
            while True:
                yield self.next()

        return infinite_generator()

    def _update(self) -> None:
        self._index += 1
        if self._index >= self._deck.shape[1]:
            self._index = 0
            # Random row selection for the next round
            self._round_index = self._rng.integers(0, self._deck.shape[0])


def _generate_shuffled_deck(
    weights: np.ndarray,
    round_size: int,
    rounds: int,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.int32]:
    # Calculate exact integer counts for one round
    float_counts = weights * round_size
    counts = np.floor(float_counts).astype(np.int32)

    # Handle rounding remainders to strictly enforce round_size
    remainder = round_size - counts.sum()
    if remainder > 0:
        # Distribute remainder to the groups with the largest fractional parts
        fractional_parts = float_counts - counts
        top_indices = np.argsort(fractional_parts)[-remainder:]
        counts[top_indices] += 1

    # Create the base 1D array for a single round
    n_groups = len(weights)
    base_round = np.repeat(np.arange(n_groups), counts)

    # Duplicate the base round to create a 2D array
    deck = np.tile(base_round, (rounds, 1))

    # Independently shuffle each row
    rng = rng or np.random.default_rng()
    return rng.permuted(deck, axis=1)
