"""Runtime execution accounting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from dronalize.runtime.state import (
    CleanupProgress,
    ExecutionStats,
    Progress,
    ProgressState,
    ScreeningProgress,
    SplitCounts,
    empty_split_counts,
    split_key,
)
from dronalize.runtime.types import CleanupRemovalSummary, CleanupSummary

if TYPE_CHECKING:
    import threading
    from collections.abc import Iterator

    from dronalize.core.categories import DatasetSplit
    from dronalize.core.scene import Scene
    from dronalize.processing.loading.models import DatasetSource
    from dronalize.processing.screening.screen import CleanupSceneStats
    from dronalize.runtime.processor import RuntimeProcessor


@dataclass(slots=True)
class CleanupRemovalAccumulator:
    """Accumulate removal statistics for one cleanup-summary bucket."""

    scene_count: int = 0
    total_rows_removed: int = 0
    total_agents_removed: int = 0
    min_rows_removed: int | None = None
    max_rows_removed: int | None = None
    min_agents_removed: int | None = None
    max_agents_removed: int | None = None

    def record(self, *, rows_removed: int, agents_removed: int) -> None:
        """Record cleanup removals for one candidate scene."""
        self.scene_count += 1
        self.total_rows_removed += rows_removed
        self.total_agents_removed += agents_removed
        self.min_rows_removed = _min_optional(self.min_rows_removed, rows_removed)
        self.max_rows_removed = _max_optional(self.max_rows_removed, rows_removed)
        self.min_agents_removed = _min_optional(self.min_agents_removed, agents_removed)
        self.max_agents_removed = _max_optional(self.max_agents_removed, agents_removed)

    def merge(self, summary: CleanupRemovalSummary) -> None:
        """Merge a frozen cleanup-removal summary into this accumulator."""
        if summary.scene_count <= 0:
            return

        self.scene_count += summary.scene_count
        self.total_rows_removed += summary.total_rows_removed
        self.total_agents_removed += summary.total_agents_removed
        self.min_rows_removed = _min_optional(
            self.min_rows_removed, summary.min_rows_removed_per_scene
        )
        self.max_rows_removed = _max_optional(
            self.max_rows_removed, summary.max_rows_removed_per_scene
        )
        self.min_agents_removed = _min_optional(
            self.min_agents_removed, summary.min_agents_removed_per_scene
        )
        self.max_agents_removed = _max_optional(
            self.max_agents_removed, summary.max_agents_removed_per_scene
        )

    def freeze(self) -> CleanupRemovalSummary | None:
        """Return an immutable summary, or ``None`` if no observations exist."""
        if self.scene_count == 0:
            return None

        return CleanupRemovalSummary(
            scene_count=self.scene_count,
            total_rows_removed=self.total_rows_removed,
            average_rows_removed_per_scene=self.total_rows_removed / self.scene_count,
            min_rows_removed_per_scene=self.min_rows_removed or 0,
            max_rows_removed_per_scene=self.max_rows_removed or 0,
            total_agents_removed=self.total_agents_removed,
            average_agents_removed_per_scene=self.total_agents_removed / self.scene_count,
            min_agents_removed_per_scene=self.min_agents_removed or 0,
            max_agents_removed_per_scene=self.max_agents_removed or 0,
        )


@dataclass(slots=True)
class CleanupSummaryAccumulator:
    """Accumulate cleanup summary statistics, including per-rule summaries."""

    overall: CleanupRemovalAccumulator = field(default_factory=CleanupRemovalAccumulator)
    by_rule: dict[str, CleanupRemovalAccumulator] = field(default_factory=dict)

    def record(self, stats: CleanupSceneStats | None) -> None:
        """Record cleanup statistics for one candidate scene."""
        if stats is None:
            return

        self.overall.record(rows_removed=stats.rows_removed, agents_removed=stats.agents_removed)
        for rule_stats in stats.by_rule:
            accumulator = self.by_rule.setdefault(rule_stats.rule_name, CleanupRemovalAccumulator())
            accumulator.record(
                rows_removed=rule_stats.rows_removed, agents_removed=rule_stats.agents_removed
            )

    def merge(self, summary: CleanupSummary | None) -> None:
        """Merge a frozen cleanup summary into this accumulator."""
        if summary is None:
            return

        self.overall.merge(summary.overall)
        for rule_name, rule_summary in summary.by_rule.items():
            accumulator = self.by_rule.setdefault(rule_name, CleanupRemovalAccumulator())
            accumulator.merge(rule_summary)

    def freeze(self) -> CleanupSummary | None:
        """Return an immutable cleanup summary, or ``None`` if no stats exist."""
        overall = self.overall.freeze()
        if overall is None:
            return None

        by_rule = {
            rule_name: summary
            for rule_name, accumulator in self.by_rule.items()
            if (summary := accumulator.freeze()) is not None
        }
        return CleanupSummary(overall=overall, by_rule=by_rule)


@dataclass(slots=True)
class CleanupAccounting:
    """Collect cleanup progress counters and detailed cleanup summaries."""

    accumulator: CleanupSummaryAccumulator = field(default_factory=CleanupSummaryAccumulator)
    rows_total: int = 0
    rows_removed: int = 0
    agents_total: int = 0
    agents_removed: int = 0

    def record(self, stats: CleanupSceneStats | None) -> bool:
        """Record cleanup stats and return whether any counters changed."""
        if stats is None:
            return False

        self.accumulator.record(stats)
        self.rows_total += stats.rows_before
        self.rows_removed += stats.rows_removed
        self.agents_total += stats.agents_before
        self.agents_removed += stats.agents_removed
        return True

    def progress(self) -> CleanupProgress:
        """Return lightweight cleanup counters for live progress."""
        return CleanupProgress(
            rows_total=self.rows_total,
            rows_removed=self.rows_removed,
            agents_total=self.agents_total,
            agents_removed=self.agents_removed,
        )

    def freeze(self) -> CleanupSummary | None:
        """Return the frozen cleanup summary."""
        return self.accumulator.freeze()


class RunAccounting(Protocol):
    """Bookkeeping backend used by the shared source-to-scene loop."""

    def finish_source(self) -> None:
        """Record that processing finished for one source."""
        ...

    def record_candidate(self) -> None:
        """Record one generated scene candidate."""
        ...

    def record_cleanup(self, stats: CleanupSceneStats | None) -> None:
        """Record cleanup statistics associated with one candidate scene."""
        ...

    def record_screening_result(self, *, passed: bool) -> None:
        """Record whether one candidate passed screening."""
        ...

    def claim_scene_number(self) -> int | None:
        """Claim the next output scene number, respecting the scene limit."""
        ...

    def record_written(self, split: DatasetSplit | None) -> None:
        """Record that one selected scene was written/materialized."""
        ...

    def limit_reached(self) -> bool:
        """Return whether the output scene limit has already been reached."""
        ...

    def cleanup_summary(self) -> CleanupSummary | None:
        """Return cleanup summary statistics collected by this backend."""
        ...


@dataclass(slots=True)
class LocalRunAccounting:
    """Single-process runtime accounting backend."""

    limit: int | None = None
    update_event: threading.Event | None = None
    source_count: int = 0
    candidate_count: int = 0
    written_count: int = 0
    screening_passed_count: int = 0
    screening_rejected_count: int = 0
    split_counts: SplitCounts = field(default_factory=empty_split_counts)
    cleanup: CleanupAccounting = field(default_factory=CleanupAccounting)

    def finish_source(self) -> None:
        """Record that one source finished processing."""
        self.source_count += 1
        self._changed()

    def record_candidate(self) -> None:
        """Record one generated candidate scene."""
        self.candidate_count += 1
        self._changed()

    def record_cleanup(self, stats: CleanupSceneStats | None) -> None:
        """Record cleanup statistics for one candidate scene."""
        if self.cleanup.record(stats):
            self._changed()

    def record_screening_result(self, *, passed: bool) -> None:
        """Record one screening decision."""
        if passed:
            self.screening_passed_count += 1
        else:
            self.screening_rejected_count += 1
        self._changed()

    def claim_scene_number(self) -> int | None:
        """Claim the next scene number, or return ``None`` at the limit."""
        if self.limit_reached():
            return None

        scene_number = self.written_count
        self.written_count += 1
        self._changed()
        return scene_number

    def record_written(self, split: DatasetSplit | None) -> None:
        """Record the split assignment for one written scene."""
        self.split_counts[split_key(split)] += 1
        self._changed()

    def limit_reached(self) -> bool:
        """Return whether the output scene limit has been reached."""
        return self.limit is not None and self.written_count >= self.limit

    def cleanup_summary(self) -> CleanupSummary | None:
        """Return aggregated cleanup statistics."""
        return self.cleanup.freeze()

    def stats(self, *, screening_enabled: bool) -> ExecutionStats:
        """Return immutable execution counters for this accounting backend."""
        return ExecutionStats(
            processed_sources=self.source_count,
            candidate_scenes=self.candidate_count,
            written_scenes=self.written_count,
            split_counts={
                "test": self.split_counts["test"],
                "train": self.split_counts["train"],
                "unsplit": self.split_counts["unsplit"],
                "val": self.split_counts["val"],
            },
            cleanup=self.cleanup.progress(),
            screening=ScreeningProgress(
                enabled=screening_enabled,
                passed=self.screening_passed_count,
                rejected=self.screening_rejected_count,
            ),
        )

    def snapshot(
        self,
        *,
        running: bool,
        total_sources: int | None,
        active_workers: int,
        screening_enabled: bool,
    ) -> Progress:
        """Build a public immutable progress snapshot."""
        return Progress(
            running=running,
            total_sources=total_sources,
            scene_limit=self.limit,
            active_workers=active_workers,
            stats=self.stats(screening_enabled=screening_enabled),
        )

    def _changed(self) -> None:
        if self.update_event is not None:
            self.update_event.set()


@dataclass(slots=True)
class SharedRunAccounting:
    """Multiprocessing accounting backed by ``ProgressState``."""

    progress: ProgressState
    limit: int | None = None
    cleanup: CleanupAccounting = field(default_factory=CleanupAccounting)

    def finish_source(self) -> None:
        """Record that one source finished processing."""
        self.progress.record_processed_source()

    def record_candidate(self) -> None:
        """Record one generated candidate scene."""
        self.progress.record_candidate_scene()

    def record_cleanup(self, stats: CleanupSceneStats | None) -> None:
        """Record cleanup statistics for progress and worker-local summary."""
        if not self.cleanup.record(stats) or stats is None:
            return
        self.progress.record_cleanup(
            rows_total=stats.rows_before,
            rows_removed=stats.rows_removed,
            agents_total=stats.agents_before,
            agents_removed=stats.agents_removed,
        )

    def record_screening_result(self, *, passed: bool) -> None:
        """Record one screening decision in shared progress state."""
        self.progress.record_screening_result(passed=passed)

    def claim_scene_number(self) -> int | None:
        """Claim the next scene number from shared progress state."""
        return self.progress.claim_written_scene(self.limit)

    def record_written(self, split: DatasetSplit | None) -> None:
        """Record the split assignment for one written scene."""
        self.progress.record_split(split)

    def limit_reached(self) -> bool:
        """Return whether the shared output scene limit has been reached."""
        return self.progress.written_scene_limit_reached(self.limit)

    def cleanup_summary(self) -> CleanupSummary | None:
        """Return worker-local cleanup statistics."""
        return self.cleanup.freeze()


def iter_scenes_from_source(
    processor: RuntimeProcessor, source: DatasetSource[Any], accounting: RunAccounting
) -> Iterator[Scene]:
    """Yield materialized scenes for one source while applying bookkeeping.

    Cleanup is recorded for candidates considered before the run stops. Scene
    numbers are claimed only after screening passes. A source is counted as
    processed only after all of its candidates have been extracted.
    """
    for candidate in processor.iter_candidates(source):
        if accounting.limit_reached():
            return

        accounting.record_candidate()
        accounting.record_cleanup(candidate.cleanup_stats)
        accounting.record_screening_result(passed=candidate.passes_screening)
        if not candidate.passes_screening:
            continue

        scene_number = accounting.claim_scene_number()
        if scene_number is None:
            return

        scene = processor.materialize(candidate, scene_number)
        accounting.record_written(scene.split_assignment)
        yield scene
    accounting.finish_source()


def _min_optional(current: int | None, value: int) -> int:
    return value if current is None else min(current, value)


def _max_optional(current: int | None, value: int) -> int:
    return value if current is None else max(current, value)
