"""Rich-backed progress helpers for the optional CLI."""

from __future__ import annotations

import logging
import random
import threading
import time
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING, Protocol, TypeVar

import rich.progress as rp
from rich import box
from rich.console import Group, RenderableType, RichCast
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from typing_extensions import override

from dronalize.runtime.state import Progress, SplitCounts

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
AnyEvent = Event | threading.Event

logger = logging.getLogger(__name__)


class ProgressSource(Protocol):
    """Minimal progress interface consumed by the optional display."""

    def snapshot(self) -> Progress:
        """Return a point-in-time progress snapshot."""
        ...

    def changed(self) -> AnyEvent:
        """Return the event signaled after progress changes."""
        ...


class _ExecutorDisplay(RichCast):
    """Custom renderable that stacks a progress bar and runtime statistics."""

    def __init__(self) -> None:
        self.progress: rp.Progress = rp.Progress(
            rp.SpinnerColumn(spinner_name=_random_spinner_name(), style="bold cyan"),
            rp.TextColumn("[bold white]Processing"),
            rp.BarColumn(bar_width=40, style="dim", complete_style="green"),
            rp.TaskProgressColumn(),
            rp.MofNCompleteColumn(),
            rp.TextColumn("•"),
            rp.TimeElapsedColumn(),
            rp.TextColumn("•"),
            rp.TimeRemainingColumn(),
            expand=False,
        )
        self.task_id: rp.TaskID = self.progress.add_task("run", total=0)
        self.state: Progress = Progress.empty()

    def update(self, progress_state: Progress) -> None:
        completed, total = _progress_bar_counts(progress_state)
        self.state = progress_state
        self.progress.update(self.task_id, completed=completed, total=total)

    @override
    def __rich__(self) -> Panel:
        layout: list[RenderableType] = [self.progress, ""]
        layout.append(_centered_markup(self._main_stats_markup()))

        if self.state.screening.enabled:
            layout.append(_centered_markup(self._screening_markup()))

        if (cleanup_markup := self._cleanup_markup()) is not None:
            layout.append(_centered_markup(cleanup_markup))

        if (splits_markup := self._splits_markup()) is not None:
            layout.append(_centered_markup(splits_markup))

        return Panel(Group(*layout), title_align="left", box=box.MINIMAL, expand=False)

    def _main_stats_markup(self) -> str:
        source_part = _source_progress_text(
            processed=self.state.stats.processed_sources, total=self.state.total_sources
        )
        return (
            f"[bold cyan]Workers:[/bold cyan] {self._visible_workers}"
            f"    [bold blue]Sources:[/bold blue] {source_part}"
            f"    [bold magenta]Scenes written:[/bold magenta] {self.state.stats.written_scenes}"
        )

    def _screening_markup(self) -> str:
        candidates = self.state.stats.candidate_scenes
        screening = self.state.screening
        return (
            "[bold yellow]Screening:[/bold yellow] "
            f"{screening.passed} / {candidates} passed "
            f"({_percent(screening.passed, candidates):.1f}%)"
            f" • rejected: {screening.rejected}"
        )

    def _cleanup_markup(self) -> str | None:
        cleanup = self.state.cleanup
        if cleanup.rows_removed <= 0 and cleanup.agents_removed <= 0:
            return None

        return (
            "[bold red]Cleanup:[/bold red] "
            f"{cleanup.rows_removed} / {cleanup.rows_total} rows removed"
            f" ({_percent(cleanup.rows_removed, cleanup.rows_total):.1f}%) • "
            f"{cleanup.agents_removed} / {cleanup.agents_total} agents removed"
            f" ({_percent(cleanup.agents_removed, cleanup.agents_total):.1f}%)"
        )

    def _splits_markup(self) -> str | None:
        nonzero_splits = _nonzero_splits(self.state.split_counts)
        if not nonzero_splits:
            return None

        total = sum(nonzero_splits.values())
        splits = " • ".join(
            _format_split(name, count, total) for name, count in nonzero_splits.items()
        )
        return f"[bold green]Splits:[/bold green] {splits}"

    @property
    def _visible_workers(self) -> int:
        return self.state.active_workers if self.state.running else 0


class _ProgressMonitor:
    """Background monitor that pushes executor snapshots into a Rich display."""

    def __init__(self, progress: ProgressSource, display: _ExecutorDisplay) -> None:
        self._progress: ProgressSource = progress
        self._display: _ExecutorDisplay = display
        self._stop_event: threading.Event = threading.Event()
        self._error: BaseException | None = None

    def thread(self) -> threading.Thread:
        return threading.Thread(target=self._work, daemon=True)

    def stop(self) -> None:
        self._stop_event.set()
        self._progress.changed().set()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            msg = "Rich progress monitor failed."
            raise RuntimeError(msg) from self._error

    def _wait_for_start(self, timeout: float | None) -> bool:
        if not self._progress.changed().wait(timeout):
            return False
        self._progress.changed().clear()
        self._display.update(self._progress.snapshot())
        return True

    def _work(self, timeout: float | None = 20, sleep: float | None = 0.5) -> None:
        if not self._wait_for_start(timeout):
            msg = "Timed out waiting for executor to start."
            self._error = TimeoutError(msg)
            return

        while not self._stop_event.is_set():
            if sleep is not None:
                time.sleep(sleep)
            event = self._progress.changed()
            _ = event.wait()
            event.clear()
            progress = self._progress.snapshot()
            self._display.update(progress)

            if not progress.running:
                return


def execute_with_rich_progress(
    progress: ProgressSource, run: Callable[[], T], *, enable: bool = True
) -> T:
    """Run an executor callback while rendering a Rich progress display."""
    if not enable:
        return run()

    display = _ExecutorDisplay()
    monitor = _ProgressMonitor(progress, display)
    thread = monitor.thread()

    with Live(display, refresh_per_second=4, transient=False):
        thread.start()
        try:
            result = run()
        finally:
            monitor.stop()
            thread.join()

    monitor.raise_if_failed()
    return result


def _progress_bar_counts(progress: Progress) -> tuple[int, int | None]:
    """Return completed/total values for the main progress bar."""
    if progress.scene_limit is not None:
        completed = progress.stats.written_scenes
        total: int | None = progress.scene_limit
    else:
        completed = progress.stats.processed_sources
        total = progress.total_sources

    if not progress.running and (total is None or completed < total):
        total = completed

    return completed, total


def _centered_markup(markup: str) -> Text:
    text = Text.from_markup(markup)
    text.justify = "center"
    return text


def _format_split(name: str, count: int, total: int) -> str:
    return f"{name}: {count} ({_percent(count, total):.1f}%)"


def _nonzero_splits(split_counts: SplitCounts) -> dict[str, int]:
    return {
        name: count for name, count in split_counts.items() if isinstance(count, int) and count > 0
    }


def _source_progress_text(*, processed: int, total: int | None) -> str:
    if total is None:
        return str(processed)
    return f"{processed} / {total}"


def _percent(part: int, total: int) -> float:
    return (part / total) * 100 if total > 0 else 0.0


def _random_spinner_name() -> str:
    number = random.randint(1, 12)  # noqa: S311
    return f"dots{number}" if number > 1 else "dots"
