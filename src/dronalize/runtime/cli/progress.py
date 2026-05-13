"""Rich-backed progress helpers for the optional CLI."""

from __future__ import annotations

import random
import threading
import time
from typing import TYPE_CHECKING, TypeVar

import rich.progress as rp
from rich import box
from rich.console import Group, RenderableType, RichCast
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from typing_extensions import override

from dronalize.runtime.state import SplitCounts

if TYPE_CHECKING:
    from collections.abc import Callable

    from dronalize.runtime.executor import Executor
    from dronalize.runtime.state import Progress

T = TypeVar("T")


class _ExecutorDisplay(RichCast):
    """A custom renderable that stacks a progress bar and a stats grid vertically."""

    def __init__(self) -> None:
        num = random.randint(1, 12)  # noqa: S311
        spinner_name = f"dots{num}" if num > 1 else "dots"

        self.progress: rp.Progress = rp.Progress(
            rp.SpinnerColumn(spinner_name=spinner_name, style="bold cyan"),
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
        self.workers: int = 0
        self.candidate_scenes: int = 0
        self.selected_scenes: int = 0
        self.split_counts: SplitCounts = SplitCounts(unsplit=0, train=0, val=0, test=0)
        self.screening_enabled: bool = False

    def update(self, progress_state: Progress) -> None:
        if progress_state.scene_limit is not None:
            total = progress_state.scene_limit
            completed = progress_state.selected_scenes
        else:
            total = progress_state.total_sources
            completed = progress_state.processed_sources

        self.workers = progress_state.active_workers if progress_state.running else 0
        self.candidate_scenes = progress_state.candidate_scenes
        self.selected_scenes = progress_state.selected_scenes
        self.split_counts = progress_state.split_counts
        self.screening_enabled = progress_state.screening_enabled

        if not progress_state.running and (total is None or completed < total):
            total = completed

        self.progress.update(self.task_id, completed=completed, total=total)

    @override
    def __rich__(self) -> Panel:
        def fmt_split(name: str, count: int, total: int) -> str:
            percent = (count / total) * 100 if total > 0 else 0
            return f"{name}: {count} ({percent:.1f}%)"

        layout_elements: list[RenderableType] = [self.progress, ""]
        stats_markup = (
            f"[bold cyan]Workers:[/bold cyan] {self.workers}"
            f"    [bold magenta]Scenes:[/bold magenta] {self.selected_scenes}"
        )
        stats_text = Text.from_markup(stats_markup)
        stats_text.justify = "center"
        layout_elements.append(stats_text)

        if self.screening_enabled:
            screening_percent = (
                (self.selected_scenes / self.candidate_scenes) * 100
                if self.candidate_scenes > 0
                else 0
            )
            screening_markup = (
                "[bold yellow]Screening:[/bold yellow] "
                f"{self.selected_scenes} / {self.candidate_scenes} ({screening_percent:.1f}%)"
            )
            screening_text = Text.from_markup(screening_markup)
            screening_text.justify = "center"
            layout_elements.append(screening_text)

        nonzero_splits = {
            k: v for k, v in self.split_counts.items() if isinstance(v, int) and v > 0
        }
        if nonzero_splits:
            total = sum(nonzero_splits.values())
            splits_str = " • ".join(
                fmt_split(name, count, total) for name, count in nonzero_splits.items()
            )

            splits_markup = f"[bold green]Splits:[/bold green] {splits_str}"
            splits_text = Text.from_markup(splits_markup)
            splits_text.justify = "center"

            layout_elements.append(splits_text)

        return Panel(Group(*layout_elements), title_align="left", box=box.MINIMAL, expand=False)


class _ProgressMonitor:
    def __init__(self, executor: Executor, display: _ExecutorDisplay) -> None:
        self._executor: Executor = executor
        self._display: _ExecutorDisplay = display
        self._stop_event: threading.Event = threading.Event()
        self._error: BaseException | None = None

    def thread(self) -> threading.Thread:
        return threading.Thread(target=self._work, daemon=True)

    def stop(self) -> None:
        self._stop_event.set()
        self._executor.progress_event().set()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            msg = "Rich progress monitor failed."
            raise RuntimeError(msg) from self._error

    def _wait_for_start(self, timeout: float | None) -> bool:
        if not self._executor.progress_event().wait(timeout):
            return False
        self._executor.progress_event().clear()
        self._display.update(self._executor.progress())
        return True

    def _work(self, timeout: float | None = 20, sleep: float | None = 0.5) -> None:
        if not self._wait_for_start(timeout):
            msg = "Timed out waiting for executor to start"
            self._error = TimeoutError(msg)
            return

        while not self._stop_event.is_set():
            if sleep is not None:
                time.sleep(sleep)
            _ = self._executor.progress_event().wait()
            self._executor.progress_event().clear()

            progress = self._executor.progress()
            self._display.update(progress)

            if not progress.running:
                return


def run_with_rich_progress(executor: Executor, run: Callable[[], T], *, enable: bool = True) -> T:
    """Run an executor callback while rendering a Rich progress bar."""
    if not enable:
        return run()

    display = _ExecutorDisplay()
    monitor = _ProgressMonitor(executor, display)
    thread = monitor.thread()

    result: T
    with Live(display, refresh_per_second=4, transient=False):
        thread.start()
        try:
            result = run()
        finally:
            monitor.stop()
            thread.join()

    monitor.raise_if_failed()
    return result
