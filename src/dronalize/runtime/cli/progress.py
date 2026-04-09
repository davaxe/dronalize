"""Rich-backed progress helpers for the optional CLI."""

from __future__ import annotations

import random
import threading
import time
from typing import TYPE_CHECKING

import rich.progress as rp

from dronalize.runtime import plan_dataset

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.progress import Progress as RichProgress

    from dronalize.runtime.executor import ObservableExecutor, Progress


def run_with_rich_progress(
    executor: ObservableExecutor, run: Callable[[], None], *, enable: bool = True
) -> None:
    """Execute a run while rendering Rich progress for an observable executor."""
    if not enable:
        run()
        return

    rich_progress = _get_rich_progress()
    task_id = rich_progress.add_task("Processing", total=0, scenes=0, workers=0)
    monitor = _ProgressMonitor(executor, rich_progress, task_id)
    thread = monitor.thread()

    with rich_progress:
        thread.start()
        try:
            run()
        finally:
            monitor.stop()
            thread.join()

    monitor.raise_if_failed()


class _ProgressMonitor:
    """Background monitor that projects executor progress onto Rich."""

    def __init__(
        self, executor: ObservableExecutor, rich_progress: RichProgress, task_id: rp.TaskID
    ) -> None:
        self._executor: ObservableExecutor = executor
        self._rich_progress: RichProgress = rich_progress
        self._task_id: rp.TaskID = task_id
        self._stop_event: threading.Event = threading.Event()
        self._error: BaseException | None = None

    def thread(self) -> threading.Thread:
        """Create the background thread that consumes progress notifications."""
        return threading.Thread(target=self._work, daemon=True)

    def stop(self) -> None:
        """Request that the monitor stops waiting for progress updates."""
        self._stop_event.set()
        self._executor.progress_event().set()

    def raise_if_failed(self) -> None:
        """Raise a monitor failure once the run completed cleanly."""
        if self._error is None:
            return

        msg = "Rich progress monitor failed."
        raise RuntimeError(msg) from self._error

    def _wait_for_start(self, timeout: float | None) -> bool:
        """Wait for the executor to emit its first lifecycle update."""
        if not self._executor.progress_event().wait(timeout):
            return False

        self._executor.progress_event().clear()

        self._render(self._executor.progress())
        return True

    def _work(self, timeout: float | None = 20, sleep: float | None = 0.5) -> None:
        if not self._wait_for_start(timeout):
            msg = "Timed out waiting for executor to start"
            raise TimeoutError(msg)

        while not self._stop_event.is_set():
            if sleep is not None:
                time.sleep(sleep)
            _ = self._executor.progress_event().wait()
            self._executor.progress_event().clear()
            progress = self._executor.progress()
            self._render(progress)
            if not progress.running:
                return

    def _render(self, progress: Progress) -> None:
        """Project a `Progress` snapshot onto the Rich task state."""
        if progress.total_scenes is not None:
            total = progress.total_scenes
            completed = progress.processed_scenes
        else:
            total = progress.total_sources
            completed = progress.processed_sources
        workers = progress.active_workers if progress.running else 0

        if not progress.running and (total is None or completed < total):
            total = completed

        self._rich_progress.update(
            self._task_id,
            completed=completed,
            total=total,
            scenes=progress.processed_scenes,
            workers=workers,
        )


def _get_rich_progress() -> rp.Progress:
    """Build the Rich progress instance used for live rendering."""
    num = random.randint(1, 12)
    spinner_name = f"dots{num}" if num > 1 else "dots"
    return rp.Progress(
        rp.SpinnerColumn(spinner_name=spinner_name),
        rp.TaskProgressColumn(),
        rp.BarColumn(),
        rp.MofNCompleteColumn(),
        rp.TextColumn("•"),
        rp.TimeElapsedColumn(),
        rp.TextColumn("•"),
        rp.TimeRemainingColumn(),
        rp.TextColumn("•"),
        rp.TextColumn("{task.fields[scenes]} scene(s)"),
        rp.TextColumn("•"),
        rp.TextColumn("{task.fields[workers]} worker(s)"),
        expand=False,
    )


if __name__ == "__main__":
    import os
    from pathlib import Path

    path = os.environ.get("TRAJ_DATA", None)
    if path is None:
        path = Path()

    plan = plan_dataset(
        dataset="round",
        input_dir=Path(path) / "rounD",
        output_dir=Path("test_output1"),
        storage_backend="mds",
        config_path=None,
        jobs=1,
        limit=500,
        seed=20,
    )
    with plan.open() as run:
        run_with_rich_progress(run.executor, run.run, enable=True)
