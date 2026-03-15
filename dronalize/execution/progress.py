import random
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import rich.progress as rp
from typing_extensions import override

from dronalize.categories import DatasetSplit
from dronalize.execution.assigner import (
    SplitAssigner,
    StatelessWeightedAssigner,
)
from dronalize.execution.common import Progress
from dronalize.execution.executor import ObservableWritingExecutor, WriterFactory, WritingExecutor
from dronalize.loading.writer.writer import SceneWriter

if TYPE_CHECKING:
    from rich.progress import Progress as RichProgress


class ProgressReportingExecutor(WritingExecutor):
    """Decorator that renders live progress while delegating execution."""

    def __init__(self, executor: ObservableWritingExecutor, *, enable: bool = True) -> None:
        """Initialize the progress reporting executor.

        Parameters
        ----------
        executor : ObservableWritingExecutor
            The executor to wrap and observe for progress updates.
        enable : bool, optional
            Whether to enable progress reporting. When `False`, this class
            simply delegates to the wrapped executor without rendering progress.
            There will be no overhead when using this class with `enable=False`,
            so it can be used as a drop-in wrapper when progress reporting is
            only conditionally desired.

        """
        self._inner: ObservableWritingExecutor = executor
        self._enable: bool = enable
        self._rich_progress: RichProgress = self._get_rich_progress()
        self._task_id: rp.TaskID = self._rich_progress.add_task(
            "Processing", total=0, scenes=0, workers=0
        )

    @override
    def execute(
        self,
        writer_factory: WriterFactory,
        finalize: Callable[[SceneWriter], None] | None = None,
        split_assigner: SplitAssigner | None = None,
    ) -> None:
        if not self._enable:
            self._inner.execute(writer_factory, finalize, split_assigner)
            return

        thread = self._progress_thread(self._progress_fn)
        with self._rich_progress:
            thread.start()
            self._inner.execute(writer_factory, finalize, split_assigner)
            thread.join()

    def _progress_thread(
        self,
        fn: Callable[[Progress], None],
        sleep_time: float | None = None,
        timeout: float | None = 20,
    ) -> threading.Thread:
        """Create the background thread that consumes progress notifications."""

        def _work() -> None:
            executor = self._inner
            if not executor.progress_event().wait(timeout=timeout):
                msg = "Progress thread timed out waiting for executor to start."
                raise TimeoutError(msg)

            while executor.is_running():
                _ = executor.progress_event().wait()
                executor.progress_event().clear()
                progress = executor.progress()
                fn(progress)
                if sleep_time is not None:
                    time.sleep(sleep_time)

            # Make sure the callback gets a chance to observe the final state
            # after processing completes.
            _ = fn(executor.progress())

        return threading.Thread(target=_work, daemon=True)

    def _progress_fn(self, progress: Progress) -> None:
        """Project a `Progress` snapshot onto the Rich task state."""
        if progress.running:
            self._rich_progress.update(
                self._task_id,
                completed=progress.processed_sources,
                total=progress.total_sources,
                scenes=progress.processed_scenes,
                workers=progress.active_workers,
            )
        else:
            self._rich_progress.update(
                self._task_id,
                completed=progress.processed_sources,
                total=progress.processed_sources,
                scenes=progress.processed_scenes,
                workers=0,
            )

    @staticmethod
    def _get_rich_progress() -> rp.Progress:
        """Build the Rich progress instance used for live rendering."""
        num: int = random.randint(1, 12)
        spinner_name: str = f"dots{num}" if num > 1 else "dots"
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
    from dronalize.datasets.eth_ucy import HotelLoader
    from dronalize.execution.parallel import ParallelExecutor
    from dronalize.loading.writer._dummy import DummyWriter

    split_assigner = StatelessWeightedAssigner(
        groups=[DatasetSplit.TRAIN, DatasetSplit.VAL], weights=[0.8, 0.2]
    )
    loader = HotelLoader("data", splits=None)
    progress = ProgressReportingExecutor(ParallelExecutor(loader, limit=None, processes=2))
    progress.execute(DummyWriter.as_factory(log=True), split_assigner=split_assigner)
