import queue
import threading
from collections.abc import Callable

from you_talk_too_much.cli.logger import setup_logger

logger = setup_logger(__name__)


class TaskRunner:
    """Executes submitted jobs one at a time, in order, on a dedicated thread.

    A job that raises is logged with its traceback and the runner keeps
    processing subsequent jobs - the worker thread never dies.
    """

    def __init__(self, name: str) -> None:
        """Start the worker thread."""
        self.name = name
        # None is the shutdown sentinel
        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name=f"{name}-runner", daemon=True
        )
        self._thread.start()

    def submit(self, job: Callable[[], None]) -> None:
        """Enqueue a job to run after all previously submitted jobs."""
        self._queue.put(job)

    def pending_count(self) -> int:
        """Return the number of jobs submitted but not yet finished."""
        return self._queue.unfinished_tasks

    def join(self) -> None:
        """Block until every job submitted so far has finished."""
        self._queue.join()

    def shutdown(self, timeout: float | None = None) -> None:
        """Finish all pending jobs, then stop the worker thread."""
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(f"TaskRunner '{self.name}' did not finish within timeout.")

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                job()
            except Exception:
                logger.exception(f"Job failed in TaskRunner '{self.name}'.")
            finally:
                self._queue.task_done()
