import threading
import time
from collections.abc import Callable

from you_talk_too_much.common.task_runner import TaskRunner


def _waiter(gate: threading.Event) -> Callable[[], None]:
    """Return a no-result job that blocks until the gate is set."""

    def wait() -> None:
        gate.wait(timeout=5)

    return wait


class TestTaskRunner:
    def test_runs_jobs_in_submission_order(self) -> None:
        runner = TaskRunner("test")
        results: list[int] = []

        for i in range(10):
            runner.submit(lambda i=i: results.append(i))

        runner.shutdown(timeout=5)
        assert results == list(range(10))

    def test_survives_job_exception_and_continues(self) -> None:
        runner = TaskRunner("test")
        results: list[str] = []

        def failing_job() -> None:
            raise RuntimeError("boom")

        runner.submit(failing_job)
        runner.submit(lambda: results.append("after-failure"))

        runner.shutdown(timeout=5)
        assert results == ["after-failure"]

    def test_shutdown_drains_pending_jobs(self) -> None:
        runner = TaskRunner("test")
        results: list[int] = []
        gate = threading.Event()

        runner.submit(_waiter(gate))
        for i in range(5):
            runner.submit(lambda i=i: results.append(i))

        gate.set()
        runner.shutdown(timeout=5)
        assert results == [0, 1, 2, 3, 4]

    def test_jobs_run_on_worker_thread_not_caller(self) -> None:
        runner = TaskRunner("test")
        thread_names: list[str] = []

        runner.submit(lambda: thread_names.append(threading.current_thread().name))

        runner.shutdown(timeout=5)
        assert thread_names == ["test-runner"]

    def test_pending_count_reflects_queued_jobs(self) -> None:
        runner = TaskRunner("test")
        gate = threading.Event()

        runner.submit(_waiter(gate))
        runner.submit(lambda: None)
        runner.submit(lambda: None)

        # First job is blocked on the gate; all three are unfinished
        assert runner.pending_count() == 3

        gate.set()
        runner.shutdown(timeout=5)
        assert runner.pending_count() == 0

    def test_submit_from_multiple_threads_executes_all_jobs(self) -> None:
        runner = TaskRunner("test")
        results: list[int] = []
        lock = threading.Lock()

        def add(i: int) -> None:
            with lock:
                results.append(i)

        threads = [
            threading.Thread(target=lambda i=i: runner.submit(lambda: add(i)))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        runner.shutdown(timeout=5)
        assert sorted(results) == list(range(20))

    def test_join_blocks_until_submitted_jobs_finish(self) -> None:
        runner = TaskRunner("test")
        results: list[str] = []

        runner.submit(lambda: time.sleep(0.05))
        runner.submit(lambda: results.append("done"))

        runner.join()
        assert results == ["done"]
        runner.shutdown(timeout=5)

    def test_slow_job_does_not_block_submission(self) -> None:
        runner = TaskRunner("test")
        gate = threading.Event()
        runner.submit(_waiter(gate))

        start = time.monotonic()
        runner.submit(lambda: None)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1  # submit is non-blocking
        gate.set()
        runner.shutdown(timeout=5)
