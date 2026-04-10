import time
from collections.abc import Callable


def retry_with_backoff[T](
    fn: Callable[[], T],
    retryable_exceptions: tuple[type[Exception], ...],
    max_retries: int = 3,
    base_delay: int = 5,
    should_retry: Callable[[Exception], bool] | None = None,
    on_retry: Callable[[Exception, int, float], None] | None = None,
) -> T:
    """Call fn, retrying on retryable_exceptions with exponential backoff.

    Args:
        fn: Zero-argument callable to invoke.
        retryable_exceptions: Exception types that are candidates for retry.
        max_retries: Maximum number of attempts (including the first).
        base_delay: Delay in seconds before the first retry; doubles each attempt.
        should_retry: Optional predicate; if provided, an exception is only
            retried when this returns True. Non-retryable exceptions re-raise
            immediately regardless of remaining attempts.
        on_retry: Optional callback invoked before each sleep, receiving
            (exception, attempt_number, sleep_seconds).

    Returns:
        The return value of fn on success.

    Raises:
        The last exception raised by fn after all retries are exhausted,
        or immediately when should_retry returns False.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except retryable_exceptions as exc:
            if should_retry is not None and not should_retry(exc):
                raise
            if attempt >= max_retries - 1:
                raise
            sleep_time = base_delay * (2**attempt)
            if on_retry:
                on_retry(exc, attempt + 1, sleep_time)
            time.sleep(sleep_time)
    raise RuntimeError("retry_with_backoff: unreachable")  # pragma: no cover
