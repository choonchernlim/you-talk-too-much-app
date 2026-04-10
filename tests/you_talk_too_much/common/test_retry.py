from unittest.mock import MagicMock, patch

import pytest

from you_talk_too_much.common.retry import retry_with_backoff


class _RetryError(Exception):
    pass


class _OtherError(Exception):
    pass


def test_returns_result_on_first_success():
    fn = MagicMock(return_value="ok")
    result = retry_with_backoff(fn, retryable_exceptions=(_RetryError,))
    assert result == "ok"
    fn.assert_called_once()


def test_retries_and_succeeds_on_second_attempt():
    fn = MagicMock(side_effect=[_RetryError(), "ok"])
    with patch("you_talk_too_much.common.retry.time.sleep"):
        result = retry_with_backoff(
            fn, retryable_exceptions=(_RetryError,), max_retries=2
        )
    assert result == "ok"
    assert fn.call_count == 2


def test_raises_after_exhausting_retries():
    fn = MagicMock(side_effect=_RetryError("boom"))
    with (
        patch("you_talk_too_much.common.retry.time.sleep"),
        pytest.raises(_RetryError),
    ):
        retry_with_backoff(fn, retryable_exceptions=(_RetryError,), max_retries=3)
    assert fn.call_count == 3


def test_raises_immediately_when_should_retry_returns_false():
    fn = MagicMock(side_effect=_RetryError())
    with (
        patch("you_talk_too_much.common.retry.time.sleep"),
        pytest.raises(_RetryError),
    ):
        retry_with_backoff(
            fn,
            retryable_exceptions=(_RetryError,),
            max_retries=5,
            should_retry=lambda _: False,
        )
    fn.assert_called_once()


def test_retries_only_when_should_retry_returns_true():
    fn = MagicMock(side_effect=[_RetryError(), _RetryError(), "ok"])
    with patch("you_talk_too_much.common.retry.time.sleep"):
        result = retry_with_backoff(
            fn,
            retryable_exceptions=(_RetryError,),
            max_retries=5,
            should_retry=lambda _: True,
        )
    assert result == "ok"
    assert fn.call_count == 3


def test_sleeps_with_exponential_backoff():
    fn = MagicMock(side_effect=[_RetryError(), _RetryError(), "ok"])
    with patch("you_talk_too_much.common.retry.time.sleep") as mock_sleep:
        retry_with_backoff(
            fn, retryable_exceptions=(_RetryError,), max_retries=3, base_delay=10
        )
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(10)  # attempt 0: 10 * 2**0
    mock_sleep.assert_any_call(20)  # attempt 1: 10 * 2**1


def test_calls_on_retry_callback_with_correct_args():
    fn = MagicMock(side_effect=[_RetryError("first"), "ok"])
    on_retry = MagicMock()
    with patch("you_talk_too_much.common.retry.time.sleep"):
        retry_with_backoff(
            fn,
            retryable_exceptions=(_RetryError,),
            max_retries=2,
            base_delay=7,
            on_retry=on_retry,
        )
    on_retry.assert_called_once()
    exc_arg, attempt_arg, sleep_arg = on_retry.call_args[0]
    assert isinstance(exc_arg, _RetryError)
    assert attempt_arg == 1
    assert sleep_arg == 7


def test_does_not_catch_non_retryable_exceptions():
    fn = MagicMock(side_effect=_OtherError())
    with pytest.raises(_OtherError):
        retry_with_backoff(fn, retryable_exceptions=(_RetryError,))
    fn.assert_called_once()
