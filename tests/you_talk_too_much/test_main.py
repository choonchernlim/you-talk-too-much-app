import time
from unittest.mock import MagicMock, patch

from you_talk_too_much.main import TICK_INTERVAL, LoopState, handle_key, handle_tick


class TestHandleKey:
    def test_key_1_starts_capture(self) -> None:
        session = MagicMock()
        state = LoopState()

        handle_key(session, state, "1")

        session.start.assert_called_once()
        assert state.is_capture_started is True
        assert state.last_tick > 0

    def test_key_1_ignored_when_already_capturing(self) -> None:
        session = MagicMock()
        state = LoopState(is_capture_started=True)

        handle_key(session, state, "1")

        session.start.assert_not_called()

    def test_key_1_failure_leaves_capture_stopped(self) -> None:
        session = MagicMock()
        session.start.side_effect = RuntimeError("no mic")
        state = LoopState()

        handle_key(session, state, "1")  # must not raise

        assert state.is_capture_started is False

    def test_key_2_stops_capture(self) -> None:
        session = MagicMock()
        state = LoopState(is_capture_started=True)

        handle_key(session, state, "2")

        session.stop.assert_called_once()
        assert state.is_capture_started is False

    def test_key_2_failure_still_marks_capture_stopped(self) -> None:
        session = MagicMock()
        session.stop.side_effect = RuntimeError("boom")
        state = LoopState(is_capture_started=True)

        handle_key(session, state, "2")  # must not raise

        assert state.is_capture_started is False

    def test_key_2_ignored_when_not_capturing(self) -> None:
        session = MagicMock()
        state = LoopState()

        handle_key(session, state, "2")

        session.stop.assert_not_called()

    @patch("you_talk_too_much.main.input", create=True, return_value=" my-dir ")
    def test_key_3_summarizes_existing_with_stripped_name(
        self, _mock_input: MagicMock
    ) -> None:
        session = MagicMock()
        state = LoopState()

        handle_key(session, state, "3")

        session.summarize_existing.assert_called_once_with("my-dir")

    @patch("you_talk_too_much.main.input", create=True, return_value="bad")
    def test_key_3_failure_is_contained(self, _mock_input: MagicMock) -> None:
        session = MagicMock()
        session.summarize_existing.side_effect = FileNotFoundError("nope")
        state = LoopState()

        handle_key(session, state, "3")  # must not raise

    def test_key_4_requests_quit(self) -> None:
        session = MagicMock()
        state = LoopState(is_capture_started=True)

        handle_key(session, state, "4")

        assert state.should_quit is True


class TestHandleTick:
    def test_ticks_when_capture_active_and_interval_elapsed(self) -> None:
        session = MagicMock()
        state = LoopState(
            is_capture_started=True,
            last_tick=time.monotonic() - TICK_INTERVAL - 1,
        )

        handle_tick(session, state)

        session.tick.assert_called_once()

    def test_does_not_tick_before_interval(self) -> None:
        session = MagicMock()
        state = LoopState(is_capture_started=True, last_tick=time.monotonic())

        handle_tick(session, state)

        session.tick.assert_not_called()

    def test_does_not_tick_when_not_capturing(self) -> None:
        session = MagicMock()
        state = LoopState()

        handle_tick(session, state)

        session.tick.assert_not_called()

    def test_tick_failure_is_contained_and_recording_continues(self) -> None:
        session = MagicMock()
        session.tick.side_effect = RuntimeError("transcription hiccup")
        state = LoopState(
            is_capture_started=True,
            last_tick=time.monotonic() - TICK_INTERVAL - 1,
        )

        handle_tick(session, state)  # must not raise

        assert state.is_capture_started is True
        assert state.last_tick > 0
