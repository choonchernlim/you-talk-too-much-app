import queue
from unittest.mock import MagicMock, patch

import numpy as np

from you_talk_too_much.audio.capturer import (
    AudioCapturer,
    _drain_all,
    _extract_tail,
)


class TestExtractTail:
    def test_extracts_exact_tail_from_single_chunk(self) -> None:
        buffer = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
        result = _extract_tail(buffer, 3)
        np.testing.assert_array_equal(result, [3.0, 4.0, 5.0])

    def test_extracts_tail_spanning_multiple_chunks(self) -> None:
        buffer = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        result = _extract_tail(buffer, 4)
        np.testing.assert_array_equal(result, [3.0, 4.0, 5.0, 6.0])

    def test_returns_all_samples_when_requesting_more_than_available(self) -> None:
        buffer = [np.array([1.0, 2.0, 3.0])]
        result = _extract_tail(buffer, 10)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_handles_2d_chunks_from_sounddevice(self) -> None:
        # sounddevice produces (frames, channels) shaped arrays
        buffer = [np.array([[1.0], [2.0], [3.0]]), np.array([[4.0], [5.0]])]
        result = _extract_tail(buffer, 3)
        np.testing.assert_array_equal(result, [3.0, 4.0, 5.0])

    def test_extracts_single_sample(self) -> None:
        buffer = [np.array([1.0, 2.0, 3.0])]
        result = _extract_tail(buffer, 1)
        np.testing.assert_array_equal(result, [3.0])


class TestDrainAll:
    def test_empties_queue(self) -> None:
        q: queue.Queue[int] = queue.Queue()
        q.put(1)
        q.put(2)
        q.put(3)
        _drain_all(q)
        assert q.empty()

    def test_handles_empty_queue(self) -> None:
        q: queue.Queue[int] = queue.Queue()
        _drain_all(q)
        assert q.empty()


@patch("you_talk_too_much.audio.capturer.load_silero_vad")
class TestAudioCapturerDrainQueue:
    def test_moves_all_queue_items_to_buffer(self, _mock_vad: MagicMock) -> None:
        callback = MagicMock()
        capturer = AudioCapturer(on_audio_ready=callback)

        chunk1 = np.array([1.0, 2.0])
        chunk2 = np.array([3.0, 4.0])
        capturer._chunk_queue.put(chunk1)
        capturer._chunk_queue.put(chunk2)

        capturer._drain_queue()

        assert len(capturer._buffer) == 2
        np.testing.assert_array_equal(capturer._buffer[0], chunk1)
        np.testing.assert_array_equal(capturer._buffer[1], chunk2)
        assert capturer._chunk_queue.empty()

    def test_drain_on_empty_queue_is_noop(self, _mock_vad: MagicMock) -> None:
        capturer = AudioCapturer(on_audio_ready=MagicMock())
        capturer._drain_queue()
        assert capturer._buffer == []


@patch("you_talk_too_much.audio.capturer.load_silero_vad")
class TestAudioCapturerProcessAndClear:
    def test_concatenates_buffer_and_calls_callback(self, _mock_vad: MagicMock) -> None:
        callback = MagicMock()
        capturer = AudioCapturer(on_audio_ready=callback)
        capturer._buffer = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        capturer._process_and_clear()

        callback.assert_called_once()
        audio_data = callback.call_args[0][0]
        np.testing.assert_array_equal(audio_data, [1.0, 2.0, 3.0, 4.0])
        assert capturer._buffer == []

    def test_skips_callback_when_buffer_empty(self, _mock_vad: MagicMock) -> None:
        callback = MagicMock()
        capturer = AudioCapturer(on_audio_ready=callback)

        capturer._process_and_clear()

        callback.assert_not_called()


@patch("you_talk_too_much.audio.capturer.get_speech_timestamps")
@patch("you_talk_too_much.audio.capturer.load_silero_vad")
class TestAudioCapturerTick:
    def test_skips_processing_when_buffer_below_minimum(
        self, _mock_vad: MagicMock, mock_vad_check: MagicMock
    ) -> None:
        capturer = AudioCapturer(on_audio_ready=MagicMock())
        # Add less than 5 seconds of audio (RATE * 5 = 80000 samples)
        capturer._buffer = [np.zeros(1000)]

        capturer.tick()

        mock_vad_check.assert_not_called()

    def test_processes_buffer_when_silence_detected(
        self, _mock_vad: MagicMock, mock_vad_check: MagicMock
    ) -> None:
        callback = MagicMock()
        capturer = AudioCapturer(on_audio_ready=callback)
        # Add > 5 seconds of audio
        capturer._buffer = [np.ones(AudioCapturer.RATE * 6)]

        # No speech timestamps = silence
        mock_vad_check.return_value = []

        capturer.tick()

        callback.assert_called_once()
        assert capturer._buffer == []

    def test_does_not_process_when_speech_detected(
        self, _mock_vad: MagicMock, mock_vad_check: MagicMock
    ) -> None:
        callback = MagicMock()
        capturer = AudioCapturer(on_audio_ready=callback)
        capturer._buffer = [np.ones(AudioCapturer.RATE * 6)]

        # Speech detected
        mock_vad_check.return_value = [{"start": 0, "end": 100}]

        capturer.tick()

        callback.assert_not_called()
        assert len(capturer._buffer) == 1

    def test_drains_queue_before_vad_check(
        self, _mock_vad: MagicMock, mock_vad_check: MagicMock
    ) -> None:
        capturer = AudioCapturer(on_audio_ready=MagicMock())
        mock_vad_check.return_value = [{"start": 0, "end": 100}]

        # Put chunks in queue (not in buffer yet)
        capturer._chunk_queue.put(np.ones(AudioCapturer.RATE * 6))

        capturer.tick()

        # Queue should be drained into buffer
        assert capturer._chunk_queue.empty()
        assert len(capturer._buffer) == 1


@patch("you_talk_too_much.audio.capturer.sd")
@patch("you_talk_too_much.audio.capturer.load_silero_vad")
class TestAudioCapturerStartStop:
    def test_start_opens_stream(self, _mock_vad: MagicMock, mock_sd: MagicMock) -> None:
        capturer = AudioCapturer(on_audio_ready=MagicMock())
        capturer.start()

        mock_sd.InputStream.assert_called_once()
        mock_sd.InputStream.return_value.start.assert_called_once()

    def test_stop_closes_stream_and_flushes(
        self, _mock_vad: MagicMock, mock_sd: MagicMock
    ) -> None:
        callback = MagicMock()
        capturer = AudioCapturer(on_audio_ready=callback)
        capturer.start()

        # Simulate audio in queue
        capturer._chunk_queue.put(np.array([1.0, 2.0, 3.0]))

        capturer.stop()

        mock_sd.InputStream.return_value.abort.assert_called_once()
        mock_sd.InputStream.return_value.close.assert_called_once()
        callback.assert_called_once()

    def test_start_clears_previous_state(
        self, _mock_vad: MagicMock, _mock_sd: MagicMock
    ) -> None:
        capturer = AudioCapturer(on_audio_ready=MagicMock())
        capturer._buffer = [np.array([1.0])]
        capturer._chunk_queue.put(np.array([2.0]))

        capturer.start()

        assert capturer._buffer == []
        assert capturer._chunk_queue.empty()

    def test_stop_when_no_stream_is_safe(
        self, _mock_vad: MagicMock, _mock_sd: MagicMock
    ) -> None:
        capturer = AudioCapturer(on_audio_ready=MagicMock())
        capturer._stream = None
        capturer.stop()  # should not raise
