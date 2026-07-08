import queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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
        capturer._buffer = [np.ones(AudioCapturer.TARGET_RATE * 6)]

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
        capturer._buffer = [np.ones(AudioCapturer.TARGET_RATE * 6)]

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
        capturer._chunk_queue.put(np.ones(AudioCapturer.TARGET_RATE * 6))

        capturer.tick()

        # Queue should be drained into buffer
        assert capturer._chunk_queue.empty()
        assert len(capturer._buffer) == 1


@patch("you_talk_too_much.audio.capturer.sd")
@patch("you_talk_too_much.audio.capturer.load_silero_vad")
class TestAudioCapturerStartStop:
    def test_start_opens_stream_at_native_rate(
        self, _mock_vad: MagicMock, mock_sd: MagicMock
    ) -> None:
        mock_sd.query_devices.return_value = {"default_samplerate": 48000.0}
        capturer = AudioCapturer(on_audio_ready=MagicMock())
        capturer.start()

        mock_sd.InputStream.assert_called_once_with(
            samplerate=48000,
            channels=1,
            callback=capturer._audio_callback,
            dtype="float32",
        )
        mock_sd.InputStream.return_value.start.assert_called_once()

    def test_start_computes_resample_ratio(
        self, _mock_vad: MagicMock, mock_sd: MagicMock
    ) -> None:
        mock_sd.query_devices.return_value = {"default_samplerate": 48000.0}
        capturer = AudioCapturer(on_audio_ready=MagicMock())
        capturer.start()

        # 48000 -> 16000 = ratio 3:1, so up=1 down=3
        assert capturer._resample_up == 1
        assert capturer._resample_down == 3

    def test_start_sets_unity_ratio_when_native_matches_target(
        self, _mock_vad: MagicMock, mock_sd: MagicMock
    ) -> None:
        mock_sd.query_devices.return_value = {"default_samplerate": 16000.0}
        capturer = AudioCapturer(on_audio_ready=MagicMock())
        capturer.start()

        assert capturer._resample_up == 1
        assert capturer._resample_down == 1

    def test_stop_closes_stream_and_flushes(
        self, _mock_vad: MagicMock, mock_sd: MagicMock
    ) -> None:
        mock_sd.query_devices.return_value = {"default_samplerate": 16000.0}
        callback = MagicMock()
        capturer = AudioCapturer(on_audio_ready=callback)
        capturer.start()

        # Simulate audio in queue
        capturer._chunk_queue.put(np.array([1.0, 2.0, 3.0]))

        capturer.stop()

        # Graceful stop() (not abort()) to keep PortAudio state healthy
        mock_sd.InputStream.return_value.stop.assert_called_once()
        mock_sd.InputStream.return_value.abort.assert_not_called()
        mock_sd.InputStream.return_value.close.assert_called_once()
        callback.assert_called_once()

    def test_stop_flushes_and_clears_stream_even_when_stream_stop_raises(
        self, _mock_vad: MagicMock, mock_sd: MagicMock
    ) -> None:
        mock_sd.query_devices.return_value = {"default_samplerate": 16000.0}
        callback = MagicMock()
        capturer = AudioCapturer(on_audio_ready=callback)
        capturer.start()

        capturer._chunk_queue.put(np.array([1.0, 2.0, 3.0]))
        mock_sd.InputStream.return_value.stop.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            capturer.stop()

        mock_sd.InputStream.return_value.close.assert_called_once()
        assert capturer._stream is None
        callback.assert_called_once()

    def test_start_clears_previous_state(
        self, _mock_vad: MagicMock, mock_sd: MagicMock
    ) -> None:
        mock_sd.query_devices.return_value = {"default_samplerate": 48000.0}
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


class FakePortAudioError(Exception):
    pass


@patch("you_talk_too_much.audio.capturer.time.sleep")
@patch("you_talk_too_much.audio.capturer.sd")
@patch("you_talk_too_much.audio.capturer.load_silero_vad")
class TestAudioCapturerStartRetry:
    def _configure_sd(self, mock_sd: MagicMock) -> None:
        mock_sd.PortAudioError = FakePortAudioError
        mock_sd.query_devices.return_value = {"default_samplerate": 16000.0}

    def test_start_reinitializes_portaudio_and_retries_on_open_failure(
        self, _mock_vad: MagicMock, mock_sd: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        self._configure_sd(mock_sd)
        good_stream = MagicMock()
        mock_sd.InputStream.side_effect = [
            FakePortAudioError("open failed"),
            good_stream,
        ]

        capturer = AudioCapturer(on_audio_ready=MagicMock())
        capturer.start()

        mock_sd._terminate.assert_called_once()
        mock_sd._initialize.assert_called_once()
        assert capturer._stream is good_stream
        good_stream.start.assert_called_once()

    def test_start_closes_stream_and_retries_when_stream_start_fails(
        self, _mock_vad: MagicMock, mock_sd: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        self._configure_sd(mock_sd)
        bad_stream = MagicMock()
        bad_stream.start.side_effect = FakePortAudioError("start failed")
        good_stream = MagicMock()
        mock_sd.InputStream.side_effect = [bad_stream, good_stream]

        capturer = AudioCapturer(on_audio_ready=MagicMock())
        capturer.start()

        bad_stream.close.assert_called_once_with(ignore_errors=True)
        mock_sd._terminate.assert_called_once()
        mock_sd._initialize.assert_called_once()
        assert capturer._stream is good_stream

    def test_start_raises_when_retry_also_fails(
        self, _mock_vad: MagicMock, mock_sd: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        self._configure_sd(mock_sd)
        mock_sd.InputStream.side_effect = FakePortAudioError("open failed")

        capturer = AudioCapturer(on_audio_ready=MagicMock())

        with pytest.raises(FakePortAudioError):
            capturer.start()

        assert capturer._stream is None


@patch("you_talk_too_much.audio.capturer.load_silero_vad")
class TestAudioCapturerResampling:
    def test_drain_queue_resamples_when_rates_differ(
        self, _mock_vad: MagicMock
    ) -> None:
        capturer = AudioCapturer(on_audio_ready=MagicMock())
        # Simulate 48kHz -> 16kHz (3:1 downsample)
        capturer._resample_up = 1
        capturer._resample_down = 3

        # 300 samples at 48kHz should become 100 samples at 16kHz
        chunk = np.ones(300, dtype=np.float32)
        capturer._chunk_queue.put(chunk)

        capturer._drain_queue()

        assert len(capturer._buffer) == 1
        assert capturer._buffer[0].shape[0] == 100

    def test_drain_queue_skips_resample_when_rates_match(
        self, _mock_vad: MagicMock
    ) -> None:
        capturer = AudioCapturer(on_audio_ready=MagicMock())
        # Unity ratio — no resampling needed
        capturer._resample_up = 1
        capturer._resample_down = 1

        chunk = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        capturer._chunk_queue.put(chunk)

        capturer._drain_queue()

        assert len(capturer._buffer) == 1
        np.testing.assert_array_equal(capturer._buffer[0], chunk)

    def test_resampled_chunks_are_float32(self, _mock_vad: MagicMock) -> None:
        capturer = AudioCapturer(on_audio_ready=MagicMock())
        capturer._resample_up = 1
        capturer._resample_down = 3

        chunk = np.ones(300, dtype=np.float32)
        capturer._chunk_queue.put(chunk)

        capturer._drain_queue()

        assert capturer._buffer[0].dtype == np.float32
