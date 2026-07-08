# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the app
uv run you-talk-too-much

# Lint / format
uv run ruff check .
uv run ruff format .

# Type check
uv run ty check

# Run all tests (manual tests excluded by default)
uv run pytest

# Run a single test file
uv run pytest tests/you_talk_too_much/audio/test_capturer.py

# Run a single test
uv run pytest tests/you_talk_too_much/audio/test_capturer.py::TestAudioCapturerTick::test_processes_buffer_when_silence_detected

# Run manual tests (require live credentials / external services)
uv run pytest -m manual

# Run all pre-commit checks (ruff, ty, secrets, pytest)
pre-commit run --all-files
```

## Architecture

**What it does:** Records meetings via microphone, transcribes with speaker labels, summarizes with an LLM, and writes the summary to Microsoft OneNote.

**Data flow:**

1. `main.py` — CLI menu loop; user presses keys to start/stop recording. The main thread only polls keys and runs the cheap VAD tick; all heavy work is offloaded.
2. `app.py` (`AppSession`) — orchestrates session lifecycles across two serial background lanes (see thread model below)
3. `audio/capturer.py` (`AudioCapturer`) — captures audio at the device's native rate, resamples to 16 kHz, buffers in a queue, and emits chunks when VAD detects silence
4. `transcription/transcriber.py` (`MLXTranscriber`) — transcribes chunks with MLX-Whisper (Apple Silicon), runs Pyannote diarization, and tracks speakers across segments using cosine similarity on embeddings (threshold 0.72)
5. On stop: `llm/summarizer.py` (`LLM`) reads the full conversation from disk and calls Vertex AI Gemini (extraction call, then a structured-output call returning markdown + topic) to produce the summary
6. `integrations/onenote.py` (`OneNoteClient`) authenticates via MSAL and POSTs the HTML to Microsoft Graph API

**Thread model:** exactly three threads, coordinated only by job ordering (no locks):

- The **PortAudio callback thread** writes raw audio into `AudioCapturer._chunk_queue`.
- The **transcription runner** (`common/task_runner.py`, `TaskRunner`) serially runs chunk transcription jobs and transcript file writes; it owns `MLXTranscriber`/`SpeakerTracker` (even `reset` is submitted as a job).
- The **summarize runner** serially runs LLM + OneNote jobs; it is handed an immutable `TranscriptSession` snapshot per finished recording, so a new recording can start while the previous summary uploads. `stop()` returns immediately; `AppSession.shutdown()` (called from `main.run()`'s `finally`) waits for all pending work.

**Transcript persistence:** `storage/file_manager.py` — `FileManager` creates/loads timestamped directories and returns frozen `TranscriptSession` handles that write `conversation.txt`, `raw.jsonl`, and the summary files. Background jobs must only touch files through their own `TranscriptSession`.

**Configuration:** `config.py` uses Pydantic Settings via lazy `get_settings()` (cached); all credentials (GCP, Azure, HuggingFace) load from `.env` on first use and fail fast if any required key is missing. See `.env.sample` for required keys. Tests get dummy env values from `tests/conftest.py`.

**Diagnostics:** all log output (with thread names and full tracebacks) also goes to a rotating file at `~/.you-talk-too-much/logs/app.log`; check it first when debugging intermittent/threading issues.

## Key constraints

- **Ruff line length: 88.** Rules include ANN (annotations required), N (naming), S (security), ARG (unused args), SIM, and others. Tests relax ANN, S101, and a few others — see `pyproject.toml`.
- **Type checking with `ty` (strict).** Unresolved references and invalid argument types are errors.
- **`@pytest.mark.manual`** marks tests that hit live external APIs; these are excluded from the default `pytest` run and from CI.
- Tests mirror source structure exactly: source at `src/you_talk_too_much/foo/bar.py` → test at `tests/you_talk_too_much/foo/test_bar.py`.
- Use `uv add` / `uv run` — never `pip` or `python` directly.
