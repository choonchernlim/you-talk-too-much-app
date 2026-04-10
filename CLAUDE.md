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

1. `main.py` — CLI menu loop; user presses keys to start/stop recording
2. `app.py` (`AppSession`) — orchestrates the session lifecycle
3. `audio/capturer.py` (`AudioCapturer`) — captures audio at the device's native rate, resamples to 16 kHz, buffers in a queue, and emits chunks when VAD detects silence
4. `transcription/transcriber.py` (`MLXTranscriber`) — transcribes chunks with MLX-Whisper (Apple Silicon), runs Pyannote diarization, and tracks speakers across segments using cosine similarity on embeddings (threshold 0.72)
5. On stop: `llm/summarizer.py` (`LLM`) reads the full conversation from disk and calls Vertex AI Gemini to produce a markdown + HTML summary
6. `integrations/onenote.py` (`OneNoteClient`) authenticates via MSAL and POSTs the HTML to Microsoft Graph API

**Transcript persistence:** `storage/file_manager.py` (`FileManager`) writes each transcribed segment to `conversation.txt` and raw diarization output to `raw.jsonl` under a timestamped directory.

**Configuration:** `config.py` uses Pydantic Settings; all credentials (GCP, Azure, HuggingFace) are loaded from `.env` at startup. The app fails fast if any required key is missing. See `.env.sample` for required keys.

## Key constraints

- **Ruff line length: 88.** Rules include ANN (annotations required), N (naming), S (security), ARG (unused args), SIM, and others. Tests relax ANN, S101, and a few others — see `pyproject.toml`.
- **Type checking with `ty` (strict).** Unresolved references and invalid argument types are errors.
- **`@pytest.mark.manual`** marks tests that hit live external APIs; these are excluded from the default `pytest` run and from CI.
- Tests mirror source structure exactly: source at `src/you_talk_too_much/foo/bar.py` → test at `tests/you_talk_too_much/foo/test_bar.py`.
- Use `uv add` / `uv run` — never `pip` or `python` directly.
