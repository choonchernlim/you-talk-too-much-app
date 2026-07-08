import os
from unittest.mock import MagicMock, patch

# Importing you_talk_too_much.app pulls in config, which fails fast when
# required settings are missing (e.g. in CI, where there is no .env file).
# Provide dummy values; real env vars / .env still take precedence locally.
for _key in (
    "GCP_VERTEX_PROJECT",
    "GCP_VERTEX_LOCATION",
    "GCP_VERTEX_SA_KEY",
    "GCP_VERTEX_MODEL",
    "ONENOTE_SECTION_NAME",
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "HF_WHISPER_MODEL",
    "HF_DIARIZATION_MODEL",
    "HF_EMBEDDING_MODEL",
    "HF_TOKEN",
):
    os.environ.setdefault(_key, "test-value")

from you_talk_too_much.app import AppSession  # noqa: E402


@patch("you_talk_too_much.app.OneNoteClient")
@patch("you_talk_too_much.app.LLM")
@patch("you_talk_too_much.app.AudioCapturer")
@patch("you_talk_too_much.app.MLXTranscriber")
@patch("you_talk_too_much.app.FileManager")
class TestAppSessionStop:
    def test_stop_survives_summarize_failure(
        self,
        mock_file_manager: MagicMock,
        _mock_transcriber: MagicMock,
        mock_capturer: MagicMock,
        mock_llm: MagicMock,
        _mock_onenote: MagicMock,
    ) -> None:
        mock_file_manager.return_value.read_conversation.return_value = "some text"
        mock_llm.return_value.summarize.side_effect = RuntimeError("LLM down")

        session = AppSession()
        session.stop()  # must not raise

        mock_capturer.return_value.stop.assert_called_once()

    def test_stop_survives_onenote_failure(
        self,
        mock_file_manager: MagicMock,
        _mock_transcriber: MagicMock,
        mock_capturer: MagicMock,
        mock_llm: MagicMock,
        mock_onenote: MagicMock,
    ) -> None:
        mock_file_manager.return_value.read_conversation.return_value = "some text"
        mock_llm.return_value.summarize.return_value = ("summary", "topic")
        mock_onenote.return_value.create_page.side_effect = RuntimeError("Graph down")

        session = AppSession()
        session.stop()  # must not raise

        mock_capturer.return_value.stop.assert_called_once()

    def test_stop_summarizes_and_pushes_on_success(
        self,
        mock_file_manager: MagicMock,
        _mock_transcriber: MagicMock,
        mock_capturer: MagicMock,
        mock_llm: MagicMock,
        mock_onenote: MagicMock,
    ) -> None:
        mock_file_manager.return_value.read_conversation.return_value = "some text"
        mock_llm.return_value.summarize.return_value = ("summary", "topic")

        session = AppSession()
        session.stop()

        mock_capturer.return_value.stop.assert_called_once()
        mock_onenote.return_value.create_page.assert_called_once()
