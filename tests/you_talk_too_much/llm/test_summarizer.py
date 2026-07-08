from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from google.genai import errors

from you_talk_too_much.config import get_settings
from you_talk_too_much.llm.prompts import EXTRACTION_PROMPT, FORMAT_PROMPT
from you_talk_too_much.llm.summarizer import LLM, SummaryOutput


@pytest.fixture
def llm_instance():
    """Fixture to create a real LLM instance using config from .env."""
    settings = get_settings()
    return LLM(
        project=settings.gcp_vertex_project,
        location=settings.gcp_vertex_location,
        sa_key=settings.gcp_vertex_sa_key,
        model=settings.gcp_vertex_model,
    )


@pytest.fixture
def mock_llm() -> LLM:
    """LLM instance with a mocked Vertex AI client — no real credentials needed."""
    with (
        patch("google.oauth2.service_account.Credentials.from_service_account_file"),
        patch("google.genai.Client"),
    ):
        instance = LLM(project="p", location="l", sa_key="k.json", model="m")
    instance.client = MagicMock()
    return instance


def _api_error(code: int) -> errors.APIError:
    """Construct a minimal errors.APIError with the given HTTP status code.

    Uses __new__ to bypass the constructor — the google-genai APIError constructor
    signature varies across versions, but isinstance checks still pass.
    """
    err = errors.APIError.__new__(errors.APIError)
    err.code = code
    return err


def _response(text: str | None = "generated text", parsed: object = None) -> Mock:
    response = Mock()
    response.text = text
    response.parsed = parsed
    return response


def test_generate_returns_response(mock_llm: LLM) -> None:
    mock_client = cast("MagicMock", mock_llm.client)
    mock_client.models.generate_content.return_value = _response("generated text")

    result = mock_llm._generate("prompt", "content")

    assert result.text == "generated text"


def test_generate_passes_response_schema_for_structured_output(mock_llm: LLM) -> None:
    mock_client = cast("MagicMock", mock_llm.client)
    mock_client.models.generate_content.return_value = _response("{}")

    mock_llm._generate("prompt", "content", response_schema=SummaryOutput)

    config = mock_client.models.generate_content.call_args.kwargs["config"]
    assert config.response_schema is SummaryOutput
    assert config.response_mime_type == "application/json"


def test_generate_omits_schema_for_plain_text(mock_llm: LLM) -> None:
    mock_client = cast("MagicMock", mock_llm.client)
    mock_client.models.generate_content.return_value = _response("ok")

    mock_llm._generate("prompt", "content")

    config = mock_client.models.generate_content.call_args.kwargs["config"]
    assert config.response_schema is None
    assert config.response_mime_type is None


def test_generate_raises_when_response_is_empty(mock_llm: LLM) -> None:
    mock_client = cast("MagicMock", mock_llm.client)
    mock_client.models.generate_content.return_value = _response(text=None)

    with pytest.raises(RuntimeError, match="Empty response"):
        mock_llm._generate("prompt", "content")


def test_generate_retries_on_429_then_succeeds(mock_llm: LLM) -> None:
    mock_client = cast("MagicMock", mock_llm.client)
    mock_client.models.generate_content.side_effect = [
        _api_error(429),
        _response("ok"),
    ]

    with patch("you_talk_too_much.common.retry.time.sleep") as mock_sleep:
        result = mock_llm._generate("prompt", "content")

    assert result.text == "ok"
    mock_sleep.assert_called_once_with(5)  # BASE_RETRY_DELAY * 2**0


def test_generate_raises_api_error_after_exhausting_retries(mock_llm: LLM) -> None:
    mock_client = cast("MagicMock", mock_llm.client)
    mock_client.models.generate_content.side_effect = _api_error(429)

    with (
        patch("you_talk_too_much.common.retry.time.sleep"),
        pytest.raises(errors.APIError),
    ):
        mock_llm._generate("prompt", "content")

    assert mock_client.models.generate_content.call_count == 4  # MAX_RETRIES


def test_generate_raises_immediately_on_non_429_api_error(mock_llm: LLM) -> None:
    mock_client = cast("MagicMock", mock_llm.client)
    mock_client.models.generate_content.side_effect = _api_error(500)

    with pytest.raises(errors.APIError):
        mock_llm._generate("prompt", "content")

    assert mock_client.models.generate_content.call_count == 1


def test_extract_calls_generate_with_extraction_prompt(mock_llm: LLM) -> None:
    with patch.object(
        mock_llm, "_generate", return_value=_response("raw notes")
    ) as mock_gen:
        result = mock_llm._extract("transcript text")

    mock_gen.assert_called_once_with(EXTRACTION_PROMPT, "transcript text")
    assert result == "raw notes"


def test_format_with_topic_returns_parsed_fields(mock_llm: LLM) -> None:
    parsed = SummaryOutput(summary_markdown="# TL;DR\n\n* summary", topic="Care; CTN")
    with patch.object(
        mock_llm, "_generate", return_value=_response("{}", parsed=parsed)
    ) as mock_gen:
        md, topic = mock_llm._format_with_topic("extracted notes")

    mock_gen.assert_called_once_with(
        FORMAT_PROMPT, "extracted notes", response_schema=SummaryOutput
    )
    assert md == "# TL;DR\n\n* summary"
    assert topic == "Care; CTN"


def test_format_with_topic_collapses_topic_whitespace(mock_llm: LLM) -> None:
    parsed = SummaryOutput(summary_markdown="md", topic="Azoda\nCTN\nFAE\n")
    with patch.object(mock_llm, "_generate", return_value=_response("{}", parsed)):
        _md, topic = mock_llm._format_with_topic("notes")

    assert topic == "Azoda CTN FAE"


def test_format_with_topic_raises_on_unparsed_response(mock_llm: LLM) -> None:
    with (
        patch.object(mock_llm, "_generate", return_value=_response("{}", parsed=None)),
        pytest.raises(RuntimeError, match="Unexpected structured response"),
    ):
        mock_llm._format_with_topic("notes")


def test_summarize_chains_extract_then_format_with_topic(mock_llm: LLM) -> None:
    with (
        patch.object(mock_llm, "_extract", return_value="raw") as mock_extract,
        patch.object(
            mock_llm, "_format_with_topic", return_value=("md", "Care Team")
        ) as mock_format,
    ):
        md, topic = mock_llm.summarize("transcript")

    mock_extract.assert_called_once_with("transcript")
    mock_format.assert_called_once_with("raw")
    assert md == "md"
    assert topic == "Care Team"


@pytest.mark.manual
def test_llm_summarize_success(llm_instance):
    """Test the summarize method with a real request to Vertex AI."""
    # read this file transcripts/2026-03-23 AM 11:35/conversation.txt
    with Path("transcripts/2026-03-23 AM 11:35/conversation.txt").open() as f:
        sample_text = f.read()

    text_content, _topic = llm_instance.summarize(sample_text)

    # save text content into a file "test_output.md" for manual inspection
    with Path("test_output.md").open("w") as f:
        f.write(text_content)

    assert (
        "Executive Summary" in text_content
        or "Key Decisions & Discussion Points" in text_content
    )
