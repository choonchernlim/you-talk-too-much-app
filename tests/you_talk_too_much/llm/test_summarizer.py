from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from google.genai import errors

from you_talk_too_much.config import settings
from you_talk_too_much.llm.summarizer import (
    _EXTRACTION_PROMPT,
    _FORMAT_PROMPT,
    LLM,
    MAX_OUTPUT_TOKENS,
)


@pytest.fixture
def llm_instance():
    """Fixture to create a real LLM instance using config from .env."""
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


def test_generate_returns_response_text(mock_llm: LLM) -> None:
    response = Mock()
    response.text = "generated text"
    mock_client = cast("MagicMock", mock_llm.client)
    mock_client.models.generate_content.return_value = response

    result = mock_llm._generate("prompt", "content", 1024)

    assert result == "generated text"


def test_generate_raises_when_response_is_empty(mock_llm: LLM) -> None:
    response = Mock()
    response.text = None
    mock_client = cast("MagicMock", mock_llm.client)
    mock_client.models.generate_content.return_value = response

    with pytest.raises(RuntimeError, match="Empty response"):
        mock_llm._generate("prompt", "content", 1024)


def test_generate_retries_on_429_then_succeeds(mock_llm: LLM) -> None:
    success = Mock()
    success.text = "ok"
    mock_client = cast("MagicMock", mock_llm.client)
    mock_client.models.generate_content.side_effect = [
        _api_error(429),
        success,
    ]

    with patch("time.sleep") as mock_sleep:
        result = mock_llm._generate("prompt", "content", 1024)

    assert result == "ok"
    mock_sleep.assert_called_once_with(5)  # BASE_RETRY_DELAY * 2**0


def test_generate_raises_api_error_after_exhausting_retries(mock_llm: LLM) -> None:
    mock_client = cast("MagicMock", mock_llm.client)
    mock_client.models.generate_content.side_effect = _api_error(429)

    with patch("time.sleep"), pytest.raises(errors.APIError):
        mock_llm._generate("prompt", "content", 1024)

    assert mock_client.models.generate_content.call_count == 4  # MAX_RETRIES


def test_generate_raises_immediately_on_non_429_api_error(mock_llm: LLM) -> None:
    mock_client = cast("MagicMock", mock_llm.client)
    mock_client.models.generate_content.side_effect = _api_error(500)

    with pytest.raises(errors.APIError):
        mock_llm._generate("prompt", "content", 1024)

    assert mock_client.models.generate_content.call_count == 1


def test_extract_calls_generate_with_extraction_prompt_and_token_limit(
    mock_llm: LLM,
) -> None:
    with patch.object(mock_llm, "_generate", return_value="raw notes") as mock_gen:
        result = mock_llm._extract("transcript text")

    mock_gen.assert_called_once_with(
        _EXTRACTION_PROMPT, "transcript text", MAX_OUTPUT_TOKENS
    )
    assert result == "raw notes"


def test_format_calls_generate_with_format_prompt_and_token_limit(
    mock_llm: LLM,
) -> None:
    with patch.object(
        mock_llm, "_generate", return_value="# TL;DR\n\n* summary"
    ) as mock_gen:
        md, _html = mock_llm._format("extracted notes")

    mock_gen.assert_called_once_with(
        _FORMAT_PROMPT, "extracted notes", MAX_OUTPUT_TOKENS
    )
    assert md == "# TL;DR\n\n* summary"


def test_format_converts_markdown_to_html(mock_llm: LLM) -> None:
    with patch.object(
        mock_llm, "_generate", return_value="# TL;DR\n\n* **Label:** detail"
    ):
        _, html = mock_llm._format("extracted notes")

    assert "<h1>" in html
    assert "<li>" in html
    assert "<strong>" in html


def test_summarize_chains_extract_then_format(mock_llm: LLM) -> None:
    with (
        patch.object(mock_llm, "_extract", return_value="raw") as mock_extract,
        patch.object(
            mock_llm, "_format", return_value=("md", "<p>html</p>")
        ) as mock_format,
    ):
        md, html = mock_llm.summarize("transcript")

    mock_extract.assert_called_once_with("transcript")
    mock_format.assert_called_once_with("raw")
    assert md == "md"
    assert html == "<p>html</p>"


# @pytest.mark.manual
# def test_llm_initialization(llm_instance):
#     """Test that the LLM is initialized correctly with real settings."""
#     assert llm_instance.model_id == settings.gcp_vertex_model
#     assert llm_instance.client is not None


@pytest.mark.manual
def test_llm_summarize_success(llm_instance):
    """Test the summarize method with a real request to Vertex AI."""
    # read this file transcripts/2026-03-23 AM 11:35/conversation.txt
    with Path("transcripts/2026-03-23 AM 11:35/conversation.txt").open() as f:
        sample_text = f.read()

    text_content, html_content = llm_instance.summarize(sample_text)

    # save html_content into a file "test_output.html" for manual inspection
    with Path("test_output.html").open("w") as f:
        f.write(html_content)
    #
    # # Since it's real LLM output, we can't assert exact strings,
    # # but we can verify formatting.
    # assert text_content, "Text content should not be empty"
    # assert html_content, "HTML content should not be empty"
    #
    assert (
        "Executive Summary" in text_content
        or "Key Decisions & Discussion Points" in text_content
    )
    # assert "<h1>" in html_content or "<h2>" in html_content
