from pathlib import Path

import pytest

from you_talk_too_much.config import settings
from you_talk_too_much.llm.summarizer import LLM


@pytest.fixture
def llm_instance():
    """Fixture to create a real LLM instance using config from .env."""
    return LLM(
        project=settings.gcp_vertex_project,
        location=settings.gcp_vertex_location,
        sa_key=settings.gcp_vertex_sa_key,
        model=settings.gcp_vertex_model,
    )


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
