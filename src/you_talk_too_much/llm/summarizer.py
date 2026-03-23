import time

from google import genai
from google.genai import errors
from google.oauth2 import service_account
from markdown import markdown

from you_talk_too_much.cli.logger import setup_logger

logger = setup_logger(__name__)

# Constants for HTTP Status Codes
HTTP_429_TOO_MANY_REQUESTS = 429


class LLM:
    """LLM wrapper for Vertex AI summarization."""

    def __init__(self, project: str, location: str, sa_key: str, model: str) -> None:
        """Initialize the LLM client with Vertex AI."""
        logger.info(f"Initializing Vertex LLM ({model})...")

        # Authentication using service account key file
        credentials = service_account.Credentials.from_service_account_file(
            sa_key, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        self.client = genai.Client(
            vertexai=True, project=project, location=location, credentials=credentials
        )
        self.model_id = model

    def summarize(self, doc_content: str) -> tuple[str, str]:
        """Summarize the conversation text using Vertex AI. Returns (markdown, html)."""
        logger.info("Summarizing conversation text...")

        prompt = """
You are an expert executive assistant, capable of summarizing lengthy
work meeting conversations clearly and extracting key business value.

<INSTRUCTIONS>
1. The summarization must be strictly grounded to the provided document.
2. Use the exact markdown format below when responding.
3. Follow the order of the sections and details as requested.
4. If you are unsure about any detail or if a section is not applicable,
do not hallucinate; instead, state 'Not discussed' or omit the point.
</INSTRUCTIONS>

<SECTION STRUCTURE>
1. 'TL;DR' section: 1 bullet point summarizing what the meeting discussion is about.
2. 'Executive Summary' section: 3 to 5 bullet points providing the
highest-level overview of the meeting's purpose and outcome.
3. 'Key Decisions & Discussion Points' section: Detailed bullet points
capturing major topics, debates, context, and final decisions.
4. 'Action Items' section: Bullet points outlining specific tasks,
clearly stating the assignee (if mentioned) and deadline (if mentioned).
</SECTION STRUCTURE>

<MARKDOWN FORMAT>
# TL;DR

* [TEXT]

# Executive Summary

* [TEXT]
* [TEXT]

# Key Decisions & Discussion Points

## [TOPIC]

* **[SHORT SUMMARY]:** [TEXT]
* **[SHORT SUMMARY]:** [TEXT]

# Action Items

* [TEXT]
* [TEXT]
</MARKDOWN FORMAT>

<MARKDOWN RULES>
1. [SUBTOPIC] must be in bold.
2. [TEXT] must NOT be in bold.
</MARKDOWN RULES>
        """

        max_retries = 4
        base_delay = 5  # Initial wait time for 429 errors

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[prompt, doc_content],
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=8192,
                        temperature=0.3,
                        top_p=0.95,
                    ),
                )

                if not response.text:
                    err_msg = "Empty response from Vertex AI during summarization."
                    raise RuntimeError(err_msg)

                html_content = markdown(response.text)
                return response.text, html_content

            except errors.APIError as e:
                # 429 Too Many Requests
                if e.code == HTTP_429_TOO_MANY_REQUESTS and attempt < max_retries - 1:
                    sleep_time = base_delay * (2**attempt)
                    logger.warning(
                        f"Vertex AI rate limit exceeded (429). Retrying in "
                        f"{sleep_time}s (Attempt {attempt + 1}/{max_retries - 1})..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error("Failed to summarize text due to an API error.")
                    raise

        raise RuntimeError("Failed to summarize text after multiple retries.")
