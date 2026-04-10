import time

from google import genai
from google.genai import errors
from google.oauth2 import service_account
from markdown import markdown

from you_talk_too_much.cli.logger import setup_logger
from you_talk_too_much.llm.prompts import EXTRACTION_PROMPT, FORMAT_PROMPT, TOPIC_PROMPT

logger = setup_logger(__name__)

# Constants for HTTP Status Codes
HTTP_429_TOO_MANY_REQUESTS = 429

MAX_RETRIES = 4
BASE_RETRY_DELAY = 5


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

    def _generate(self, prompt: str, content: str) -> str:
        """Call Vertex AI with retry on rate limiting. Returns response text."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[prompt, content],
                    config=genai.types.GenerateContentConfig(
                        temperature=0.3,
                        top_p=0.95,
                    ),
                )
                if not response.text:
                    raise RuntimeError("Empty response from Vertex AI.")
                return response.text
            except errors.APIError as e:
                if e.code == HTTP_429_TOO_MANY_REQUESTS and attempt < MAX_RETRIES - 1:
                    sleep_time = BASE_RETRY_DELAY * (2**attempt)
                    logger.warning(
                        f"Vertex AI rate limit exceeded (429). Retrying in "
                        f"{sleep_time}s (Attempt {attempt + 1}/{MAX_RETRIES - 1})..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error("Failed due to an API error.")
                    raise
        raise RuntimeError("Failed to generate content after retries.")

    def _format(self, extracted: str) -> tuple[str, str]:
        """Format extracted notes into the final markdown + HTML summary."""
        logger.info("Formatting extracted notes into summary...")
        md = self._generate(FORMAT_PROMPT, extracted)
        return md, markdown(md)

    def _extract_topic(self, summary: str) -> str:
        """Extract a 2-3 word topic label from the markdown summary."""
        logger.info("Extracting topic label from summary...")
        raw = self._generate(TOPIC_PROMPT, summary).strip()
        return " ".join(raw.split())

    def _extract(self, doc_content: str) -> str:
        """Extract all discussion points and decisions from the transcript.

        Returns raw notes.
        """
        logger.info("Extracting discussion details from transcript...")
        return self._generate(EXTRACTION_PROMPT, doc_content)

    def summarize(self, doc_content: str) -> tuple[str, str, str]:
        """Summarize the conversation text using Vertex AI.

        Returns (markdown, html, topic) where topic is a 2-3 word label.
        """
        logger.info("Summarizing conversation text...")
        extracted = self._extract(doc_content)
        md, html = self._format(extracted)
        topic = self._extract_topic(md)
        return md, html, topic
