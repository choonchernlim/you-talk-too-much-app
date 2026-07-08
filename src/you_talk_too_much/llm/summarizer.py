from google import genai
from google.genai import errors
from google.oauth2 import service_account
from pydantic import BaseModel

from you_talk_too_much.cli.logger import setup_logger
from you_talk_too_much.common.retry import retry_with_backoff
from you_talk_too_much.llm.prompts import EXTRACTION_PROMPT, FORMAT_PROMPT

logger = setup_logger(__name__)

HTTP_429_TOO_MANY_REQUESTS = 429
MAX_RETRIES = 4
BASE_RETRY_DELAY = 5


class SummaryOutput(BaseModel):
    """Structured output of the combined format + topic call."""

    summary_markdown: str
    topic: str


class LLM:
    """LLM wrapper for Vertex AI summarization."""

    def __init__(self, project: str, location: str, sa_key: str, model: str) -> None:
        """Initialize the LLM client with Vertex AI."""
        logger.info(f"Initializing Vertex LLM ({model})...")

        credentials = service_account.Credentials.from_service_account_file(
            sa_key, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        self.client = genai.Client(
            vertexai=True, project=project, location=location, credentials=credentials
        )
        self.model_id = model

    def _generate(
        self,
        prompt: str,
        content: str,
        response_schema: type[BaseModel] | None = None,
    ) -> genai.types.GenerateContentResponse:
        """Call Vertex AI, retrying on 429 rate-limit errors.

        Returns the full response; pass `response_schema` for structured
        (JSON) output, available via `response.parsed`.
        """
        config = genai.types.GenerateContentConfig(
            temperature=0.3,
            top_p=0.95,
            response_mime_type="application/json" if response_schema else None,
            response_schema=response_schema,
        )

        def _call() -> genai.types.GenerateContentResponse:
            response = self.client.models.generate_content(
                model=self.model_id, contents=[prompt, content], config=config
            )
            if not response.text:
                raise RuntimeError("Empty response from Vertex AI.")
            return response

        def _on_retry(_exc: Exception, attempt: int, sleep_time: float) -> None:
            logger.warning(
                f"Vertex AI rate limit exceeded (429). Retrying in "
                f"{sleep_time}s (Attempt {attempt}/{MAX_RETRIES - 1})..."
            )

        try:
            return retry_with_backoff(
                fn=_call,
                retryable_exceptions=(errors.APIError,),
                max_retries=MAX_RETRIES,
                base_delay=BASE_RETRY_DELAY,
                should_retry=lambda e: (
                    isinstance(e, errors.APIError)
                    and e.code == HTTP_429_TOO_MANY_REQUESTS
                ),
                on_retry=_on_retry,
            )
        except errors.APIError:
            logger.error("Failed due to an API error.")
            raise

    def _extract(self, doc_content: str) -> str:
        """Extract all discussion points and decisions from the transcript.

        Returns raw notes.
        """
        logger.info("Extracting discussion details from transcript...")
        return self._generate(EXTRACTION_PROMPT, doc_content).text or ""

    def _format_with_topic(self, extracted: str) -> tuple[str, str]:
        """Format extracted notes into (markdown summary, topic) in one call."""
        logger.info("Formatting summary and extracting topic...")
        response = self._generate(
            FORMAT_PROMPT, extracted, response_schema=SummaryOutput
        )
        parsed = response.parsed
        if not isinstance(parsed, SummaryOutput):
            raise RuntimeError("Unexpected structured response from Vertex AI.")
        topic = " ".join(parsed.topic.split())
        return parsed.summary_markdown, topic

    def summarize(self, doc_content: str) -> tuple[str, str]:
        """Summarize the conversation text using Vertex AI.

        Returns (markdown, topic) where topic is a 3-5 keyword label.
        """
        logger.info("Summarizing conversation text...")
        extracted = self._extract(doc_content)
        return self._format_with_topic(extracted)
