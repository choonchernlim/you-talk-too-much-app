import time

from google import genai
from google.genai import errors
from google.oauth2 import service_account
from markdown import markdown

from you_talk_too_much.cli.logger import setup_logger

logger = setup_logger(__name__)

# Constants for HTTP Status Codes
HTTP_429_TOO_MANY_REQUESTS = 429

MAX_OUTPUT_TOKENS = 32_768
MAX_RETRIES = 4
BASE_RETRY_DELAY = 5

_EXTRACTION_PROMPT = """\
You are a meticulous meeting analyst reviewing a transcript.

Your task is to extract ALL discussion content — every topic discussed, every position
raised, every concern voiced, every tradeoff weighed, and every decision made (including
items that were not resolved).

For each topic:
1. What prompted the discussion (context)
2. Every significant point raised — include ALL positions, not just the final conclusion
3. Tradeoffs, concerns, objections, and alternatives considered
4. The final decision or outcome (or "Not reached" if none)
5. The explicit reasoning or rationale behind any decision

Critical: Do NOT compress or summarise. If something was discussed at length, capture it
at length. Organise by topic.

<FORMAT>
## [TOPIC NAME]

**Context:** [what prompted this topic]

**Discussion:**
- [each significant point raised]

**Tradeoffs / Concerns:**
- [each concern, objection, or alternative]

**Decision:** [outcome, or "Not reached"]

**Rationale:** [reasoning behind the decision]
</FORMAT>
"""


_FORMAT_PROMPT = """\
You are an expert executive assistant.

Based on the detailed meeting notes provided, produce a structured summary.

<INSTRUCTIONS>
1. The summary must be strictly grounded in the provided notes.
2. Use the exact markdown format below.
3. For Key Decisions & Discussion Points, use concise labels of your own choosing.
   Each bullet must cover multiple related points in rich text — do not create a
   separate bullet for each micro-point. Capture context, options evaluated, tradeoffs,
   decision, and
   rationale within as few bullets as practical per topic.
4. Do not compress or omit detail from the notes.
5. If a section is not applicable, state 'Not discussed'.
</INSTRUCTIONS>

<MARKDOWN FORMAT>
# TL;DR

* [TEXT]

# Executive Summary

* [TEXT]
* [TEXT]

# Key Decisions & Discussion Points

## [TOPIC]

* **[SHORT LABEL]:** [TEXT]
* **[SHORT LABEL]:** [TEXT]

# Action Items

* [TEXT]
* [TEXT]
</MARKDOWN FORMAT>

<MARKDOWN RULES>
1. [SHORT LABEL] must be in bold.
2. [TEXT] must NOT be in bold.
</MARKDOWN RULES>
"""


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

    def _generate(self, prompt: str, content: str, max_output_tokens: int) -> str:
        """Call Vertex AI with retry on rate limiting. Returns response text."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[prompt, content],
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=max_output_tokens,
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
        md = self._generate(_FORMAT_PROMPT, extracted, MAX_OUTPUT_TOKENS)
        return md, markdown(md)

    def _extract(self, doc_content: str) -> str:
        """Extract all discussion points and decisions from the transcript.

        Returns raw notes.
        """
        logger.info("Extracting discussion details from transcript...")
        return self._generate(_EXTRACTION_PROMPT, doc_content, MAX_OUTPUT_TOKENS)

    def summarize(self, doc_content: str) -> tuple[str, str]:
        """Summarize the conversation text using Vertex AI. Returns (markdown, html)."""
        logger.info("Summarizing conversation text...")
        extracted = self._extract(doc_content)
        return self._format(extracted)
