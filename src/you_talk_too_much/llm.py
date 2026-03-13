from google import genai
from google.oauth2 import service_account
from markdown import markdown

from you_talk_too_much.log_config import setup_logger
from you_talk_too_much.utils import read_file, write_file

logger = setup_logger(__name__)


class LLM:
    """LLM wrapper for Vertex AI summarization."""

    def __init__(self, project: str, location: str, sa_key: str, model: str) -> None:
        """Initialize the LLM client with Vertex AI."""
        logger.info("Initializing Vertex LLM (using new google-genai SDK)...")

        # Authentication using service account key file
        credentials = service_account.Credentials.from_service_account_file(
            sa_key, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        self.client = genai.Client(
            vertexai=True, project=project, location=location, credentials=credentials
        )
        self.model_id = model

    def summarize(self, conversation_file_path: str) -> str:
        """Summarize the conversation text using Vertex AI."""
        logger.info(f"Summarizing conversation [File: {conversation_file_path}] ...")

        prompt = """
        You are a helpful agent, capable in summarizing lengthy conversations clearly.

        <INSTRUCTIONS>
        1. The summarization must be grounded to the provided document.
        2. Use the exact markdown format below when responding.
        3. Follow the order of the sections and provide the details as requested.
        4. If you are unsure about the content, please do not make up any information.
        </INSTRUCTIONS>

        <SECTION STRUCTURE>
        1. "Executive Summary" section: 3 to 5 bullet points for key details.
        2. "Detailed Summary" section: Details and key decisions in bullet points.
        3. "Action Items" section: 3 to 5 bullet points on actions to be taken.
        </SECTION STRUCTURE>

        <MARKDOWN FORMAT>
        # Executive Summary

        * [TEXT]
        * [TEXT]

        # Detailed Summary

        ## [TOPIC]

        * **[SHORT SUMMARY]:** [TEXT]
        * **[SHORT SUMMARY]:** [TEXT]

        # Action Items

        * [TEXT]
        * [TEXT]
        </MARKDOWN FORMAT>

        <MARKDOWN RULES>
        1. [SUBTOPIC] must be in bold.
        2. [TEXT] must NOT in bold.
        </MARKDOWN RULES>
        """

        doc_content = read_file(conversation_file_path)

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt, doc_content],
            config={
                "max_output_tokens": 8192,
                "temperature": 1,
                "top_p": 0.95,
            },
        )

        if not response.text:
            logger.error("Empty response from LLM.")
            return ""

        html_content = markdown(response.text)

        # remove file extension
        filename = conversation_file_path.rsplit(".", 1)[0]

        write_file(f"{filename}.md", response.text)
        write_file(f"{filename}.html", html_content)

        return html_content
