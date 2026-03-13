import msal
import requests

from you_talk_too_much.log_config import setup_logger

logger = setup_logger(__name__)


class OneNoteClient:
    """OneNote client using Microsoft Graph API."""

    def __init__(
        self, onenote_section_name: str, az_client_id: str, az_tenant_id: str
    ) -> None:
        """Initialize the OneNote client."""
        logger.info("Initializing OneNote Client...")

        self.az_client_id = az_client_id
        self.az_tenant_id = az_tenant_id
        self.onenote_section_name = onenote_section_name

        self.scopes = ["Notes.ReadWrite.All"]
        self.authority = f"https://login.microsoftonline.com/{self.az_tenant_id}"

        # Initialize the MSAL public client
        self.app = msal.PublicClientApplication(
            self.az_client_id, authority=self.authority
        )

    def get_headers(self) -> dict:
        """Get headers with a fresh access token."""
        # Always get a fresh access token to prevent expiration
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "text/html",
        }

    def _get_access_token(self) -> str:
        """Fetch an access token using MSAL."""
        accounts = self.app.get_accounts()
        result = None

        if accounts:
            result = self.app.acquire_token_silent(self.scopes, account=accounts[0])

        if not result:
            result = self.app.acquire_token_interactive(scopes=self.scopes)

        if "access_token" not in result:
            raise Exception(f"Could not acquire access token: {result.get('error')}")

        return result["access_token"]

    def get_pages(self, page_id: str = "") -> dict:
        """Fetch OneNote pages or a specific page."""
        url = f"https://graph.microsoft.com/v1.0/me/onenote/pages/{page_id}"
        response = requests.get(url, headers=self.get_headers(), timeout=10)
        return response.json()

    def create_page(self, title: str, html_summary: str) -> None:
        """Create a new page in the specified OneNote section."""
        logger.info(f"Creating OneNote page [Title: {title}] ...")

        section_id = self._get_section_id()
        url = f"https://graph.microsoft.com/v1.0/me/onenote/sections/{section_id}/pages"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
        </head>
        <body>
            {html_summary}
        </body>
        </html>
        """

        response = requests.post(
            url, headers=self.get_headers(), data=html_content, timeout=10
        )

        # 201 is the status code for 'Created'
        if response.status_code == 201:  # noqa: PLR2004
            logger.info("OneNote page created successfully.")
        else:
            logger.error(f"Failed to create OneNote page: {response.text}")

    def _get_section_id(self) -> str:
        """Find the ID of the section with the specified name."""
        url = "https://graph.microsoft.com/v1.0/me/onenote/sections"
        response = requests.get(url, headers=self.get_headers(), timeout=10)
        sections = response.json().get("value", [])

        for section in sections:
            if section["displayName"] == self.onenote_section_name:
                return section["id"]

        raise Exception(f"Section '{self.onenote_section_name}' not found.")
