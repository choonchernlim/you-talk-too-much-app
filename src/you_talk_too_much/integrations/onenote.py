import time
from pathlib import Path

import msal
import requests

from you_talk_too_much.cli.logger import setup_logger

logger = setup_logger(__name__)

_CACHE_FILE = Path.home() / ".you-talk-too-much" / "msal_token_cache.bin"

_TRANSIENT_ERRORS = (
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectionError,
)


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

        self._cache = msal.SerializableTokenCache()
        if _CACHE_FILE.exists():
            self._cache.deserialize(_CACHE_FILE.read_text())

        # Initialize the MSAL public client
        self.app = msal.PublicClientApplication(
            self.az_client_id, authority=self.authority, token_cache=self._cache
        )

    def _save_cache(self) -> None:
        """Persist the token cache to disk if it has changed."""
        if self._cache.has_state_changed:
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(self._cache.serialize())

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
            self._save_cache()

        if "access_token" not in result:
            raise Exception(f"Could not acquire access token: {result.get('error')}")

        return result["access_token"]

    def get_pages(self, page_id: str = "") -> dict:
        """Fetch OneNote pages or a specific page."""
        url = f"https://graph.microsoft.com/v1.0/me/onenote/pages/{page_id}"
        response = requests.get(url, headers=self.get_headers(), timeout=30)
        response.raise_for_status()
        return response.json()

    def create_page(self, title: str, html_summary: str) -> None:
        """Create a new page in the specified OneNote section."""
        logger.info(f"Creating OneNote page [Title: {title}] ...")

        max_retries = 3
        base_delay = 5

        for attempt in range(max_retries):
            try:
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
                    url, headers=self.get_headers(), data=html_content, timeout=30
                )
                response.raise_for_status()
                logger.info("OneNote page created successfully.")
                return

            except _TRANSIENT_ERRORS:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2**attempt)
                    logger.warning(
                        f"OneNote request timed out. Retrying in {sleep_time}s "
                        f"(Attempt {attempt + 1}/{max_retries - 1})..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        "Failed to create OneNote page after multiple retries."
                    )
                    raise

    def _get_section_id(self) -> str:
        """Find the ID of the section with the specified name."""
        url = "https://graph.microsoft.com/v1.0/me/onenote/sections"
        response = requests.get(url, headers=self.get_headers(), timeout=30)
        response.raise_for_status()
        sections = response.json().get("value", [])

        for section in sections:
            if section["displayName"] == self.onenote_section_name:
                return section["id"]

        raise Exception(f"Section '{self.onenote_section_name}' not found.")
