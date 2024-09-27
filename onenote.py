import logging
import os

import msal
import requests
from dotenv import load_dotenv
from msal_extensions import build_encrypted_persistence, PersistedTokenCache

# Create a logger for this module
logger = logging.getLogger(__name__)

load_dotenv()

ONENOTE_SECTION_NAME = os.getenv('ONENOTE_SECTION_NAME')
AZURE_CLIENT_ID = os.getenv('AZURE_CLIENT_ID')
AZURE_TENANT_ID = os.getenv('AZURE_TENANT_ID')
AUTHORITY = f'https://login.microsoftonline.com/{AZURE_TENANT_ID}'
SCOPES = ['Notes.Read', 'User.Read']

os.environ['BROWSER'] = 'open -a /Applications/Google\\ Chrome.app %s'


class OneNote:
    def __init__(self) -> None:
        self.headers = {
            'Authorization': f'Bearer {self._get_access_token()}',
            'Accept': 'application/json',
            'Content-Type': 'application/xhtml+xml'
        }

        self.section_id = self.get_section_id_by_name(ONENOTE_SECTION_NAME)

    @staticmethod
    def _get_access_token() -> str:
        app = msal.PublicClientApplication(
            AZURE_CLIENT_ID,
            authority=AUTHORITY,
            token_cache=PersistedTokenCache(build_encrypted_persistence('.msal_token_cache.json')),
        )

        result = None
        accounts = app.get_accounts()

        if accounts:
            result = app.acquire_token_silent(SCOPES, account=accounts[0])

        if not result:
            result = app.acquire_token_interactive(scopes=SCOPES)

        return result['access_token']

    def get_pages(self, page_id='') -> dict:
        url = f'https://graph.microsoft.com/v1.0/me/onenote/pages/{page_id}'

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def create_page(self, title: str, html_summary: str) -> None:
        logger.info(f'Creating OneNote page [Title: {title}] ...')

        url = f'https://graph.microsoft.com/v1.0/me/onenote/sections/{self.section_id}/pages'

        body = f"""
        <!DOCTYPE html>
        <html htmlns="https://www.w3.org/1999/xhtml" lang="en-US">
            <head>
                <title>{title}</title>
            </head>
            <body style="font-family:Calibri;font-size:12pt">
                <h1>Topics</h1>
                <ul>
                    <li>TBD</li>
                </ul>

                {html_summary.replace(f'<h1>', f'<br/><h1>')}
            </body>
        </html>
        """

        response = requests.post(url, headers=self.headers, data=body)
        response.raise_for_status()

    def get_section_id_by_name(self, name: str) -> str:
        url = 'https://graph.microsoft.com/v1.0/me/onenote/sections'

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        sections = response.json()['value']

        section_id = next((s['id'] for s in sections if s['displayName'] == name), None)

        assert section_id is not None, f'Section [name: {name}] not found'

        return section_id
