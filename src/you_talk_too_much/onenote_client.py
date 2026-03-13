import os

import msal
import requests
from msal_extensions import build_encrypted_persistence, PersistedTokenCache

from you_talk_too_much.log_config import setup_logger

logger = setup_logger(__name__)


class OneNoteClient:
    def __init__(self, onenote_section_name, az_client_id, az_tenant_id) -> None:
        logger.info('Initializing OneNote Client...')

        self.az_client_id = az_client_id
        self.az_tenant_id = az_tenant_id

        # Set Chrome browser for interactive authentication
        os.environ['BROWSER'] = 'open -a /Applications/Google\\ Chrome.app %s'

        self.section_id = self.get_section_id_by_name(onenote_section_name)

    def get_headers(self) -> dict:
        # always get a fresh access token to prevent it from expiration if it was fetched too soon
        return {
            'Authorization': f'Bearer {self._get_access_token()}',
            'Accept': 'application/json',
            'Content-Type': 'application/xhtml+xml'
        }

    def _get_access_token(self) -> str:
        app = msal.PublicClientApplication(
            self.az_client_id,
            authority=f'https://login.microsoftonline.com/{self.az_tenant_id}',
            token_cache=PersistedTokenCache(build_encrypted_persistence('.msal_token_cache.json')),
        )

        scopes = ['Notes.Read', 'User.Read']
        result = None
        accounts = app.get_accounts()

        if accounts:
            result = app.acquire_token_silent(scopes, account=accounts[0])

        if not result:
            result = app.acquire_token_interactive(scopes=scopes)

        return result['access_token']

    def get_pages(self, page_id='') -> dict:
        url = f'https://graph.microsoft.com/v1.0/me/onenote/pages/{page_id}'

        response = requests.get(url, headers=self.get_headers())
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

        response = requests.post(url, headers=self.get_headers(), data=body)
        response.raise_for_status()

    def get_section_id_by_name(self, name: str) -> str:
        url = 'https://graph.microsoft.com/v1.0/me/onenote/sections'

        response = requests.get(url, headers=self.get_headers())
        response.raise_for_status()

        sections = response.json()['value']

        section_id = next((s['id'] for s in sections if s['displayName'] == name), None)

        assert section_id is not None, f'Section [name: {name}] not found'

        return section_id
