import os

import msal
import requests
# use dotenv to load environment variables
from dotenv import load_dotenv
from msal_extensions import build_encrypted_persistence, PersistedTokenCache

load_dotenv()

AZURE_CLIENT_ID = os.getenv('AZURE_CLIENT_ID')
AZURE_TENANT_ID = os.getenv('AZURE_TENANT_ID')
AUTHORITY = f'https://login.microsoftonline.com/{AZURE_TENANT_ID}'
SCOPES = ['Notes.Read', 'User.Read']

os.environ['BROWSER'] = 'open -a /Applications/Google\\ Chrome.app %s'


def get_access_token() -> str:
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

    access_token = result['access_token']

    return access_token


def get_onenote_pages(access_token: str) -> dict:
    url = 'https://graph.microsoft.com/v1.0/me/onenote/pages'

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return response.json()


def run() -> None:
    access_token = get_access_token()
    print(access_token)

    onenote_pages = get_onenote_pages(access_token)
    print(onenote_pages)


if __name__ == '__main__':
    run()
