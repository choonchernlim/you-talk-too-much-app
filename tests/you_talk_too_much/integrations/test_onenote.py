from unittest.mock import MagicMock, patch

import pytest
import requests

from you_talk_too_much.integrations.onenote import _CACHE_FILE, OneNoteClient


def _make_client(tmp_path=None):
    """Create a OneNoteClient with all MSAL and cache I/O mocked out."""
    cache_patch = patch(
        "you_talk_too_much.integrations.onenote.msal.SerializableTokenCache"
    )
    app_patch = patch(
        "you_talk_too_much.integrations.onenote.msal.PublicClientApplication"
    )
    cache_file_patch = patch(
        "you_talk_too_much.integrations.onenote._CACHE_FILE",
        tmp_path / "msal_token_cache.bin" if tmp_path else _CACHE_FILE,
    )
    return cache_patch, app_patch, cache_file_patch


@pytest.fixture
def client():
    cache_patch, app_patch, cache_file_patch = _make_client()
    with (
        cache_patch,
        app_patch,
        cache_file_patch,
        patch.object(
            OneNoteClient,
            "get_headers",
            return_value={"Authorization": "Bearer fake-token"},
        ),
    ):
        yield OneNoteClient(
            onenote_section_name="Test Section",
            az_client_id="fake-client-id",
            az_tenant_id="fake-tenant-id",
        )


def _mock_section_response(section_id: str = "section-123") -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {
        "value": [{"displayName": "Test Section", "id": section_id}]
    }
    return resp


def test_create_page_succeeds_on_first_attempt(client):
    section_resp = _mock_section_response()
    post_resp = MagicMock()

    with (
        patch(
            "you_talk_too_much.integrations.onenote.requests.get",
            return_value=section_resp,
        ),
        patch(
            "you_talk_too_much.integrations.onenote.requests.post",
            return_value=post_resp,
        ),
        patch("you_talk_too_much.integrations.onenote.time.sleep") as mock_sleep,
    ):
        client.create_page("Title", "<p>body</p>")

    post_resp.raise_for_status.assert_called_once()
    mock_sleep.assert_not_called()


def test_create_page_retries_once_on_read_timeout_then_succeeds(client):
    section_resp = _mock_section_response()
    post_resp = MagicMock()

    call_count = {"n": 0}

    def post_side_effect(*_args, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise requests.exceptions.ReadTimeout
        return post_resp

    with (
        patch(
            "you_talk_too_much.integrations.onenote.requests.get",
            return_value=section_resp,
        ),
        patch(
            "you_talk_too_much.integrations.onenote.requests.post",
            side_effect=post_side_effect,
        ),
        patch("you_talk_too_much.integrations.onenote.time.sleep") as mock_sleep,
    ):
        client.create_page("Title", "<p>body</p>")

    assert call_count["n"] == 2
    mock_sleep.assert_called_once_with(5)


def test_create_page_raises_after_exhausting_all_retries(client):
    section_resp = _mock_section_response()

    with (
        patch(
            "you_talk_too_much.integrations.onenote.requests.get",
            return_value=section_resp,
        ),
        patch(
            "you_talk_too_much.integrations.onenote.requests.post",
            side_effect=requests.exceptions.ReadTimeout,
        ),
        patch("you_talk_too_much.integrations.onenote.time.sleep"),
        pytest.raises(requests.exceptions.ReadTimeout),
    ):
        client.create_page("Title", "<p>body</p>")


def test_create_page_does_not_retry_on_http_error(client):
    section_resp = _mock_section_response()
    post_resp = MagicMock()
    post_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "403 Forbidden"
    )

    with (
        patch(
            "you_talk_too_much.integrations.onenote.requests.get",
            return_value=section_resp,
        ),
        patch(
            "you_talk_too_much.integrations.onenote.requests.post",
            return_value=post_resp,
        ),
        patch("you_talk_too_much.integrations.onenote.time.sleep") as mock_sleep,
        pytest.raises(requests.exceptions.HTTPError),
    ):
        client.create_page("Title", "<p>body</p>")

    mock_sleep.assert_not_called()


# --- Token cache tests ---


def _make_auth_client(tmp_path, *, silent_result, interactive_result=None):
    """Build a OneNoteClient with a tmp cache file but mocked MSAL app."""
    mock_app = MagicMock()
    mock_app.get_accounts.return_value = (
        [MagicMock()] if silent_result is not None else []
    )
    mock_app.acquire_token_silent.return_value = silent_result
    if interactive_result is not None:
        mock_app.acquire_token_interactive.return_value = interactive_result

    cache_file = tmp_path / "msal_token_cache.bin"

    with (
        patch("you_talk_too_much.integrations.onenote.msal.SerializableTokenCache"),
        patch(
            "you_talk_too_much.integrations.onenote.msal.PublicClientApplication",
            return_value=mock_app,
        ),
        patch(
            "you_talk_too_much.integrations.onenote._CACHE_FILE",
            cache_file,
        ),
    ):
        c = OneNoteClient("S", "cid", "tid")

    c.app = mock_app
    return c, cache_file


def test_get_access_token_uses_silent_when_accounts_cached(tmp_path):
    silent_result = {"access_token": "tok-silent"}
    client, _ = _make_auth_client(tmp_path, silent_result=silent_result)

    acquired = client._get_access_token()

    client.app.acquire_token_silent.assert_called_once()
    client.app.acquire_token_interactive.assert_not_called()
    assert acquired == "tok-silent"


def test_get_access_token_falls_back_to_interactive_when_silent_fails(tmp_path):
    interactive_result = {"access_token": "tok-interactive"}
    client, _ = _make_auth_client(
        tmp_path, silent_result=None, interactive_result=interactive_result
    )
    client._cache = MagicMock()
    client._cache.has_state_changed = True
    client._cache.serialize.return_value = "{}"

    acquired = client._get_access_token()

    client.app.acquire_token_interactive.assert_called_once()
    assert acquired == "tok-interactive"


def test_save_cache_writes_file_when_state_changed(tmp_path):
    cache_file = tmp_path / "msal_token_cache.bin"
    client, _ = _make_auth_client(tmp_path, silent_result=None)
    client._cache = MagicMock()
    client._cache.has_state_changed = True
    client._cache.serialize.return_value = '{"tokens": "data"}'

    with patch("you_talk_too_much.integrations.onenote._CACHE_FILE", cache_file):
        client._save_cache()

    assert cache_file.exists()
    assert cache_file.read_text() == '{"tokens": "data"}'


def test_save_cache_skips_write_when_state_unchanged(tmp_path):
    cache_file = tmp_path / "msal_token_cache.bin"
    client, _ = _make_auth_client(tmp_path, silent_result=None)
    client._cache = MagicMock()
    client._cache.has_state_changed = False

    with patch("you_talk_too_much.integrations.onenote._CACHE_FILE", cache_file):
        client._save_cache()

    assert not cache_file.exists()
