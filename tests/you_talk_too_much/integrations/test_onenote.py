from unittest.mock import MagicMock, patch

import pytest
import requests

from you_talk_too_much.integrations.onenote import OneNoteClient


@pytest.fixture
def client():
    with (
        patch("you_talk_too_much.integrations.onenote.msal.PublicClientApplication"),
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
