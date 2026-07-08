import os

from dotenv import load_dotenv

# Importing you_talk_too_much.config fails fast when required settings are
# missing (e.g. in CI, where there is no .env file). Load .env into the
# environment first so real values win, then fill in dummy values for
# whatever is still missing; otherwise the dummies would shadow .env
# (env vars take precedence over dotenv in pydantic-settings) and break
# manual tests that need live credentials.
load_dotenv()
for _key in (
    "GCP_VERTEX_PROJECT",
    "GCP_VERTEX_LOCATION",
    "GCP_VERTEX_SA_KEY",
    "GCP_VERTEX_MODEL",
    "ONENOTE_SECTION_NAME",
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "HF_WHISPER_MODEL",
    "HF_DIARIZATION_MODEL",
    "HF_EMBEDDING_MODEL",
    "HF_TOKEN",
):
    os.environ.setdefault(_key, "test-value")
