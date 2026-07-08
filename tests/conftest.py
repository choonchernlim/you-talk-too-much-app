import os

# Importing you_talk_too_much.config fails fast when required settings are
# missing (e.g. in CI, where there is no .env file). Provide dummy values
# before any test module imports the app; real env vars and .env still take
# precedence locally.
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
