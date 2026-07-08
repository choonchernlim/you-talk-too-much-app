from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # GCP Vertex AI
    gcp_vertex_project: str
    gcp_vertex_location: str
    gcp_vertex_sa_key: str
    gcp_vertex_model: str

    # Microsoft / OneNote
    onenote_section_name: str
    azure_client_id: str
    azure_tenant_id: str

    # HuggingFace Models
    hf_whisper_model: str
    hf_diarization_model: str
    hf_embedding_model: str
    hf_token: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings, loaded on first use.

    Lazy so that importing app modules never fails on missing settings;
    validation happens when the settings are first needed.
    """
    return Settings()  # type: ignore[call-arg]
