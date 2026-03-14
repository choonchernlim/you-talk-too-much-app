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


# Global settings instance
settings = Settings()  # type: ignore
