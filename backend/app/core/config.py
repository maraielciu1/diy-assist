from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DIY-Assist API"
    environment: str = "dev"
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "manual_chunks"
    embed_model_name: str = "BAAI/bge-small-en-v1.5"
    top_k_default: int = 5
    previous_steps_window: int = 2
    ifixit_api_base_url: str = "https://www.ifixit.com/api/2.0"
    guardrails_enabled: bool = True
    raw_data_dir: str = "./data/raw"
    slm_model_name: str = "qwen2.5-3b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_timeout_seconds: int = 60
    use_ifixit_live_lookup: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
