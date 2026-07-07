from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via EDGAR_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="EDGAR_", env_file=".env", extra="ignore")

    # SEC requires a descriptive User-Agent with contact info on every request.
    sec_user_agent: str = "EDGAR-Analyst research tool contact@example.com"
    sec_base_url: str = "https://www.sec.gov"
    sec_data_url: str = "https://data.sec.gov"

    # LLM routing: cheap model for classification, frontier model for synthesis.
    # "stub" runs the deterministic offline client (tests, CI, no-key demos).
    llm_provider: str = "auto"  # auto | anthropic | stub
    router_model: str = "claude-haiku-4-5"
    synthesis_model: str = "claude-opus-4-8"

    # Retrieval
    chunk_size_chars: int = 1600
    chunk_overlap_chars: int = 200
    retrieval_top_k: int = 8
    rrf_k: int = 60

    # Storage: sqlite path for the default store; DSN switches to Postgres/pgvector.
    db_path: str = "edgar_analyst.db"
    postgres_dsn: str = ""

    # Eval gates (CI fails below these)
    eval_min_recall_at_5: float = 0.80
    eval_min_mrr: float = 0.60
    eval_min_citation_precision: float = 0.90


@lru_cache
def get_settings() -> Settings:
    return Settings()
