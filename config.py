import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
EVIDENCE_DIR = ROOT_DIR / "evidence"
DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
DEFAULT_OPENAI_BASE_URL = "https://luongchidung.online/v1"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_RAGAS_MODEL = DEFAULT_OPENAI_MODEL


@dataclass(frozen=True)
class AppConfig:
    langsmith_tracing_v2: str
    langsmith_api_key: str
    langsmith_project: str
    langsmith_endpoint: str
    openai_api_key: str
    openai_model: str
    openai_base_url: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    ragas_llm_api_key: str
    ragas_llm_model: str
    ragas_embeddings_model: str
    ragas_base_url: str
    embedding_mode: str
    debug: bool
    log_level: str


def _require(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_config() -> AppConfig:
    load_dotenv(ROOT_DIR / ".env")

    langsmith_endpoint = (
        os.getenv("LANGCHAIN_ENDPOINT")
        or os.getenv("LANGSMITH_ENDPOINT")
        or DEFAULT_LANGSMITH_ENDPOINT
    )
    openai_base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_ENDPOINT")
        or DEFAULT_OPENAI_BASE_URL
    )
    ragas_base_url = (
        os.getenv("RAGAS_LLM_ENDPOINT")
        or os.getenv("RAGAS_BASE_URL")
        or openai_base_url
    )
    embedding_mode = os.getenv("EMBEDDING_MODE", "remote").lower()
    embedding_base_url = (
        os.getenv("EMBEDDING_BASE_URL")
        or os.getenv("OPENAI_EMBEDDING_BASE_URL")
        or openai_base_url
    )
    embedding_api_key = (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("OPENAI_EMBEDDING_API_KEY")
        or _require("OPENAI_API_KEY")
    )

    config = AppConfig(
        langsmith_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "true"),
        langsmith_api_key=_require("LANGCHAIN_API_KEY"),
        langsmith_project=_require("LANGCHAIN_PROJECT", "day22-langsmith-lab"),
        langsmith_endpoint=langsmith_endpoint,
        openai_api_key=_require("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        openai_base_url=openai_base_url,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        ragas_llm_api_key=os.getenv("RAGAS_LLM_API_KEY") or _require("OPENAI_API_KEY"),
        ragas_llm_model=os.getenv("RAGAS_LLM_MODEL", os.getenv("OPENAI_MODEL", DEFAULT_RAGAS_MODEL)),
        ragas_embeddings_model=os.getenv("RAGAS_EMBEDDINGS_MODEL", DEFAULT_EMBEDDING_MODEL),
        ragas_base_url=ragas_base_url,
        embedding_mode=embedding_mode,
        debug=os.getenv("DEBUG", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    os.environ["LANGCHAIN_TRACING_V2"] = config.langsmith_tracing_v2
    os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = config.langsmith_endpoint
    os.environ["OPENAI_API_KEY"] = config.openai_api_key

    DATA_DIR.mkdir(exist_ok=True)
    EVIDENCE_DIR.mkdir(exist_ok=True)
    return config


def print_config_summary(config: AppConfig) -> None:
    print("Config loaded successfully")
    print(f"  LangSmith project : {config.langsmith_project}")
    print(f"  OpenAI endpoint   : {config.openai_base_url}")
    print(f"  Default LLM model : {config.openai_model}")
    print(f"  Embedding mode    : {config.embedding_mode}")
    print(f"  Embedding endpoint: {config.embedding_base_url}")
    print(f"  Embedding model   : {config.embedding_model}")


if __name__ == "__main__":
    cfg = load_config()
    print_config_summary(cfg)
