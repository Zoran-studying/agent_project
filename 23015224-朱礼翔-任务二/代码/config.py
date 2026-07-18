"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(slots=True)
class Settings:
    """Runtime settings. Secrets are never given a source-code default."""

    base_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_BASE_URL", "https://api_2604_w5t3.zlth.cn/v1"
        )
    )
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "qwen3.6-35b-a3b")
    )
    verify_ssl: bool = field(
        default_factory=lambda: os.getenv("OPENAI_VERIFY_SSL", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    enable_thinking: bool = field(
        default_factory=lambda: os.getenv("OPENAI_ENABLE_THINKING", "false").lower()
        in {"1", "true", "yes", "on"}
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("OPENAI_MAX_TOKENS", "3000"))
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "embedding-3"
        )
    )
    embedding_base_url: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_BASE_URL",
            "https://open.bigmodel.cn/api/paas/v4/embeddings",
        )
    )
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")
        )
    )
    embedding_dimensions: int | None = field(
        default_factory=lambda: (
            int(value) if (value := os.getenv("EMBEDDING_DIMENSIONS", "")) else None
        )
    )
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "5")))
    score_threshold: float = field(
        default_factory=lambda: float(os.getenv("RAG_SCORE_THRESHOLD", "0.45"))
    )
    chunk_size: int = 700
    chunk_overlap: int = 120
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"

    @property
    def materials_dir(self) -> Path:
        return self.data_dir / "materials"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "vector_store"

    @property
    def history_db(self) -> Path:
        return self.data_dir / "chat_history.sqlite3"

    @property
    def embedding_client_base_url(self) -> str:
        return self.embedding_base_url.removesuffix("/embeddings")

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.materials_dir,
            self.vector_dir,
            self.output_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def validate_api(self) -> None:
        if not self.api_key:
            raise ValueError("未配置 OPENAI_API_KEY，请在环境变量或设置窗口中填写。")

    def validate_embedding_api(self) -> None:
        if not self.embedding_api_key:
            raise ValueError(
                "未配置 EMBEDDING_API_KEY 或 OPENAI_API_KEY，请在 .env 中填写。"
            )


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
