from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence, Tuple

DEFAULT_SOURCE_URLS: Tuple[str, ...] = (
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
)


@dataclass(frozen=True)
class AgenticRagSettings:
    source_urls: Sequence[str] = field(default_factory=lambda: DEFAULT_SOURCE_URLS)
    chunk_size: int = 100
    chunk_overlap: int = 50
    retrieval_k: int = 4
    chat_model: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview"
        )
    )
