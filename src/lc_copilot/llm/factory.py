from __future__ import annotations

from ..config import Settings
from ..rag.embeddings import Embedder, HashingEmbedder, RemoteEmbedder
from .base import LLMClient
from .offline import OfflineLLM


def build_llm(s: Settings) -> tuple[LLMClient, Embedder]:
    if s.llm_provider == "offline":
        return OfflineLLM(), HashingEmbedder()
    from .openai_compat import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(s, azure=s.llm_provider == "azure_openai")
    return llm, RemoteEmbedder(llm)
