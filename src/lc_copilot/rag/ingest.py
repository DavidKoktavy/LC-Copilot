"""Document ingestion: files → (Azure Document Intelligence for PDFs/images) → Markdown → chunks → index."""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

import httpx

from ..config import Settings
from ..resilience import PermanentError, RetryPolicy, TransientError, call_with_retry
from .chunking import Chunk, chunk_markdown
from .store import VectorStore

log = logging.getLogger("lc_copilot.ingest")
TEXT_EXT = {".md", ".txt"}
DI_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".docx"}


async def document_intelligence_markdown(path: Path, s: Settings) -> str:
    """Azure AI Document Intelligence ``prebuilt-layout`` → Markdown (keeps headings and tables).

    Async long-running operation: POST analyze → 202 + Operation-Location → poll until done.
    """
    url = (
        f"{s.azure_docintel_endpoint.rstrip('/')}/documentintelligence/documentModels/prebuilt-layout:analyze"
        "?api-version=2024-11-30&outputContentFormat=markdown"
    )
    headers = {"Ocp-Apim-Subscription-Key": s.azure_docintel_api_key}
    body = {"base64Source": base64.b64encode(path.read_bytes()).decode()}

    async with httpx.AsyncClient(timeout=30) as c:
        async def submit() -> str:
            r = await c.post(url, headers=headers, json=body)
            if r.status_code == 429 or r.status_code >= 500:
                raise TransientError(f"docintel {r.status_code}")
            if r.status_code != 202:
                raise PermanentError(f"docintel {r.status_code}: {r.text[:300]}")
            return r.headers["Operation-Location"]

        op = await call_with_retry(submit, RetryPolicy(attempts=4, timeout_s=30))
        for _ in range(120):  # ~2 min ceiling
            r = await c.get(op, headers=headers)
            data = r.json()
            if data.get("status") == "succeeded":
                return data["analyzeResult"]["content"]
            if data.get("status") == "failed":
                raise PermanentError(f"docintel analysis failed for {path.name}")
            await asyncio.sleep(float(r.headers.get("Retry-After", 1)))
    raise TransientError(f"docintel timed out for {path.name}")


async def load_chunks(docs_dir: Path, s: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.iterdir()):
        ext = path.suffix.lower()
        if ext in TEXT_EXT:
            raw = path.read_text(encoding="utf-8")
        elif ext in DI_EXT and s.azure_docintel_endpoint:
            raw = await document_intelligence_markdown(path, s)
        else:
            if ext in DI_EXT:
                log.warning("skipping %s: set LC_AZURE_DOCINTEL_* to ingest binary documents", path.name)
            continue
        chunks.extend(chunk_markdown(path.stem, raw, s.chunk_size, s.chunk_overlap))
    return chunks


async def ingest(store: VectorStore, s: Settings) -> int:
    return await store.upsert(await load_chunks(s.docs_dir, s))
