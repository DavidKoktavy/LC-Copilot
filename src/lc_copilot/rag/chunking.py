"""Structure-aware chunking.

Splits Markdown (which is also what Azure Document Intelligence ``prebuilt-layout`` emits)
on headings first, then packs sentences into size-bounded chunks with overlap. Each chunk keeps
its heading path ("Documentary Credit LC-2026-0142 > Dates") and is prefixed with it, which
markedly improves retrieval for short, context-free paragraphs such as "Partial shipments are
prohibited."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def indexed_text(self) -> str:
        return f"{self.title} — {self.section}\n{self.text}"


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    m = _FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[m.end():]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


def _sections(body: str) -> list[tuple[str, str]]:
    path: list[str] = []
    out: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            # the H1 repeats the document title, so drop it from the section path when deeper headings exist
            shown = path[1:] if len(path) > 1 else path
            out.append((" > ".join(shown) or "Document", text))
        buf.clear()

    for line in body.splitlines():
        if m := _HEADING.match(line):
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return out


def chunk_markdown(doc_id: str, raw: str, size: int = 700, overlap: int = 120) -> list[Chunk]:
    meta, body = parse_frontmatter(raw)
    title = meta.get("title", doc_id)
    chunks: list[Chunk] = []
    for section, text in _sections(body):
        # numbered list items count as sentences
        units = [u for para in re.split(r"\n\s*\n|\n(?=\d+\.\s)", text) for u in split_sentences(para)]
        window: list[str] = []
        for unit in units:
            if window and len(" ".join(window + [unit])) > size:
                chunks.append(_mk(doc_id, title, section, window, meta, len(chunks)))
                # carry trailing sentences as overlap
                carry: list[str] = []
                for s in reversed(window):
                    if len(" ".join([s] + carry)) > overlap:
                        break
                    carry.insert(0, s)
                window = carry
            window.append(unit)
        if window:
            chunks.append(_mk(doc_id, title, section, window, meta, len(chunks)))
    return chunks


def _mk(doc_id: str, title: str, section: str, sents: list[str], meta: dict, n: int) -> Chunk:
    return Chunk(f"{doc_id}#{n}", doc_id, title, section, " ".join(sents), dict(meta))
