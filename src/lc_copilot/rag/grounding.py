"""Grounding checks — the gate between a generated answer and the user.

Document answers: every factual sentence must carry a ``[n]`` citation that points at a
retrieved chunk *and* be lexically supported by that chunk. Data answers: every number in
the answer must appear in the SQL result (catches the classic Talk2Data failure of an LLM
"rounding" or inventing a total). Cheap, deterministic, and run on every request; an LLM-as-judge
groundedness score is added in the eval harness when a real model is configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .chunking import Chunk, split_sentences
from .text import numbers, tokens

CITE = re.compile(r"\[(\d+)\]")


@dataclass
class GroundingReport:
    score: float
    cited: list[int] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported and not self.invalid_citations

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "cited": self.cited,
            "unsupported": self.unsupported,
            "invalid_citations": self.invalid_citations,
        }


def check_document_answer(answer: str, contexts: list[Chunk], min_support: float = 0.5) -> GroundingReport:
    sentences = [s for s in split_sentences(answer) if tokens(CITE.sub("", s))]
    if not sentences:
        return GroundingReport(0.0, unsupported=[answer])
    supports, unsupported, cited, invalid = [], [], set(), set()
    for sent in sentences:
        refs = [int(n) for n in CITE.findall(sent)]
        bad = [r for r in refs if not 1 <= r <= len(contexts)]
        invalid.update(bad)
        good = [r for r in refs if r not in bad]
        cited.update(good)
        claim = set(tokens(CITE.sub("", sent)))
        if not good:
            supports.append(0.0)
            unsupported.append(sent)
            continue
        source = set(tokens(" ".join(contexts[r - 1].indexed_text for r in good)))
        support = len(claim & source) / max(len(claim), 1)
        # numbers must match exactly, not just "mostly"
        if numbers(CITE.sub("", sent)) - numbers(" ".join(contexts[r - 1].text for r in good)):
            support = min(support, 0.0)
        supports.append(support)
        if support < min_support:
            unsupported.append(sent)
    return GroundingReport(sum(supports) / len(supports), sorted(cited), unsupported, sorted(invalid))


def check_data_answer(answer: str, rows: list[dict], question: str = "") -> GroundingReport:
    allowed = numbers(question)
    for row in rows:
        for v in row.values():
            allowed |= numbers(str(v))
            if isinstance(v, float):
                allowed |= numbers(f"{v:.2f}")
    allowed |= {str(len(rows))}
    claimed = numbers(answer)
    bad = sorted(claimed - allowed)
    score = 1.0 if not claimed else (len(claimed) - len(bad)) / len(claimed)
    return GroundingReport(score, unsupported=[f"number not in result: {b}" for b in bad])
