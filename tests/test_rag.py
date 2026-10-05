from lc_copilot.rag.chunking import Chunk, chunk_markdown
from lc_copilot.rag.grounding import check_data_answer, check_document_answer
from lc_copilot.rag.text import numbers
from tests.conftest import run

DOC = """---
title: Test Credit
lc_number: LC-2026-0001
---
# Test Credit
## Dates
The credit expires on 2026-11-30. The presentation period is 21 days.
## Shipment
Partial shipments are prohibited. Transhipment is allowed.
"""


def test_chunks_follow_headings_and_keep_metadata():
    chunks = chunk_markdown("t", DOC)
    assert [c.section for c in chunks] == ["Dates", "Shipment"]
    assert all(c.metadata["lc_number"] == "LC-2026-0001" for c in chunks)
    assert chunks[1].indexed_text.startswith("Test Credit — Shipment")


def test_long_sections_split_with_overlap():
    body = "# T\n## S\n" + " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_markdown("t", body, size=200, overlap=60)
    assert len(chunks) > 5
    assert all(len(c.text) <= 260 for c in chunks)
    first_tail = chunks[0].text.split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1].text  # overlap carried forward


def test_entity_filter_keeps_other_credits_out(svc):
    hits = run(svc.retriever.retrieve("Are partial shipments allowed under LC-2026-0187?"))
    lcs = {h.chunk.metadata.get("lc_number") for h in hits} - {None}
    assert lcs <= {"LC-2026-0187"}
    assert hits[0].chunk.doc_id == "LC-2026-0187"


def test_hybrid_search_finds_policy_by_keyword(svc):
    hits = run(svc.retriever.retrieve("four-eyes approval of refusal notices"))
    assert "policy-document-examination" in [h.chunk.doc_id for h in hits[:3]]


CTX = [Chunk("a#0", "a", "Credit A", "Shipment", "Partial shipments are prohibited. The tolerance is 5%.")]


def test_grounding_accepts_cited_supported_answer():
    rep = check_document_answer("Partial shipments are prohibited [1].", CTX)
    assert rep.ok and rep.score == 1.0


def test_grounding_rejects_uncited_and_hallucinated_numbers():
    assert not check_document_answer("Partial shipments are prohibited.", CTX).ok
    assert not check_document_answer("The tolerance is 10% [1].", CTX).ok
    assert check_document_answer("Partial shipments are prohibited [4].", CTX).invalid_citations == [4]


def test_data_grounding_catches_invented_totals():
    rows = [{"currency": "USD", "total_amount": 1890000.0}]
    assert check_data_answer("Total USD exposure is 1,890,000.00.", rows).ok
    assert not check_data_answer("Total USD exposure is roughly 1,900,000.", rows).ok


def test_number_normalisation():
    assert numbers("USD 1,250,000.00 and 5% over 21 days") == {"1250000", "5", "21"}
    assert numbers("LC-2026-0142 on 2026-11-30") == set()
