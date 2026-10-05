"""Shared text utilities: tokenisation, entity extraction, number normalisation."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being do does did
    what which who whom whose when where why how this that these those it its into than then there their
    under over can could should would may might must shall will i we you he she they me us our your
    any all each per not no yes about have has had say says said tell give show""".split()  # noqa: SIM905
)

LC_ID = re.compile(r"\bLC-\d{4}-\d{4}\b", re.IGNORECASE)
PRS_ID = re.compile(r"\bPRS-\d{4}-\d{2}\b", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_NUM = re.compile(r"(?<![\w-])\d[\d,]*(?:\.\d+)?(?![\w-])")


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if not keep_stop and t in STOPWORDS:
            continue
        out.append(_stem(t))
    return out


def _stem(t: str) -> str:
    """Tiny suffix stripper — enough to match shipment/shipments, examine/examination."""
    if "-" in t or t.isdigit():
        return t
    for suf in ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "es", "ed", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def entity_ids(text: str) -> dict[str, list[str]]:
    return {
        "lc_number": sorted({m.upper() for m in LC_ID.findall(text)}),
        "presentation_id": sorted({m.upper() for m in PRS_ID.findall(text)}),
    }


def numbers(text: str) -> set[str]:
    """Normalised numbers in text: '1,250,000.00' -> '1250000', '5%' -> '5'."""
    out = set()
    for raw in _NUM.findall(text):
        n = raw.replace(",", "")
        try:
            f = float(n)
        except ValueError:
            continue
        out.add(str(int(f)) if f == int(f) else f"{f:.2f}".rstrip("0").rstrip("."))
    return out
