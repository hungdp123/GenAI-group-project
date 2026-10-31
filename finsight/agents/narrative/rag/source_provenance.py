"""Attach approved paid-source attribution at runtime without rewriting saved indexes."""
from __future__ import annotations

from functools import lru_cache
import json
import logging
from pathlib import Path

from finsight.agents.narrative.corpus.processing.corpus import CANONICAL_MANIFESTS
from finsight.contracts.schemas import PAID_SOURCE_URL_EXCEPTION

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _manifest_attributions(path: str, modified_ns: int, size: int) -> dict[tuple[str, str], dict]:
    """Cache by file revision so later manifest edits are visible in a running UI."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Could not read source attribution manifest %s", path)
        return {}
    rows = payload.get("documents", []) if isinstance(payload, dict) else []
    attributions = {}
    for row in rows if isinstance(rows, list) else []:
        if (not isinstance(row, dict) or row.get("source_access") != "paid_subscription"
                or row.get("source_url_requirement") != PAID_SOURCE_URL_EXCEPTION):
            continue
        document_id, ticker, page = row.get("document_id"), row.get("company_ticker"), row.get("source_page")
        if not all(isinstance(value, str) and value.strip() for value in (document_id, ticker, page)):
            continue
        attributions[document_id, ticker.upper()] = {
            "source_page": page,
            "source_url_requirement": row["source_url_requirement"],
        }
    return attributions


def source_attribution(document_id: str | None, ticker: str | None) -> dict:
    """Only canonical manifests authorize an exception; retrieved/LLM metadata cannot."""
    if not document_id or not ticker:
        return {}
    for manifest in CANONICAL_MANIFESTS:
        path = Path(manifest).resolve()
        try:
            stat = path.stat()
        except OSError:
            continue
        attribution = _manifest_attributions(str(path), stat.st_mtime_ns, stat.st_size).get(
            (document_id, ticker.upper()))
        if attribution:
            return dict(attribution)
    return {}
