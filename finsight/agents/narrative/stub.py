"""Orchestrator narrative entry point and the retained offline demo agent.

Register ``NarrativeAgent`` for the real evidence workflow. ``StubNarrativeAgent``
continues to serve deterministic demo/test evidence.
"""

from __future__ import annotations

from datetime import date
import json
import logging
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from finsight.agents.demo_corpus import NARRATIVE_CHUNKS, company_name
from finsight.agents.narrative.rag.collect_evidence import NarrativeEvidenceAgent, build_queries
from finsight.contracts.schemas import SubClaim, SubClaimType, EvidenceItem
from finsight.llm.usage import record_tool_call

logger = logging.getLogger(__name__)

_TOPIC_BY_TYPE = {
    SubClaimType.DEMAND_ATTRIBUTION: "demand",
    SubClaimType.CONTINUATION: "continuation",
    SubClaimType.BUSINESS_RISK: "risk",
    SubClaimType.COMPARATIVE: "comparative",
    SubClaimType.HISTORICAL_GROWTH: "growth",
}


class NarrativeAgent(NarrativeEvidenceAgent):
    """Real narrative collector exposed at the original integration module.

    SubClaim -> support/counter retrieval -> deduplication -> canonical provenance
    -> LLM relevance selection, exact facts and cited evidence summary
    -> list[EvidenceItem] without stance judgments. Feedback is forwarded
    to the existing bounded query builder; the orchestrator owns retries.

    This collector never classifies passage stance or generates a claim verdict.
    The required EvidenceItem.stance uses NEUTRAL as an unassessed compatibility
    placeholder; exported reports omit it. Auditor/policy integration must account
    for unassessed evidence before producing meaningful verdicts.
    """

    def __init__(
        self,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        analysis_mode: str | None = None,
        analysis_max_tokens: int | None = None,
        analysis_dir: Path | str | None = None,
        ollama_model: str | None = None,
        ollama_host: str | None = None,
        ollama_context: int | None = None,
        **options,
    ) -> None:
        requested_mode = analysis_mode or os.getenv("FINSIGHT_NARRATIVE_ANALYSIS", "evidence")
        if requested_mode not in {"evidence", "retrieval", "stance", "basic_rag"}:
            raise ValueError("FINSIGHT_NARRATIVE_ANALYSIS must be evidence or retrieval")
        if requested_mode in {"stance", "basic_rag"}:
            logger.warning(
                "Legacy analysis mode now prepares evidence without stance/verdict. "
                "Set FINSIGHT_NARRATIVE_ANALYSIS=evidence."
            )
        self.analysis_mode = "retrieval" if requested_mode == "retrieval" else "evidence"
        self.analysis_max_tokens = (
            analysis_max_tokens
            if analysis_max_tokens is not None
            else int(os.getenv("FINSIGHT_NARRATIVE_ANALYSIS_MAX_TOKENS", "1500"))
        )
        self.ollama_model = ollama_model or os.getenv("FINSIGHT_NARRATIVE_OLLAMA_MODEL")
        self.ollama_host = ollama_host or os.getenv("FINSIGHT_NARRATIVE_OLLAMA_HOST", "http://localhost:11434")
        self.ollama_context = (
            ollama_context
            if ollama_context is not None
            else int(os.getenv("FINSIGHT_NARRATIVE_OLLAMA_CONTEXT", "8192"))
        )
        if self.analysis_max_tokens < 1 or self.ollama_context < 1:
            raise ValueError("Token/context limits must be positive")
        self.analysis_dir = Path(
            analysis_dir
            or os.getenv("FINSIGHT_NARRATIVE_ANALYSIS_DIR", "data/narrative/experiments/orchestrator_narrative")
        )
        self.last_analysis: dict | None = None
        self.last_analysis_path: Path | None = None
        super().__init__(
            top_k=top_k if top_k is not None else int(os.getenv("FINSIGHT_NARRATIVE_TOP_K", "5")),
            candidate_k=candidate_k
            if candidate_k is not None
            else int(os.getenv("FINSIGHT_NARRATIVE_CANDIDATE_K", "50")),
            **options,
        )

    def gather(self, sub_claim: SubClaim, cutoff_date: date, feedback: str | None = None) -> list[EvidenceItem]:
        self.last_analysis = None
        self.last_analysis_path = None
        started = time.perf_counter()
        items = super().gather(sub_claim, cutoff_date, feedback)
        retrieval_ms = round((time.perf_counter() - started) * 1000, 3)
        candidates = items
        preparation = {}
        if self.analysis_mode == "evidence":
            from finsight.agents.narrative.rag.evidence_summary import prepare_narrative_evidence

            items, preparation = prepare_narrative_evidence(
                sub_claim,
                cutoff_date,
                items,
                max_tokens=self.analysis_max_tokens,
                ollama_model=self.ollama_model,
                ollama_host=self.ollama_host,
                ollama_context=self.ollama_context,
            )
        report = {
            "status": "OK" if items else "NO_EVIDENCE",
            **preparation,
            "pipeline": "narrative_evidence_rag"
            if self.analysis_mode == "evidence"
            else "narrative_evidence_retrieval",
            "assessment_status": "NOT_ASSESSED",
            "ticker": sub_claim.company_ticker,
            "claim": sub_claim.text,
            "cutoff_date": cutoff_date.isoformat(),
            "backend": self.backend,
            "sub_claim_id": sub_claim.sub_claim_id,
            "sub_claim_type": sub_claim.sub_claim_type.value,
            "feedback": feedback,
            "collection_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "retrieval_elapsed_ms": retrieval_ms,
            "candidate_count": len(candidates),
            "returned_count": len(items),
            "queries": dict(zip(("support", "counter"), build_queries(sub_claim, feedback))),
            "evidence": [item.model_dump(mode="json", exclude={"stance"}) for item in items],
            "contract_note": "Internal EvidenceItem.stance=NEUTRAL is an unassessed compatibility "
            "placeholder, not a classification.",
            "verdict_authority": "orchestrator auditor and tier policy",
        }
        self.last_analysis = report
        if report["status"] in {"GENERATION_ERROR", "INVALID_GENERATION"}:
            logger.warning("Narrative evidence preparation %s; original candidate evidence retained", report["status"])
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{sub_claim.company_ticker}-{sub_claim.sub_claim_id}")
        path = self.analysis_dir / f"{slug}-{uuid4().hex}.json"
        try:
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.last_analysis_path = path
        except OSError:
            logger.warning(
                "Could not save Narrative retrieval report to %s; evidence retained in memory",
                path,
                exc_info=True,
            )
        return items


class StubNarrativeAgent:
    name = "narrative_stub"

    def gather(self, sub_claim: SubClaim, cutoff_date: date, feedback: str | None) -> list[EvidenceItem]:
        if sub_claim.company_ticker != "CMPX":
            return []
        record_tool_call()  # vector search
        topic = _TOPIC_BY_TYPE.get(sub_claim.sub_claim_type)
        items = []
        for chunk in NARRATIVE_CHUNKS:
            if chunk["publication_date"] > cutoff_date:
                continue
            if topic is not None and topic not in chunk["topics"]:
                continue
            stance = chunk["topics"][topic] if topic else next(iter(chunk["topics"].values()))
            fields = {k: v for k, v in chunk.items() if k != "topics"}
            items.append(
                EvidenceItem(
                    **fields,
                    sub_claim_id=sub_claim.sub_claim_id,
                    stance=stance,
                    company=company_name(sub_claim.company_ticker),
                    produced_by=self.name,
                )
            )
        return items
