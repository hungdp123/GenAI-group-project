# PLACEHOLDER: owned by Person 3 (Amelia). Replace with real implementation. Do not change the signature.
"""Auditor stub. Verdict from evidence stances; FINSIGHT_STUB_FAIL_CLAIMS forces failures for retry tests.

FINSIGHT_STUB_FAIL_CLAIMS="c1,c2:2" -> c1 fails its first audit, c2 fails its first two audits.
"""

from __future__ import annotations

import os
from collections import defaultdict

from finsight.contracts.schemas import (
    AuditResult,
    CitationCheck,
    Claim,
    EvidenceItem,
    NumericCheck,
    Stance,
    Verdict,
)


def _forced_failures() -> dict[str, int]:
    out: dict[str, int] = {}
    for part in os.getenv("FINSIGHT_STUB_FAIL_CLAIMS", "").split(","):
        if part.strip():
            claim_id, _, times = part.strip().partition(":")
            out[claim_id] = int(times or 1)
    return out


class StubAuditor:
    name = "auditor_stub"

    def __init__(self) -> None:
        self._attempts: dict[str, int] = defaultdict(int)

    def audit(self, claim: Claim, evidence: list[EvidenceItem]) -> AuditResult:
        self._attempts[claim.claim_id] += 1
        if self._attempts[claim.claim_id] <= _forced_failures().get(claim.claim_id, 0):
            return AuditResult(
                claim_id=claim.claim_id, passed=False, verdict=Verdict.INSUFFICIENT_EVIDENCE,
                failure_reasons=["Forced failure via FINSIGHT_STUB_FAIL_CLAIMS"],
                feedback_for_retry="Return a Tier 1 source with an exact citation span.",
            )
        supporting = [e for e in evidence if e.stance is Stance.SUPPORTING]
        challenging = [e for e in evidence if e.stance is Stance.CHALLENGING]
        if supporting and challenging:
            verdict = Verdict.PARTIALLY_SUPPORTED
        elif supporting:
            verdict = Verdict.SUPPORTED
        elif challenging:
            verdict = Verdict.CONTRADICTED
        else:
            verdict = Verdict.INSUFFICIENT_EVIDENCE
        numeric = next((e for e in evidence if e.computed_value is not None), None)
        return AuditResult(
            claim_id=claim.claim_id, passed=True, verdict=verdict,
            numeric_check=NumericCheck(metric_name=numeric.metric_name, computed=numeric.computed_value,
                                       matches=True) if numeric else None,
            citation_check=CitationCheck(evidence_id=evidence[0].evidence_id, entails=bool(supporting))
            if evidence else None,
        )
