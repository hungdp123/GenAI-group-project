"""Overall-verdict aggregation and per-sub-claim summaries. Owner: Person 5 (Hung)."""

from __future__ import annotations

from finsight.contracts.schemas import EvidenceItem, SourceTier, Verdict


def overall_verdict(verdicts: list[Verdict]) -> Verdict:
    """Deterministic overall verdict from per-sub-claim verdicts.

    Rule (TEAM MUST CONFIRM before the final report):
      1. no sub-claims, or every sub-claim INSUFFICIENT_EVIDENCE      -> INSUFFICIENT_EVIDENCE
      2. every sub-claim SUPPORTED                                  -> SUPPORTED
      3. more than half of the *analyzable* sub-claims CONTRADICTED -> CONTRADICTED
         (analyzable = not INSUFFICIENT_EVIDENCE)
      4. otherwise                                              -> PARTIALLY_SUPPORTED
    """
    if not verdicts or all(v is Verdict.INSUFFICIENT_EVIDENCE for v in verdicts):
        return Verdict.INSUFFICIENT_EVIDENCE
    if all(v is Verdict.SUPPORTED for v in verdicts):
        return Verdict.SUPPORTED
    analyzable = [v for v in verdicts if v is not Verdict.INSUFFICIENT_EVIDENCE]
    if sum(v is Verdict.CONTRADICTED for v in analyzable) > len(analyzable) / 2:
        return Verdict.CONTRADICTED
    return Verdict.PARTIALLY_SUPPORTED


def highest_tier(evidence: list[EvidenceItem]) -> SourceTier | None:
    """Most authoritative tier among the evidence (TIER_1 beats TIER_2 beats TIER_3)."""
    return min((e.source_tier for e in evidence), key=lambda t: t.rank, default=None)
