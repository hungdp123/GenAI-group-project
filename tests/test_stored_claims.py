"""Stored claims are a table of sentences. A question does not extract the post again."""

import json
from datetime import date

from finsight.agents.claim_extraction.extract import extract_claims, keep_within_cap, load_stored_claims
from finsight.contracts.schemas import SubClaim, SubClaimType
from finsight.knowledge.loader import default_knowledge_layer


def test_a_stored_record_is_used_and_the_decomposer_is_not_called(tmp_path):
    path = tmp_path / "claims.jsonl"
    path.write_text(json.dumps({
        "video_id": "abc",
        "company_ticker": "AAPL",
        "sub_claim_id": "c1",
        "text": "Revenue grew about 14%.",
        "sub_claim_type": "HISTORICAL_GROWTH",
        "source_span": "Revenue grew about 14%.",
    }) + "\n", encoding="utf-8")
    stored = load_stored_claims(path, "abc", "AAPL")
    assert stored is not None and stored[0].text == "Revenue grew about 14%."
    assert load_stored_claims(path, "missing", "AAPL") is None

    out = extract_claims(
        text="ignored transcript",
        cutoff=date(2026, 9, 29),
        company_ticker="AAPL",
        companies={"AAPL": ["Apple"]},
        from_stored_post=True,
        knowledge=default_knowledge_layer(),
        call=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("call")),
        stored=stored,
    )
    assert out.sub_claims[0].sub_claim_type is SubClaimType.HISTORICAL_GROWTH
    assert out.status is None


def test_a_judged_revenue_measure_keeps_a_growth_row_on_the_financial_agent(tmp_path):
    path = tmp_path / "claims.jsonl"
    rows = [
        {"text": "Google grew by 24% year-over-year.", "label_new": {"measure": "revenue"}},
        {"text": "22% is massive.", "inherited_measure": "revenue"},
        # A judged profit measure is not trusted: an earnings-report sentence stays OTHER.
        {"text": "Nvidia absolutely crushed this earnings report.", "label_new": {"measure": "profit"}},
    ]
    path.write_text("".join(json.dumps({
        "video_id": "abc", "company_ticker": "GOOGL", "sub_claim_id": f"c{i}",
        "sub_claim_type": "HISTORICAL_GROWTH", "measure": None, **row,
    }) + "\n" for i, row in enumerate(rows)), encoding="utf-8")
    stored = load_stored_claims(path, "abc", "GOOGL")
    assert [c.sub_claim_type for c in stored] == [
        SubClaimType.HISTORICAL_GROWTH, SubClaimType.HISTORICAL_GROWTH, SubClaimType.OTHER,
    ]


def test_the_cap_of_8_lists_the_quotes_it_leaves_out():
    claims = [
        SubClaim(sub_claim_id=f"c{i}", text=f"point {i}", sub_claim_type=SubClaimType.BUSINESS_RISK)
        for i in range(9)
    ]
    claims.append(SubClaim(sub_claim_id="keep", text="biggest winner", sub_claim_type=SubClaimType.COMPARATIVE))
    kept, dropped = keep_within_cap(claims)
    assert len(kept) == 8
    assert any(claim.text == "biggest winner" for claim in kept)
    assert "point 8" in dropped
