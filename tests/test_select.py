"""Extraction helpers kept in select.py. Matching tests are in test_claim_matching.py."""

import json
from datetime import date

from finsight.agents.claim_extraction.extract import extract_claims
from finsight.agents.claim_extraction.select import (
    NOT_IN_VIDEO,
    explain_miss,
    matching_claims,
    stated_claim,
)
from finsight.contracts.schemas import SubClaim, SubClaimType
from finsight.knowledge.loader import default_knowledge_layer


_ALIASES = ["Company X", "CMPX"]


def _claim(claim_id: str, text: str, claim_type: SubClaimType) -> SubClaim:
    return SubClaim(
        sub_claim_id=claim_id, text=text, sub_claim_type=claim_type, company_ticker="CMPX",
    )


REVENUE = _claim(
    "rev",
    "Company X reported revenues of 96.2 billion, up 106% year-over-year.",
    SubClaimType.HISTORICAL_GROWTH,
)
MARGIN = _claim("margin", "Company X margins are expanding.", SubClaimType.PROFITABILITY_CASH_LEVERAGE)


def _extract(text: str, stored: list[SubClaim], focus: str | None):
    return extract_claims(
        text=text,
        cutoff=date(2025, 7, 31),
        company_ticker="CMPX",
        companies={"CMPX": _ALIASES},
        from_stored_post=True,
        knowledge=default_knowledge_layer(),
        call=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("call")),
        stored=stored,
        focus=focus,
    )


def test_a_prompt_that_only_names_the_creator_states_no_claim():
    assert stated_claim("Is Joseph Carlson right about Company X?", "joseph_carlson", _ALIASES) is None
    assert stated_claim("Is Mark Roussin right about NVIDIA?", "mark_roussin", ["NVIDIA", "Nvidia", "NVDA"]) is None


def test_revenue_doubled_matches_the_stored_growth_sentence():
    focus = stated_claim("Joseph Carlson said Company X revenue doubled.", "joseph_carlson", _ALIASES)
    assert focus is not None
    assert matching_claims(focus, [REVENUE, MARGIN]) == [REVENUE]
    about_hundred = stated_claim(
        "Joseph Carlson said Company X revenue was up about 100%.", "joseph_carlson", _ALIASES,
    )
    assert matching_claims(about_hundred or "", [REVENUE, MARGIN]) == [REVENUE]


def test_a_stated_claim_checks_only_the_matching_rows():
    out = _extract(
        f"{REVENUE.text} {MARGIN.text}",
        [REVENUE, MARGIN],
        stated_claim("Joseph Carlson said Company X revenue doubled.", "joseph_carlson", _ALIASES),
    )
    assert [claim.sub_claim_id for claim in out.sub_claims] == ["rev"]
    assert out.miss is None


def test_a_question_in_the_transcript_is_explained_and_not_verified():
    out = _extract(
        f"{MARGIN.text} Did Company X revenue double?",
        [MARGIN],
        "revenue doubled",
    )
    assert out.sub_claims == []
    assert out.miss is not None
    assert "Did Company X revenue double?" in out.miss
    assert "question" in out.miss


def test_a_line_that_is_not_in_the_video_stops():
    out = _extract(MARGIN.text, [MARGIN], "opened a factory in Ohio")
    assert out.sub_claims == []
    assert out.miss == NOT_IN_VIDEO


def test_a_price_line_is_explained_from_the_transcript():
    sentence = "The stock is trading at 40."
    message = explain_miss("trading at 40", sentence)
    assert sentence in message
    assert "price" in message


def test_a_sentence_the_build_missed_is_not_verified():
    sentence = "Company X bought back stock last quarter."
    message = explain_miss("bought back stock", sentence)
    assert "not in the stored claim list" in message


def test_a_post_without_stored_claims_is_not_split():
    out = extract_claims(
        text="Company X revenue grew 24% and margins expanded.",
        cutoff=date(2025, 7, 31),
        company_ticker="CMPX",
        companies={"CMPX": _ALIASES},
        from_stored_post=True,
        knowledge=default_knowledge_layer(),
        call=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("call")),
    )
    assert out.sub_claims == []
    assert out.clarification_question == "This post has no stored claims to check."


def test_the_cap_of_8_applies_after_the_match():
    stored = [
        _claim(f"r{i}", "Customer concentration is a risk.", SubClaimType.BUSINESS_RISK)
        for i in range(9)
    ]
    out = _extract("Customer concentration is a risk.", stored, "customer concentration is a risk")
    assert len(out.sub_claims) == 8
    assert out.dropped_by_cap














def test_stored_record_round_trip_still_skips_the_decomposer(tmp_path):
    path = tmp_path / "claims.jsonl"
    path.write_text(json.dumps({
        "video_id": "abc", "company_ticker": "CMPX", "sub_claim_id": "rev",
        "text": REVENUE.text, "sub_claim_type": "HISTORICAL_GROWTH",
    }) + "\n", encoding="utf-8")
    from finsight.agents.claim_extraction.extract import load_stored_claims

    stored = load_stored_claims(path, "abc", "CMPX")
    assert stored is not None
    out = _extract(REVENUE.text, stored, None)
    assert out.sub_claims[0].sub_claim_type is SubClaimType.HISTORICAL_GROWTH
    assert out.miss is None
