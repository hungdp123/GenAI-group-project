"""A stated claim selects a stored row that shares its figure. A transcript line is confirmed first."""

import json
from datetime import date

from finsight.agents.claim_extraction.extract import extract_claims
from finsight.agents.claim_extraction.select import (
    NOT_IN_STORED,
    NOT_IN_VIDEO,
    choose_line,
    explain_miss,
    matching_claims,
    shares_figure,
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


def test_a_stated_percentage_does_not_match_a_different_percentage():
    assert shares_figure("revenue grew 24%", "search revenue increased by 17%") is False
    assert shares_figure("revenue grew 24%", "23% revenue growth") is False
    assert shares_figure("revenue grew 24%", "revenue grew by 24% year-over-year") is True
    assert shares_figure("revenue was up about 100%", "up 106% year-over-year") is True
    assert shares_figure("revenue doubled", "up 106% year-over-year") is True


def test_one_stored_row_is_checked_without_asking():
    focus = "revenue doubled"
    choice = choose_line(focus, [(REVENUE, "vid1"), (MARGIN, "vid1")], [_post("vid1", REVENUE.text)])
    assert choice.question is None
    assert choice.video_id == "vid1"
    assert choice.quotes == [REVENUE.text]


def test_24_percent_does_not_attach_to_the_17_percent_row():
    search = _claim("search", "Google search revenue increased by 17%.", SubClaimType.HISTORICAL_GROWTH)
    spoken = "Revenue grew by 24% year-over-year, marking its twelfth consecutive quarter of double-digit growth."
    choice = choose_line(
        "Google revenue grew 24% last quarter, its twelfth straight quarter of double-digit growth.",
        [(search, "vid1")],
        [_post("vid1", f"{search.text} {spoken}")],
        ticker="GOOGL",
    )
    assert choice.question is None
    assert choice.video_id == "vid1"
    assert any("24%" in quote for quote in choice.quotes)
    assert all("17%" not in quote for quote in choice.quotes)


def test_a_company_stays_in_force_until_another_company_is_named():
    transcript = " ".join([
        "Shares of Meta are down 8% after reporting.",
        "Investors got spooked and hit the sell button.",
        "How about operating profits?",
        "We have a company that is investing leading to higher revenues, higher operating profits, "
        "and bigger operating margins.",
        "Getting back to Amazon, revenues are at all-time highs and operating margins are at all-time highs.",
    ])
    choice = choose_line(
        "revenue, operating profit and operating margin all went up",
        [],
        [_post("vid1", transcript)],
        ticker="META",
    )
    assert any("bigger operating margins" in quote for quote in choice.quotes)
    assert all("Amazon" not in quote for quote in choice.quotes)


def test_one_spoken_sentence_is_shown_before_it_is_checked():
    spoken = "Did Company X revenue double?"
    choice = choose_line("revenue doubled", [(MARGIN, "vid1")], [_post("vid1", spoken)])
    assert choice.question is None
    assert choice.quotes == [spoken]
    confirmed = choose_line(
        "revenue doubled", [(MARGIN, "vid1")], [_post("vid1", spoken)], confirmed_video="vid1",
    )
    assert confirmed.question is None
    assert confirmed.quotes == [spoken]


def test_nothing_close_asks_the_user_to_restate_the_line():
    choice = choose_line("opened a factory in Ohio", [(MARGIN, "vid1")], [_post("vid1", MARGIN.text)])
    assert choice.quotes == []
    assert choice.question == NOT_IN_STORED


def _post(video_id: str, transcript: str) -> dict:
    return {"video_id": video_id, "upload_date": "2026-07-24", "transcript": transcript}


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


def test_direction_agreement_beats_word_overlap():
    """'Meta's free cash flow went down': the line that says FCF is going down wins over a line that repeats
    'free cash flow' but is about margins, even when only the second one is in the claims catalog."""
    from finsight.agents.claim_extraction.select import choose_line
    from finsight.contracts.schemas import SubClaim, SubClaimType

    wanted = "So, free cash flow is going down because Meta doesn't have enough money to pay for this CapEx."
    other = ("So, CapEx going up a little bit, revenue staying the same, equals a stock that's going to have even "
             "lower margins, more of a drag on the free cash flow, even more negative free cash flow.")
    post = {"video_id": "v1", "upload_date": "2026-04-30", "transcript": f"Meta is spending a lot. {other} {wanted}"}
    stored = [(SubClaim(sub_claim_id="t272", text=other, sub_claim_type=SubClaimType.HISTORICAL_GROWTH,
                        company_ticker="META"), "v1")]
    choice = choose_line("Meta's free cash flow went down last quarter.", stored, [post], ticker="META")
    assert choice.quotes and choice.quotes[0] == wanted
