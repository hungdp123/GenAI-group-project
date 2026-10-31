"""Claim matching: filter -> rank -> pick -> align (finsight/agents/claim_matching).

These replace the old choose_line tests in test_select.py. The behaviour change on purpose: a different figure
on the same measure is now a `conflict` (the influencer's own line is checked, the mismatch is shown) instead of
"that line is not in the video".
"""

from __future__ import annotations

from datetime import date

from finsight.agents.claim_matching.catalog import Line, lines_for, posts_from_catalog, transcript_lines
from finsight.agents.claim_matching.facts import align, split_facts
from finsight.agents.claim_matching.matcher import clean_user_claim, match_claim
from finsight.agents.claim_matching.subclaims import build_sub_claims, headline_ids, to_contract
from finsight.contracts.schemas import SubClaim, SubClaimType
from finsight.llm.client import set_mock_responder


def _line(text: str, video: str = "vid1", ticker: str = "CMPX", line_id: str | None = None,
          when: str = "2026-07-24", channel: str = "joseph_carlson") -> Line:
    return Line(line_id=line_id or f"{video}-{abs(hash(text)) % 10000}", video_id=video, channel=channel,
                title=f"Video {video}", upload_date=date.fromisoformat(when), ticker=ticker, text=text, row={})


REVENUE = "Company X reported revenues of 96.2 billion, up 106% year-over-year."
MARGIN = "Company X margins are expanding."


# -- facts ---------------------------------------------------------------------------------


def test_spans_are_the_speakers_exact_words():
    text = ("Revenues of 82.9 billion, which was an 18.3% year-over-year growth, "
            "beat handily by nearly 1 and 1/2 billion.")
    facts = split_facts(text)
    assert [fact.span for fact in facts] == [
        "Revenues of 82.9 billion", "which was an 18.3% year-over-year growth",
        "beat handily by nearly 1 and 1/2 billion"]
    assert all(fact.span in text for fact in facts)
    assert facts[1].inherited == {"revenue"}  # the measure carries forward to the next clause


def test_two_figures_in_one_clause_are_two_facts():
    facts = split_facts("Alphabet's revenue grew 22% to $110 billion last quarter.")
    assert [fact.value for fact in facts] == [(22.0, "percent"), (110.0, "billion")]


def test_a_plain_sentence_is_not_cut_at_and():
    facts = split_facts("Starbucks is using artificial intelligence to reduce its reliance on Microsoft "
                        "and IBM software.")
    assert len(facts) == 1


def test_a_leading_to_clause_reports_no_move():
    facts = split_facts("We have a company that is investing leading to higher revenues, higher operating profits.")
    assert all(fact.direction is None for fact in facts)


# -- alignment -----------------------------------------------------------------------------


def _buckets(user: str, quote: str) -> list[tuple[str, str | None]]:
    return [(item.bucket, item.note) for item in align(split_facts(user), [split_facts(quote)])]


def test_the_influencer_said_more():
    out = _buckets("Google grew 24% last quarter.",
                   "Google grew by 24% year-over-year, marking its twelfth consecutive quarter of double-digit growth.")
    assert [bucket for bucket, _ in out] == ["both", "quote_only"]


def test_the_user_added_a_detail():
    out = _buckets("Alphabet's revenue grew 22% to $110 billion last quarter.",
                   "In fact, 22% this most recent quarter.")
    assert out == [("both", None), ("user_only", "They did not say $110 billion in the matched line.")]


def test_a_different_figure_is_a_conflict_not_a_miss():
    out = _buckets("Microsoft's revenue grew 40%.",
                   "Revenues of 82.9 billion, which was an 18.3% year-over-year growth.")
    assert ("conflict", "You said 40%; they said 18.3%.") in out


def test_a_segment_figure_is_not_the_company_figure():
    out = _buckets("Alphabet's revenue was $20 billion last quarter.",
                   "Google Cloud though was impressive with a big beat, and revenues rising to 20 billion "
                   "on the quarter.")
    assert out == [("conflict", "They said $20 billion about Google Cloud, not about the whole company.")]


def test_three_measures_line_up_with_three_clauses():
    out = _buckets("Meta's revenue, operating profit and operating margin all went up.",
                   "Revenues are at all-time highs, operating profits at all-time highs, operating margins are at 41%.")
    assert [bucket for bucket, _ in out] == ["both", "both", "both"]


# -- matching ------------------------------------------------------------------------------


def test_revenue_doubled_picks_the_revenue_line_not_the_margin_line():
    out = match_claim("revenue doubled", [_line(REVENUE), _line(MARGIN)], mode="rules")
    assert out.status == "MATCHED" and [line.text for line in out.lines] == [REVENUE]


def test_24_percent_prefers_the_24_percent_transcript_line_over_a_17_percent_catalog_line():
    search = _line("Google search revenue increased by 17%.", ticker="GOOGL")
    spoken = _line("Revenue grew by 24% year-over-year, marking its twelfth consecutive quarter of "
                   "double-digit growth.", ticker="GOOGL")
    out = match_claim("Google revenue grew 24% last quarter, its twelfth straight quarter of double-digit growth.",
                      [search], fallback=[spoken], mode="rules")
    assert [line.text for line in out.lines] == [spoken.text]


def test_reported_highs_beat_a_leading_to_sentence():
    thesis = _line("We have a company that is investing leading to higher revenues, higher operating profits, "
                   "and bigger operating margins.", ticker="META")
    results = _line("Revenues are at all-time highs, operating profits at all-time highs, operating margins are at 41% "
                    "trailing only 2024 when margins reached 42% over the course of the past 7 years.", ticker="META")
    out = match_claim("Meta's Q1 2026 revenue, operating profit and operating margin all went up.",
                      [thesis, results], mode="rules")
    assert [line.text for line in out.lines] == [results.text]


def test_a_named_segment_is_not_matched_to_a_different_product():
    aws = _line("In Q4 of 2025, AWS reported 24% year-over-year growth with a $142 billion revenue run rate.",
                ticker="AMZN")
    chip = _line("They say that their revenue run rate for their chip business is 20 billion, triple-digit growth.",
                 ticker="AMZN")
    out = match_claim("AWS grew 24% year over year in Q4 2025 with a $142 billion revenue run rate.", [chip, aws],
                      mode="rules")
    assert [line.text for line in out.lines] == [aws.text]
    assert {item.bucket for item in out.aligned} == {"both"}


def test_a_bare_figure_is_used_but_a_named_measure_wins():
    bare = _line("22% is massive.", video="april", ticker="GOOGL")
    margin = _line("operating margin is 22%.", video="april", ticker="GOOGL")
    named = _line("Alphabet revenue grew by 22% last quarter.", video="june", ticker="GOOGL")
    assert [line.text for line in match_claim("Alphabet revenue grew 22%", [bare, margin], mode="rules").lines] == [
        bare.text]
    both = match_claim("Alphabet revenue grew 22%", [bare, named], mode="rules")
    assert both.status == "MATCHED" and both.video_id == "june"


def test_a_figure_nobody_said_is_not_matched():
    wrong = _line("Meta recently said the revenue run rate for video generation tools hits 10 billion.", ticker="META")
    out = match_claim("Meta booked $2.4 billion in costs related to legal proceedings last quarter.", [wrong],
                      mode="rules")
    assert out.status == "NONE" and out.question


def test_direction_beats_word_overlap():
    wanted = _line("So, free cash flow is going down because Meta doesn't have enough money to pay for this CapEx.",
                   ticker="META")
    other = _line("So, CapEx going up a little bit, revenue staying the same, equals a stock that's going to have even "
                  "lower margins, more of a drag on the free cash flow, even more negative free cash flow.",
                  ticker="META")
    out = match_claim("Meta's free cash flow went down last quarter.", [other], fallback=[wanted], mode="rules")
    assert out.lines and out.lines[0].text == wanted.text


def test_the_same_figure_in_two_videos_asks_which_post():
    first = _line("Nvidia reported revenues of 96.2 billion, up 106% year-over-year.", video="aaaa1111", ticker="NVDA")
    second = _line("Nvidia reported revenues of 96.2 billion, up 106% year-over-year.", video="bbbb2222",
                   ticker="NVDA", when="2026-09-01")
    out = match_claim("Nvidia's revenue was $96.2 billion, up 106%.", [first, second], mode="rules")
    assert out.status == "ASK"
    assert "aaaa1111" in out.question and "bbbb2222" in out.question and "Which post" in out.question
    picked = match_claim("Nvidia's revenue was $96.2 billion, up 106%.", [first, second], video_id="bbbb2222",
                         mode="rules")
    assert picked.status == "MATCHED" and picked.video_id == "bbbb2222"


def test_a_price_view_is_set_aside_not_matched():
    revenue = _line("Outside of those, engagement's good and that's why the revenue's up 33%.", video="v1",
                    ticker="META")
    price = _line("So, I believe in 5 years, Meta could be worth $2,000.", video="v2", ticker="META")
    out = match_claim("Meta's revenue grew 33% in Q1 2026 and that the stock could be worth $2,000 in five years.",
                      [revenue, price], mode="rules")
    assert [line.text for line in out.lines] == [revenue.text]
    assert out.skipped and "worth" in out.skipped[0].span


def test_cleaning_the_prompt():
    assert clean_user_claim("  said  's revenue grew 40%. Is that right?") == "revenue grew 40%"
    assert clean_user_claim("says that AWS grew 28%") == "AWS grew 28%"


# -- LLM pick (paid mode; scripted here) ---------------------------------------------------


def test_the_llm_pick_can_only_choose_listed_ids():
    lines = [_line(REVENUE), _line(MARGIN)]
    seen: list[str] = []

    def pick_margin(system: str, prompt: str) -> str:
        seen.append(prompt)
        key = next(row.split(" ")[0] for row in prompt.splitlines() if MARGIN in row)
        return '{"match_ids": ["%s"], "reason": "margins"}' % key

    set_mock_responder(pick_margin)
    try:
        out = match_claim("margins are getting bigger", lines, mode="llm")
        assert out.method == "llm" and [line.text for line in out.lines] == [MARGIN]
        assert '<data name="user_claim">' in seen[0]  # the user's text goes in as data, not instructions
        set_mock_responder(lambda system, prompt: '{"match_ids": ["L9"], "reason": "invented"}')
        fallback = match_claim("margins are getting bigger", lines, mode="llm")
        assert fallback.method == "rules"  # an id outside the list is not trusted
    finally:
        set_mock_responder(None)


# -- catalog + sub-claims --------------------------------------------------------------------


def test_filter_keeps_this_influencer_company_and_cutoff():
    lines = [
        _line("Company X revenue grew 10%.", when="2026-05-01"),
        _line("Company X revenue grew 12%.", when="2026-08-01"),
        _line("Company X revenue grew 14%.", channel="mark_roussin"),
        _line("Other Co revenue grew 9%.", ticker="NVDA"),
    ]
    kept = lines_for(lines, influencer="joseph_carlson", ticker="CMPX", cutoff=date(2026, 6, 1))
    assert [line.text for line in kept] == ["Company X revenue grew 10%."]


def test_posts_can_be_built_from_the_catalog():
    posts = posts_from_catalog([_line("A.", video="v1"), _line("B.", video="v1"), _line("C.", video="v2")])
    assert [(post["video_id"], post["transcript"]) for post in posts] == [("v1", "A. B."), ("v2", "C.")]
    assert transcript_lines(posts, "CMPX") == []  # nothing new to search in a catalog-built post


def _claim(text: str, sub_id: str = "made") -> tuple[SubClaim, str | None]:
    return SubClaim(sub_claim_id=sub_id, text=text, sub_claim_type=SubClaimType.OTHER, company_ticker="GOOGL"), None


def test_a_line_that_mixes_asked_and_unasked_facts_is_split():
    quote = "Google grew by 24% year-over-year, marking its twelfth consecutive quarter of double-digit growth."
    stored = SubClaim(sub_claim_id="t123", text=quote, sub_claim_type=SubClaimType.HISTORICAL_GROWTH,
                      company_ticker="GOOGL")
    out = match_claim("Google grew 24% last quarter.", [_line(quote, ticker="GOOGL", line_id="t123")], mode="rules")
    claims, match, _reasons = build_sub_claims(to_contract(out), [stored], "GOOGL", _claim)
    assert [(claim.sub_claim_id, claim.alignment) for claim in claims] == [("t123-f1", "both"),
                                                                           ("t123-f2", "quote_only")]
    assert claims[1].text == "[Alphabet] marking its twelfth consecutive quarter of double-digit growth"
    assert all(claim.parent_span == quote and claim.expansion_rule is None for claim in claims)
    assert all(claim.sub_claim_type is SubClaimType.HISTORICAL_GROWTH for claim in claims)
    assert headline_ids(claims) == ["t123-f1"]
    assert [fact.sub_claim_id for fact in match.facts] == ["t123-f1", "t123-f2"]


def test_a_line_the_user_restated_in_full_is_still_split_per_fact():
    quote = "Google grew by 24% year-over-year, marking its twelfth consecutive quarter of double-digit growth."
    stored = SubClaim(sub_claim_id="t123", text=quote, sub_claim_type=SubClaimType.HISTORICAL_GROWTH,
                      company_ticker="GOOGL")
    out = match_claim("Google grew 24% last quarter, its twelfth straight quarter of double-digit growth.",
                      [_line(quote, ticker="GOOGL", line_id="t123")], mode="rules")
    claims, _match, _reasons = build_sub_claims(to_contract(out), [stored], "GOOGL", _claim)
    # Each fact gets its own verdict: the 24% can be Supported while the twelfth quarter is Insufficient.
    assert [(claim.sub_claim_id, claim.alignment) for claim in claims] == [("t123-f1", "both"), ("t123-f2", "both")]
    assert headline_ids(claims) == ["t123-f1", "t123-f2"]


# -- pipeline: a line said after the user's cut-off ---------------------------------------------


def test_a_line_said_after_the_cutoff_is_refused_with_its_date(tmp_path, monkeypatch):
    import json

    from finsight.agents.orchestrator.graph import execute
    from finsight.contracts.schemas import AssessmentStatus

    claims = tmp_path / "claims"
    claims.mkdir()
    rows = [
        {"video_id": "early", "influencer": "joseph_carlson", "title": "Q1", "upload_date": "2026-04-13",
         "company_ticker": "META", "sub_claim_id": "early-1", "sub_claim_type": "HISTORICAL_GROWTH",
         "text": "Meta's revenue has grown 22% on a trailing 12-month basis."},
        {"video_id": "later", "influencer": "joseph_carlson", "title": "Q1 results", "upload_date": "2026-04-30",
         "company_ticker": "META", "sub_claim_id": "later-1", "sub_claim_type": "HISTORICAL_GROWTH",
         "text": "Meta's revenue is up 33%."},
    ]
    (claims / "claim_update_v3.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    posts = tmp_path / "youtube"
    posts.mkdir()  # no transcripts: posts come from the catalog
    monkeypatch.setenv("FINSIGHT_POSTS_DIR", str(posts))
    out = execute("Joseph Carlson said Meta's revenue grew 33% last quarter. Cut-off: 29 Apr 2026", None)
    assert out.assessment.status is AssessmentStatus.REFUSED
    assert "said this on 30 Apr 2026, after the cut-off 29 Apr 2026" in out.assessment.refusal_reason
    checked = execute("Joseph Carlson said Meta's revenue grew 33% last quarter. Cut-off: 30 Apr 2026", None)
    assert checked.assessment.status is AssessmentStatus.COMPLETED
    assert checked.assessment.match.quote_ids == ["later-1"]
    assert checked.assessment.headline_sub_claim_ids == ["later-1"]


# -- eval scoring -----------------------------------------------------------------------------------


def test_match_and_alignment_scores_are_separate():
    from finsight.contracts.schemas import ClaimMatch, FactAlignment
    from finsight.eval.match_eval import match_correct, score_alignment
    from finsight.eval.metrics import GoldFact, GoldMatch

    gold = GoldMatch(status="MATCHED", quote_ids=["a"], acceptable_quote_ids=["b"])
    predicted = ClaimMatch(status="MATCHED", user_claim="x", quote_ids=["a", "b"], facts=[
        FactAlignment(bucket="both", quote_span="revenue grew 24%", user_span="24%"),
        FactAlignment(bucket="both", quote_span="marking its twelfth quarter"),
    ])
    assert match_correct(gold, predicted, None)
    assert not match_correct(GoldMatch(status="MATCHED", quote_ids=["a"]), predicted, None)  # b not allowed
    facts = [GoldFact(bucket="both", quote_span="revenue grew 24%", quote_id="a", user_span="24%"),
             GoldFact(bucket="quote_only", quote_span="marking its twelfth quarter", quote_id="a")]
    score = score_alignment(facts, predicted)
    assert (score.correct, score.total) == (1, 2) and "quote_only -> both" in score.misses[0]
    asked = GoldMatch(status="ASK", video_ids=["v1", "v2", "v3"])
    assert match_correct(asked, ClaimMatch(status="ASK", user_claim="x"), "Which post? v1 ... v3")
    assert not match_correct(asked, ClaimMatch(status="ASK", user_claim="x"), "Which post? v1 ... v9")


def test_the_dev_gold_set_validates_and_derives_sub_claims():
    from finsight.eval.run_eval import load_dataset, validate_datasets

    assert validate_datasets() == []
    items = {item.item_id: item for item in load_dataset("dev")}
    first = items["dev-010"]
    assert first.gold_match is not None and first.gold_match.status == "MATCHED"
    assert [g.text for g in first.gold_sub_claims] == ["Google grew by 24% year-over-year",
                                                       "marking its twelfth consecutive quarter of double-digit growth"]
    assert first.gold_overall is not None and first.gold_overall.value == "SUPPORTED"  # only the asked fact counts
    assert items["dev-001"].gold_overall.value == "SUPPORTED"  # both facts asked: S + S (streak relabelled 31 Oct)


# -- 30 Oct fixes: superlatives / estimates are their own fact; dollar amounts without "$" are checked ----


def test_a_superlative_and_an_estimate_beat_are_their_own_facts():
    line = ("Nvidia reported earnings a little over a week ago and it was their best quarter ever by far, "
            "$81.6 billion in revenue, which is up 85% year-over-year.")
    spans = [f.span for f in split_facts(line)]
    assert spans[1:] == ["$81.6 billion in revenue", "which is up 85% year-over-year"]
    spans = [f.span for f in split_facts("The company has increased their revenue by 16% year over year, "
                                         "which beat expectations.")]
    assert spans == ["The company has increased their revenue by 16% year over year", "which beat expectations"]


def test_reported_earnings_is_not_a_profit_figure():
    facts = split_facts("Nvidia reported earnings last week, $81.6 billion in revenue.")
    assert all("profit" not in fact.metrics for fact in facts)


def test_the_auditor_reads_amounts_without_a_dollar_sign():
    from finsight.agents.auditor.agent import _amount_matches, _claim_amounts

    amounts = _claim_amounts("free cash flow went from 18 billion dollars to negative 7.6 billion")
    assert [a.value for a in amounts] == [18e9, -7.6e9]
    nearly = _claim_amounts("generated nearly 10 billion in free cash flow")[0]
    assert _amount_matches(nearly, 9.46e9) and not _amount_matches(nearly, 8.8e9)
    plain = _claim_amounts("generated 10 billion in free cash flow")[0]
    assert not _amount_matches(plain, 9.46e9) and _amount_matches(plain, 9.6e9)
    assert _amount_matches(_claim_amounts("close to over 100 billion dollars")[0], 98.77e9)


def test_an_llm_pick_across_videos_asks_only_on_a_real_tie():
    strong = _line("Nvidia reported revenues of 96.2 billion, up 106% year-over-year.", video="aaaa1111",
                   ticker="NVDA")
    weak = _line("Nvidia's revenue was strong this quarter.", video="bbbb2222", ticker="NVDA", when="2026-09-01")

    def both_videos(system: str, prompt: str) -> str:
        ids = [row.split(" ")[0] for row in prompt.splitlines() if row.startswith("L")]
        return '{"match_ids": %s, "reason": "both"}' % json.dumps(ids)

    import json

    set_mock_responder(both_videos)
    try:
        out = match_claim("Nvidia's revenue was $96.2 billion, up 106%.", [strong, weak], mode="llm")
        assert out.method == "llm" and out.status == "MATCHED" and out.video_id == "aaaa1111"
        twin = _line(strong.text, video="cccc3333", ticker="NVDA", when="2026-09-02")
        asked = match_claim("Nvidia's revenue was $96.2 billion, up 106%.", [strong, twin], mode="llm")
        assert asked.status == "ASK"
    finally:
        set_mock_responder(None)


def test_the_matcher_is_recorded_as_an_agent_call(tmp_path, monkeypatch):
    import json as _json

    from finsight.agents.orchestrator.graph import execute

    claims = tmp_path / "claims"
    claims.mkdir()
    row = {"video_id": "later", "influencer": "joseph_carlson", "title": "Q1", "upload_date": "2026-04-30",
           "company_ticker": "META", "sub_claim_id": "later-1", "sub_claim_type": "HISTORICAL_GROWTH",
           "text": "Meta's revenue is up 33%."}
    (claims / "claim_update_v3.jsonl").write_text(_json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "youtube").mkdir()
    monkeypatch.setenv("FINSIGHT_POSTS_DIR", str(tmp_path / "youtube"))
    out = execute("Joseph Carlson said Meta's revenue grew 33%. Cut-off: 30 Apr 2026", None)
    assert "claim_matcher" in [call.agent for call in out.decision_log.agent_calls]


def test_one_line_gold_accepts_an_acceptable_line_instead():
    from types import SimpleNamespace

    from finsight.eval.match_eval import match_correct

    gold = SimpleNamespace(status="MATCHED", quote_ids=["a"], acceptable_quote_ids=["b"], video_ids=[])
    pick = lambda *ids: SimpleNamespace(status="MATCHED", quote_ids=list(ids))  # noqa: E731
    assert match_correct(gold, pick("a"), None)
    assert match_correct(gold, pick("b"), None)
    assert match_correct(gold, pick("a", "b"), None)
    assert not match_correct(gold, pick("c"), None)
    assert not match_correct(gold, pick("b", "c"), None)
    two = SimpleNamespace(status="MATCHED", quote_ids=["a", "x"], acceptable_quote_ids=["b"], video_ids=[])
    assert not match_correct(two, pick("b"), None)


def test_segment_figures_go_to_the_narrative_agent():
    from finsight.agents.claim_matching.subclaims import _self_contained, _segment_phrase, segment_figure

    line = "Data center revenue came in at $5.8 billion, which grew by a much higher 57%."
    assert segment_figure("Data center revenue came in at $5.8 billion")
    assert segment_figure("which grew by a much higher 57%", line, inherits=True)  # inherits the segment
    assert not segment_figure("which grew by a much higher 57%")                    # alone: no segment
    assert segment_figure("Microsoft reported 39% Azure growth")
    assert segment_figure("Google Cloud is now making over $80 billion a year in revenue.")
    assert segment_figure("Google Cloud's backlog grew from $462 billion to $514 billion")
    assert segment_figure("AI revenue run rate was up to 37 billion")
    assert segment_figure("now has authorized to repurchase 235 billion of stock")
    # Company-wide figures stay with the financial agent, including a passing mention of a segment.
    for text in ("Google grew by 24% year-over-year", "Revenues of 82.9 billion",
                 "Their net income jumped nearly 300% to $112 billion",
                 "Gross margins fell to 68% and 66% on the cloud due to heavy AI infrastructure spend"):
        assert not segment_figure(text), text
    seg = _segment_phrase(segment_figure("which grew by a much higher 57%", line, inherits=True))
    assert _self_contained("which grew by a much higher 57%", line, "AMD", frozenset({"revenue"}), seg) \
        == "[AMD data center revenue] which grew by a much higher 57%"


def test_a_fact_with_its_own_measure_keeps_the_segment_named_earlier_in_the_line():
    from finsight.agents.claim_matching.subclaims import _earlier_segment
    line = "AWS reported $35.6 billion of revenue last quarter, up 24% year-over-year with $12.5 billion of operating income."
    assert _earlier_segment("$12.5 billion of operating income", line) is not None
    other = "Google Cloud revenue grew 30%, while Alphabet's total revenue was $96 billion."
    assert _earlier_segment("Alphabet's total revenue was $96 billion", other) is None
    first = "Nvidia reported $81.6 billion in revenue, with data center revenue of $75 billion."
    assert _earlier_segment("Nvidia reported $81.6 billion in revenue, with", first) is None
