"""A stored video keeps each company's sentence. The quote stays the post's words."""

from finsight.agents.claim_extraction.build import claims_from_transcript, expand_stored_row, specific_type


def _rows(text: str) -> list[dict]:
    return claims_from_transcript(text, video_id="vid", influencer="mark_roussin", title="t", upload_date="2026-09-01")


def test_nvidia_and_amd_are_stored_as_separate_claims():
    rows = _rows(
        "Nvidia reported revenues of 96.2 billion, up 106% year-over-year. "
        "Nvidia guided next quarter revenue to approximately 108 billion dollars. "
        "Would you rather own AMD or Nvidia? "
        "If you told me Nvidia's growth was collapsing and data center revenue was slowing, "
        "then I would worry about AMD. "
        "AMD reported 11.5 billion of revenue, 6.7 billion coming from data centers, and 107% data center growth. "
        "I'm potentially committing nearly $40,000 to AMD. "
        "NVIDIA is ahead of Google on revenue growth."
    )
    by_ticker: dict[str, list[dict]] = {}
    for row in rows:
        by_ticker.setdefault(row["company_ticker"], []).append(row)

    nvidia = by_ticker["NVDA"]
    assert nvidia[0]["text"] == "Nvidia reported revenues of 96.2 billion, up 106% year-over-year."
    assert nvidia[0]["sub_claim_type"] == "HISTORICAL_GROWTH"
    assert any(row["sub_claim_type"] == "CONTINUATION" and "108 billion" in row["text"] for row in nvidia)
    assert any(
        row["sub_claim_type"] == "COMPARATIVE" and row["text"].startswith("NVIDIA is ahead of Google")
        for row in nvidia
    )
    assert "GOOGL" not in by_ticker

    amd = by_ticker["AMD"]
    assert any(row["sub_claim_type"] == "HISTORICAL_GROWTH" and "11.5 billion" in row["text"] for row in amd)
    assert any(row["measure"] == "segment" and "data center" in row["text"] for row in amd)
    assert all("40,000" not in row["text"] for row in rows)
    assert all("would worry" not in row["text"] for row in rows)
    assert all("Would you rather" not in row["text"] for row in rows)


def test_operating_income_and_eps_growth_go_to_the_financial_agent():
    rows = _rows(
        "Nvidia's operating income shot up 124% giving them a 66.2% operating margin. "
        "Tesla's earnings per share have been in decline."
    )
    nvidia = [row for row in rows if row["company_ticker"] == "NVDA"]
    assert any(row["sub_claim_type"] == "HISTORICAL_GROWTH" and row["measure"] == "operating_income" for row in nvidia)
    assert any(row["sub_claim_type"] == "PROFITABILITY" for row in nvidia)
    tesla = [row for row in rows if row["company_ticker"] == "TSLA"]
    assert tesla[0]["sub_claim_type"] == "HISTORICAL_GROWTH"
    assert tesla[0]["measure"] == "diluted_eps"


def test_cash_formulas_the_financial_agent_can_compute_are_routed_to_it():
    rows = _rows(
        "Nvidia's operating cash flow grew 20%. "
        "Nvidia's free cash flow margin is 15%. "
        "Nvidia's cash ratio is 1.2. "
        "Nvidia's free cash flow grew."
    )
    by_text = {row["text"]: row for row in rows}
    assert by_text["Nvidia's operating cash flow grew 20%."]["sub_claim_type"] == "HISTORICAL_GROWTH"
    assert by_text["Nvidia's operating cash flow grew 20%."]["measure"] == "operating_cash_flow"
    assert by_text["Nvidia's free cash flow margin is 15%."]["sub_claim_type"] == "CASH_GENERATION"
    assert by_text["Nvidia's cash ratio is 1.2."]["sub_claim_type"] == "LIQUIDITY"
    assert by_text["Nvidia's free cash flow grew."]["sub_claim_type"] == "CASH_GENERATION"
    assert specific_type("OTHER", "the cash ratio is 1.2", None) == "LIQUIDITY"


def test_three_stated_results_become_three_claims():
    text = (
        "Revenues are at all-time highs, operating profits at all-time highs, "
        "operating margins are at 41% trailing only 2024 when margins reached 42% "
        "over the course of the past 7 years."
    )
    pieces = expand_stored_row({
        "video_id": "1-jC6oE5Ffo",
        "company_ticker": "META",
        "sub_claim_id": "1-jC6oE5Ffo-META-t21",
        "sub_claim_type": "PROFITABILITY_CASH_LEVERAGE",
        "text": text,
    })
    assert [piece["text"] for piece in pieces] == [
        "Revenues are at all-time highs",
        "operating profits at all-time highs",
        "operating margins are at 41% trailing only 2024 when margins reached 42% "
        "over the course of the past 7 years",
    ]
    assert [piece["sub_claim_type"] for piece in pieces] == [
        "HISTORICAL_GROWTH", "HISTORICAL_GROWTH", "PROFITABILITY",
    ]
    assert pieces[1]["measure"] == "operating_income"
    assert pieces[2]["measure"] == "operating_margin"
    kept = expand_stored_row({
        "sub_claim_id": "rev",
        "sub_claim_type": "HISTORICAL_GROWTH",
        "text": "Company X reported revenues of 96.2 billion, up 106% year-over-year.",
    })
    assert len(kept) == 1
    assert kept[0]["sub_claim_id"] == "rev"


def test_a_claim_with_no_formula_stays_off_the_financial_agent():
    rows = _rows(
        "Tesla's gross profit has been in decline. "
        "Google's net income fell once you exclude the investment gains. "
        "Amazon's revenue run rate is 142 billion dollars. "
        "Google's Services margin sat at 40%."
    )
    financial = {"HISTORICAL_GROWTH", "NET_INCOME_GROWTH", "PROFITABILITY", "CASH_GENERATION", "LIQUIDITY", "LEVERAGE"}
    assert rows
    assert all(row["sub_claim_type"] not in financial for row in rows)
    assert specific_type("NET_INCOME_GROWTH", "gross profit has been in decline", None) == "OTHER"
    assert specific_type("OTHER", "operating income shot up 124%", None) == "HISTORICAL_GROWTH"


def test_two_companies_with_one_measure_become_two_claims():
    text = "For context, Alphabet's operating margins are 34% and AMD's are around 17%."
    rows = _rows(text)
    by_ticker = {row["company_ticker"]: row for row in rows}
    assert set(by_ticker) == {"GOOGL", "AMD"}
    assert by_ticker["GOOGL"]["text"] == "Alphabet's operating margins are 34%"
    assert by_ticker["GOOGL"]["sub_claim_type"] == "PROFITABILITY"
    assert by_ticker["GOOGL"]["measure"] == "operating_margin"
    assert by_ticker["AMD"]["text"] == "AMD's are around 17%"
    assert by_ticker["AMD"]["sub_claim_type"] == "PROFITABILITY"
    assert by_ticker["AMD"]["measure"] == "operating_margin"
    assert by_ticker["AMD"]["parent_span"] == text

    pieces = expand_stored_row({
        "video_id": "f_BmT1yaSQY",
        "company_ticker": "GOOGL",
        "sub_claim_id": "f_BmT1yaSQY-GOOGL-t12",
        "sub_claim_type": "PROFITABILITY",
        "text": text,
    })
    assert [(piece["company_ticker"], piece["measure"], piece["text"]) for piece in pieces] == [
        ("GOOGL", "operating_margin", "Alphabet's operating margins are 34%"),
        ("AMD", "operating_margin", "AMD's are around 17%"),
    ]


def test_a_comparison_of_two_companies_stays_one_claim():
    rows = _rows("NVIDIA is ahead of Google on revenue growth.")
    assert len(rows) == 1
    assert rows[0]["company_ticker"] == "NVDA"
    assert rows[0]["sub_claim_type"] == "COMPARATIVE"


def test_a_stored_catalog_sentence_is_replaced_by_one_row_per_company():
    from finsight.agents.claim_extraction.build import apply_company_splits

    text = "For context, Alphabet's operating margins are 34% and AMD's are around 17%."
    rows, splits = apply_company_splits([{
        "video_id": "f_BmT1yaSQY",
        "company_ticker": "GOOGL",
        "judge_company": "GOOGL",
        "sub_claim_id": "f_BmT1yaSQY-GOOGL-t12",
        "sub_claim_type": "PROFITABILITY",
        "text": text,
    }])
    assert splits == 1
    assert [row["company_ticker"] for row in rows] == ["GOOGL", "AMD"]
    assert rows[1]["judge_company"] == "AMD"
    assert rows[1]["parent_span"] == text
    ebitda = (
        "So, you got a 65% EBITDA margin there for Nvidia, 55% for Broadcom, "
        "more than twice what we see with AMD at 20% and Intel at 26%."
    )
    whole, skipped = apply_company_splits([{
        "sub_claim_id": "ebitda",
        "sub_claim_type": "PROFITABILITY",
        "text": ebitda,
    }])
    assert skipped == 0
    assert len(whole) == 1
    assert whole[0]["text"] == ebitda
    kept, unchanged = apply_company_splits([{
        "sub_claim_id": "cmp",
        "sub_claim_type": "COMPARATIVE",
        "text": "NVIDIA is ahead of Google on revenue growth.",
    }])
    assert unchanged == 0
    assert len(kept) == 1


def test_estimates_and_earnings_reports_stay_off_the_financial_agent():
    rows = _rows(
        "Meta said what was not accounted for in the earnings per share estimates were "
        "$2.4 billion in costs related to legal proceedings. "
        "In this earnings report, Meta's daily active users reached 3.6 billion. "
        "Meta's earnings per share came in lower than expected due to one-time taxes."
    )
    financial = {"HISTORICAL_GROWTH", "NET_INCOME_GROWTH", "PROFITABILITY", "CASH_GENERATION", "LIQUIDITY", "LEVERAGE"}
    assert rows
    assert all(row["sub_claim_type"] not in financial for row in rows)
    # Rows stored before this rule are re-typed when the orchestrator loads them.
    assert specific_type("NET_INCOME_GROWTH", "In this earnings report, daily active users reached 3.6 billion",
                         None) == "OTHER"
    assert specific_type("OTHER", "the earnings per share estimates missed $2.4 billion in legal costs",
                         None) == "OTHER"
    assert specific_type("OTHER", "earnings per share grew 20%", None) == "HISTORICAL_GROWTH"
    assert _rows("Meta's earnings grew 30% last year.")[0]["sub_claim_type"] == "NET_INCOME_GROWTH"
