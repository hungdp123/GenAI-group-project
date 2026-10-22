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
