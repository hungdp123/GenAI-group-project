"""A stored video keeps each company's sentence. The quote stays the post's words."""

from finsight.agents.claim_extraction.build import claims_from_transcript


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
