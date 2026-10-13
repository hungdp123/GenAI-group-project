"""Tesla, TSLA, and Tesla, Inc. are one company. A product name maps to the parent."""

from finsight.agents.claim_extraction.prepare import prepare_post
from finsight.agents.claim_extraction.thesaurus import find_mentions, resolve_ticker


def _tickers(text: str) -> list[str]:
    return list(dict.fromkeys(ticker for ticker, _surface, _start in find_mentions(text)))


def test_tesla_spellings_are_one_ticker():
    for spelling in ("Tesla", "TESLA", "TSLA", "Tesla Inc", "Tesla, Inc.", "Tesla Motors"):
        assert resolve_ticker(f"{spelling} grew revenue.") == "TSLA"


def test_alphabet_tickers_and_a_product():
    assert resolve_ticker("GOOG and GOOGL are the same as Alphabet and Google.") == "GOOGL"
    assert resolve_ticker("AWS revenue grew.") == "AMZN"
    assert resolve_ticker("Azure margins expanded.") == "MSFT"
    assert resolve_ticker("Instagram revenue grew.") == "META"
    assert _tickers("Subscribe on YouTube.") == []
    assert resolve_ticker("YouTube ad revenue grew.") == "GOOGL"
    assert _tickers("See https://amazon.com/shop for the link. NVIDIA grew revenue.") == ["NVDA"]


def test_the_subject_is_not_the_name_after_than():
    assert resolve_ticker("NVIDIA is ahead of Google on revenue growth.") == "NVDA"


def test_extraction_drops_and_holds_the_agreed_sentences():
    text = (
        "Tesla revenue grew. "
        "Analysts say Tesla will double. "
        "What about the chart? "
        "If demand continued, Tesla would be bigger. "
        "Just kidding. "
        "Apple, Microsoft, and Netflix are all fine. "
        "They expanded margins. "
        "AWS grew 17%. "
        "The AWS segment grew, so the company grew 17%. "
        "Revenue was $1O0 billion. "
        "I take that back."
    )
    prepared = prepare_post(text, "AMZN")
    reasons = [reason for reason, _quote in prepared.dropped]
    assert "not the host" in reasons
    assert "question" in reasons
    assert "hypothetical" in reasons
    assert "joke" in reasons
    assert "list mention" in reasons
    assert "unclear they" in reasons
    assert "retracted" in reasons
    assert "Tesla revenue grew." not in prepared.text
    held = {claim.text: claim.sub_claim_id for claim in prepared.held}
    assert any(text.startswith("AWS grew") for text in held)
    assert any("whole" not in text and text.startswith("AWS grew") for text in held)
    company_hold = next(claim for claim in prepared.held if "the company grew" in claim.text)
    assert prepared.insufficient_reasons[company_hold.sub_claim_id].startswith("A segment figure")
    assert not any("$1O0" in claim.text for claim in prepared.held)
    assert any("$1O0" in quote for _reason, quote in prepared.dropped)


def test_a_garbled_number_is_insufficient_evidence():
    prepared = prepare_post("Revenue was $1O0 billion.", "AMZN")
    claim = prepared.held[0]
    assert prepared.insufficient_reasons[claim.sub_claim_id].startswith("The number or ticker")


def test_a_marked_time_is_kept_and_not_invented():
    prepared = prepare_post("[02:14] AWS grew 17%. Amazon revenue grew.", "AMZN")
    timed = next(claim for claim in prepared.held if "AWS" in claim.text)
    assert timed.quote_time == "02:14"
    assert "Amazon revenue grew." in prepared.text
