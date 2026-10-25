"""Field comparison, company stretch, and the one-time label cache."""

import json

from finsight.agents.claim_extraction.build import claims_catalog_path
from finsight.agents.claim_extraction.fields import (
    company_for_sentences,
    fields_match,
    parse_fields,
)
from finsight.agents.claim_extraction.ingest import (
    _context_span,
    _judge_batch,
    apply_block_label,
    apply_judge_label,
    block_labels,
    labels_differ,
    build_updated_claims,
    figure_without_measure,
    label_bare_context,
    label_figure_measures,
    label_unresolved,
    needs_context_label,
    paid_completer,
)


def test_a_stated_24_percent_does_not_match_17_or_23():
    prompt = parse_fields("Google revenue grew 24%")
    assert fields_match(prompt, parse_fields("search revenue increased by 17%")) is False
    assert fields_match(prompt, parse_fields("23% revenue growth")) is False
    assert fields_match(prompt, parse_fields("revenue grew by 24.2% year-over-year")) is True


def test_a_large_percentage_keeps_a_wider_band():
    prompt = parse_fields("data center growth of over 100%")
    assert fields_match(prompt, parse_fields("107% data center growth")) is True


def test_every_stated_metric_has_to_be_on_the_line():
    prompt = parse_fields(
        "revenue, operating profit and operating margin all went up", company="META",
    )
    fragment = parse_fields("How about operating profits?", company="META")
    full = parse_fields(
        "higher revenues, higher operating profits, and bigger operating margins", company="META",
    )
    assert fields_match(prompt, fragment) is False
    assert fields_match(prompt, full) is True


def test_a_line_with_the_figure_and_no_metric_can_still_match():
    prompt = parse_fields("Google revenue grew 24%")
    spoken = parse_fields("grew by 24% year-over-year, marking its twelfth consecutive quarter")
    assert fields_match(prompt, spoken) is True


def test_a_prompt_with_no_fields_matches_nothing():
    assert fields_match(parse_fields("opened a factory in Ohio"), parse_fields("margins are expanding")) is False


def test_an_unknown_ticker_ends_the_company_stretch():
    companies = company_for_sentences([
        "Shares of Meta are down.",
        "Celsius, stock ticker CELH, grew revenue.",
        "Revenue grew 140% with operating margins of 18%.",
    ])
    assert companies == ["META", None, None]


def test_marc_benioff_is_salesforce():
    assert company_for_sentences(["Marc Benioff has also announced a new platform."]) == ["CRM"]


def test_a_paid_label_fills_a_company_the_rules_missed():
    rows = [{
        "text": "Revenue grew 10%.",
        "fields": {"company": None, "metrics": ["revenue"], "values": [{"value": 10, "unit": "percent"}]},
    }]
    labeled = label_unresolved(rows, lambda _prompt: "1 GOOGL")
    assert labeled[0]["fields"]["company"] == "GOOGL"
    assert labeled[0]["company_source"] == "llm"
    assert paid_completer("mock") is None


def test_a_bare_figure_takes_the_measure_from_the_lines_around_it():
    rows = [
        {
            "sub_claim_type": "OTHER",
            "text": "In fact, 22% this most recent quarter.",
            "fields": {"company": "GOOGL", "metrics": [], "values": [{"value": 22, "unit": "percent"}]},
        },
        {
            "sub_claim_type": "HISTORICAL_GROWTH",
            "text": "Google revenue grew by 17.5%.",
            "fields": {"metrics": ["revenue"]},
        },
    ]
    seen: list[str] = []

    def completer(prompt: str) -> str:
        seen.append(prompt)
        return "1 revenue"

    labeled = label_figure_measures(
        rows,
        completer,
        lambda row: "Before: Google revenue grew by 17.5%. Sentence: " + row["text"],
    )
    assert "22%" in seen[0]
    assert "Google revenue grew" not in seen[0].split("Sentence:", 1)[-1].split("\n", 1)[0]
    assert labeled[0]["sub_claim_type"] == "HISTORICAL_GROWTH"
    assert labeled[0]["inherited_measure"] == "revenue"
    assert "revenue" in labeled[0]["fields"]["metrics"]
    assert labeled[0]["measure_source"] == "llm"
    assert labeled[1]["sub_claim_type"] == "HISTORICAL_GROWTH"
    assert "measure_source" not in labeled[1]


def test_a_price_figure_stays_other():
    rows = [{"sub_claim_type": "OTHER", "text": "The stock is up 25%.", "fields": {"metrics": []}}]
    labeled = label_figure_measures(rows, lambda _prompt: "1 NONE")
    assert labeled[0]["sub_claim_type"] == "OTHER"
    assert labeled[0]["inherited_measure"] is None
    assert labeled[0]["measure_source"] == "llm"


def test_a_debt_amount_stays_other():
    rows = [{"sub_claim_type": "OTHER", "text": "That is $12 billion.", "fields": {"metrics": []}}]
    labeled = label_figure_measures(rows, lambda _prompt: "1 debt")
    assert labeled[0]["sub_claim_type"] == "OTHER"
    assert labeled[0]["inherited_measure"] == "debt"
    assert "debt" in labeled[0]["fields"]["metrics"]


def test_context_reaches_back_to_the_named_measure():
    sentences = [
        "Google grew its revenue by 17.5%.",
        "So, we're looking at accelerated growth from Google.",
        "Overall, AI is making Google grow.",
        "Now, how is AI making Google grow?",
        "Well, through a lot of various ways.",
        "Google grew by 21%.",
        "In fact, 22% this most recent quarter.",
        "22% is massive.",
    ]
    text = _context_span(sentences, 6)
    assert "revenue by 17.5%" in text
    assert text.index("Sentence:") > text.index("revenue")


def test_an_outside_company_ends_the_carried_company_and_keeps_its_measure():
    sentences = [
        "Waymo is a tough competitor.",
        "Uber management indicated that trip growth accelerated.",
        "Uber is still gaining a larger market share.",
        "Yes, in terms of the company's total bookings, we achieved a growth of 22%.",
        "This is actually an acceleration compared to the 22% increase in the first quarter.",
        "Revenue grew by 14%, which is less than the growth in bookings.",
    ]
    labels = block_labels(sentences)
    assert labels[0][0] == "GOOGL"
    assert labels[3] == ("UBER", "bookings")
    assert labels[4] == ("UBER", "bookings")
    assert labels[5] == ("UBER", "revenue")
    row = {
        "sub_claim_id": "vid-GOOGL-t80",
        "sub_claim_type": "OTHER",
        "company_ticker": "GOOGL",
        "text": sentences[4],
        "fields": {"company": "GOOGL", "metrics": []},
    }
    apply_block_label(row, "UBER", "bookings", own_measure=None)
    assert row["company_ticker"] == "UBER"
    assert row["sub_claim_id"] == "vid-UBER-t80"
    assert row["inherited_measure"] == "bookings"
    assert row["sub_claim_type"] == "OTHER"


def test_a_bare_figure_takes_the_company_from_the_surrounding_lines():
    rows = [{
        "sub_claim_id": "vid-GOOGL-t80",
        "sub_claim_type": "OTHER",
        "company_ticker": "GOOGL",
        "text": "This is actually an acceleration compared to the 22% increase in the first quarter.",
        "fields": {"company": "GOOGL", "metrics": [], "values": [{"value": 22, "unit": "percent"}]},
    }]
    assert needs_context_label(rows[0]["text"]) is True
    labeled = label_bare_context(
        rows,
        lambda _prompt: "1. UBER, NONE",
        lambda row: "Before: Uber is still gaining share. Sentence: " + row["text"],
    )
    assert labeled[0]["company_ticker"] == "UBER"
    assert labeled[0]["sub_claim_id"] == "vid-UBER-t80"
    assert labeled[0]["sub_claim_type"] == "OTHER"
    assert labeled[0]["inherited_measure"] is None


def test_a_sentence_that_names_its_company_is_not_relabeled():
    rows = [{
        "sub_claim_id": "vid-GOOGL-t1",
        "sub_claim_type": "HISTORICAL_GROWTH",
        "company_ticker": "GOOGL",
        "text": "Google revenue grew by 17.5%.",
        "fields": {"company": "GOOGL", "metrics": ["revenue"]},
    }]
    labeled = label_bare_context(rows, lambda _prompt: "1 UBER revenue")
    assert labeled[0]["company_ticker"] == "GOOGL"
    assert labeled[0]["sub_claim_id"] == "vid-GOOGL-t1"


def test_a_sentence_that_names_revenue_is_not_sent():
    assert figure_without_measure("Google revenue grew by 17.5%.") is False
    rows = [{"sub_claim_type": "OTHER", "text": "Google revenue grew by 17.5%.", "fields": {"metrics": ["revenue"]}}]
    labeled = label_figure_measures(rows, lambda _prompt: "1 margin")
    assert labeled[0]["sub_claim_type"] == "OTHER"
    assert "measure_source" not in labeled[0]


def test_a_short_follow_on_keeps_the_measure(tmp_path):
    posts = tmp_path / "youtube"
    posts.mkdir()
    (posts / "one.jsonl").write_text(json.dumps({
        "video_id": "vid1",
        "influencer": "joseph_carlson",
        "title": "Google",
        "upload_date": "2026-07-24",
        "transcript": "Google revenue grew by 21%. In fact, 22% this most recent quarter.",
    }) + "\n", encoding="utf-8")
    claims = tmp_path / "claims"
    claims.mkdir()
    (claims / "claims.jsonl").write_text("", encoding="utf-8")
    path = build_updated_claims(posts)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    follow = next(row for row in rows if "22%" in row["text"])
    assert follow["fields"]["company"] == "GOOGL"
    assert "revenue" in follow["fields"]["metrics"]
    assert follow["extractor"] == "rules"
    assert claims_catalog_path(posts) == path


def test_the_check_keeps_the_company_and_measure_that_fit_the_sentence():
    older = {"company_ticker": "GOOGL", "measure": None, "sub_claim_type": "OTHER"}
    newer = {"company_ticker": "CLOUD", "inherited_measure": "revenue", "sub_claim_type": "HISTORICAL_GROWTH"}
    assert labels_differ(older, newer) is True
    assert labels_differ(older, {"company_ticker": "GOOGL", "measure": None}) is False
    row = {
        "sub_claim_id": "vid-CLOUD-t8",
        "company_ticker": "CLOUD",
        "sub_claim_type": "HISTORICAL_GROWTH",
        "fields": {"company": "CLOUD", "metrics": []},
    }
    apply_judge_label(row, "82%, GOOGL", "revenue", "OTHER")
    assert row["company_ticker"] == "GOOGL"
    assert row["sub_claim_id"] == "vid-GOOGL-t8"
    assert row["inherited_measure"] == "revenue"
    assert row["sub_claim_type"] == "HISTORICAL_GROWTH"
    cleared = {
        "sub_claim_id": "vid-HEY-t5",
        "company_ticker": "HEY",
        "sub_claim_type": "NET_INCOME_GROWTH",
        "fields": {"company": "HEY", "metrics": ["profit"]},
    }
    apply_judge_label(cleared, "NONE", "none", "OTHER")
    assert cleared["company_ticker"] == "NONE"
    assert cleared["inherited_measure"] is None
    assert cleared["sub_claim_type"] == "OTHER"


def test_the_check_reads_one_company_and_measure_per_line():
    items = [
        {
            "old_company": "GOOGL",
            "old_measure": None,
            "new_company": "UBER",
            "new_measure": "bookings",
            "context": "Sentence: growth of 22%.",
        },
        {
            "old_company": "NVDA",
            "old_measure": None,
            "new_company": "MICRON",
            "new_measure": "margin",
            "context": "Sentence: Gross margins came in at 84.9%.",
        },
    ]
    found = _judge_batch(
        lambda _prompt: "1. UBER, bookings\n2. MICRON margin",
        items,
    )
    assert found[1] == ("UBER", "bookings")
    assert found[2] == ("MICRON", "margin")
