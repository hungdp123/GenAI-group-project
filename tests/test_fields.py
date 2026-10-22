"""Field comparison, company stretch, and the one-time label cache."""

import json

from finsight.agents.claim_extraction.build import claims_catalog_path
from finsight.agents.claim_extraction.fields import (
    company_for_sentences,
    fields_match,
    parse_fields,
)
from finsight.agents.claim_extraction.ingest import build_updated_claims, label_unresolved, paid_completer


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
