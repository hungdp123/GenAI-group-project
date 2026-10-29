"""Contextual passage and part-search checks."""

from datetime import date

from finsight.agents.narrative.corpus.processing.contextual import (
    context_prefix,
    contextualize_chunk,
    figure_phrases,
    followup_query,
    passage_matches_claim,
    passage_states_figures,
    single_growth_claim,
    split_claim_parts,
)
from finsight.agents.narrative.corpus.processing.index import search_index
from finsight.agents.narrative.rag.collect_evidence import expand_filing_query


def _chunk(**overrides):
    record = {
        "chunk_id": "AMZN-doc:section:001:abc",
        "document_id": "AMZN-doc",
        "company_ticker": "AMZN",
        "company_name": "AMAZON COM INC",
        "document_type": "earnings_release",
        "publication_date": "2026-04-30",
        "period": "2026-03-31",
        "source_tier": "TIER_2",
        "source_url": "https://example.test/amzn",
        "section": "AWS",
        "page": None,
        "citation_span": "AWS",
        "content_type": "table",
        "text": "AWS segment sales increased 28% year over year to $35.6 billion.",
    }
    record.update(overrides)
    return record


def test_prefix_keeps_period_section_and_content_type_distinct():
    prefix = context_prefix(_chunk())
    assert "AMAZON COM INC" in prefix
    assert "(AMZN)" in prefix
    assert "earnings_release" in prefix
    assert "fiscal period 2026-03-31" in prefix
    assert "section AWS" in prefix


def test_generic_section_is_left_out_of_the_prefix():
    prefix = context_prefix(_chunk(section="Document text", content_type="narrative"))
    assert "section" not in prefix
    assert "fiscal period 2026-03-31" in prefix


def test_search_text_carries_the_prefix_and_source_text_does_not():
    records = contextualize_chunk(_chunk(), count_tokens=lambda text: len(text.split()))
    assert len(records) == 1
    assert records[0]["text"] == _chunk()["text"]
    assert records[0]["search_text"].startswith("AMAZON COM INC")
    assert "section AWS" in records[0]["search_text"]
    assert not records[0]["text"].startswith("AMAZON COM INC")


def test_long_passage_is_split_inside_the_token_budget():
    sentence = "North America sales increased 13 percent. "
    records = contextualize_chunk(
        _chunk(text=sentence * 30, content_type="narrative", section="Document text"),
        token_limit=40,
        count_tokens=lambda text: len(text.split()),
    )
    assert len(records) > 1
    assert all(len(record["search_text"].split()) <= 40 for record in records)
    assert "North America" in " ".join(record["text"] for record in records)


def test_markdown_table_header_is_repeated_on_each_window():
    table = "\n".join(
        [
            "| Segment | Growth |",
            "| --- | --- |",
            "| AWS | 28% |",
            "| Stores | 11% |",
            "| Ads | 22% |",
        ]
    )
    records = contextualize_chunk(
        _chunk(text=table),
        token_limit=26,
        count_tokens=lambda text: len(text.split()),
    )
    assert len(records) > 1
    assert all(record["text"].splitlines()[0] == "| Segment | Growth |" for record in records)


def test_a_growth_row_keeps_its_segment_and_metric_apart_from_the_company_total():
    table = (
        "Year-over-year Percentage Growth: "
        "North America — 2025: 8%; 2026: 12% "
        "AWS — 2025: 17; 2026: 28 "
        "Consolidated — 2025: 9; 2026: 17"
    )
    records = contextualize_chunk(_chunk(text=table, content_type="narrative", section="Document text"))
    aws = next(record for record in records if record["text"].startswith("Year-over-year Percentage Growth: AWS"))
    total = next(record for record in records if "Consolidated —" in record["text"])
    assert "segment AWS" in aws["search_text"]
    assert "metric Year-over-year Percentage Growth" in aws["search_text"]
    assert "2026: 28" in aws["text"]
    assert "North America" not in aws["text"]
    assert "segment Consolidated" in total["search_text"]
    assert "2026: 17" in total["text"]
    assert "2026: 28" not in total["text"]
    record = contextualize_chunk(
        _chunk(text="segment sales increased 13 percent to the stated total."),
        count_tokens=lambda text: 1,
    )[0]
    payload = {"chunks": [record]}
    hits = search_index(payload, "AWS earnings_release", 3, date(2026, 5, 1), "AMZN")
    assert hits
    assert hits[0]["text"] == record["text"]
    assert search_index(payload, "AWS earnings_release", 3, date(2026, 5, 1), "MSFT") == []


def test_claim_parts_split_only_when_each_side_has_a_figure():
    assert split_claim_parts("sales and marketing spending grew 39%") == [
        "sales and marketing spending grew 39%"
    ]
    assert split_claim_parts("AWS grew 24% in Q4 2025 with a $142 billion revenue run rate.") == [
        "AWS grew 24% in Q4 2025",
        "a $142 billion revenue run rate.",
    ]
    assert figure_phrases("AWS grew 24% in Q4 2025 with a $142 billion revenue run rate.") == [
        "24%",
        "142 billion",
    ]


def test_a_context_blurb_is_prepended_without_changing_the_source_quote():
    from finsight.agents.narrative.corpus.processing.chunk_context import prepend_blurb, write_blurb

    chunk = _chunk(text="AWS — Q1 2026: $37,587; Y/Y % Change: 28%")
    chunk["search_text"] = "segment AWS metric growth\n" + chunk["text"]
    blurb = write_blurb(chunk, lambda prompt: "Amazon Q1 2026 earnings release, AWS net sales growth.")
    updated = prepend_blurb(chunk, blurb)
    assert updated["text"] == chunk["text"]
    assert updated["search_text"].startswith("Amazon Q1 2026 earnings release, AWS net sales growth.")
    assert "segment AWS" in updated["search_text"]
    assert "28%" in prompt_of(chunk)


def prompt_of(chunk):
    from finsight.agents.narrative.corpus.processing.chunk_context import blurb_prompt
    return blurb_prompt(chunk)


def test_a_percent_token_does_not_match_a_page_number_or_a_dollar_amount():
    from finsight.agents.narrative.corpus.processing.index import retrieval_tokens, tokenize

    assert "28%" in tokenize("AWS grew 28% in Q1 2026")
    assert "28%" not in tokenize("28 Table of Contents and $28,786")
    growth = retrieval_tokens("Y/Y % Change: 28%")
    claim = retrieval_tokens("AWS grew 28%")
    assert "tag_metric_growth" in growth
    assert "tag_metric_growth" in claim
    assert "tag_segment_aws" in claim


def test_a_growth_claim_keeps_only_the_matching_quarter_and_percent():
    assert single_growth_claim("Amazon's AWS grew 28% in Q1 2026.")
    assert passage_matches_claim(
        "Amazon's AWS grew 28% in Q1 2026.",
        "AWS — Q1 2026: $37,587; Y/Y % Change: 28%",
    )
    assert not passage_matches_claim(
        "Amazon's AWS grew 28% in Q1 2026.",
        "28 Table of Contents International sales increased 19% in Q1 2026",
    )
    assert not passage_matches_claim(
        "Amazon's AWS grew 28% in Q1 2026.",
        "AWS — Q4 2025: $35,579; Y/Y % Change: 24%",
    )
    claim = "AWS grew 24% year over year."
    assert passage_states_figures(claim, "AWS segment sales increased 24% year over year.")
    assert figure_phrases("$142 billion") == ["142 billion"]
    follow = followup_query(claim, ["operating income increased"], ["10-Q"])
    assert follow is not None
    assert "earnings release" in follow
    assert followup_query(claim, ["AWS grew 24%"], ["10-Q"]) is None


def test_filing_query_adds_metric_figure_and_period_words():
    expanded = expand_filing_query(
        "[SYNTHETIC FALSE-PREMISE VARIANT] AWS sales grew 28% in Q1 2026."
    )
    assert "SYNTHETIC" not in expanded
    assert "grew" in expanded
    assert "increased" in expanded
    assert "revenue" in expanded
    assert "28%" in expanded
    assert "first" in expanded and "quarter" in expanded
