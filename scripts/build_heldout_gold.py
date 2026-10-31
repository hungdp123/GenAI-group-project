"""Build finsight/eval/datasets/claims_heldout.jsonl: the held-out split, never used for tuning.

    python scripts/build_heldout_gold.py [--catalog data/influencers/claims/claim_update_v3.jsonl]

Same three gold layers as the dev set (see build_dev_gold.py and docs/claim_matching.md). Differences from dev:

  * No video used here appears in the dev set, so nothing FinSight was tuned on can be matched.
  * 20 items: 10 whose facts go to the financial agent, 10 to the narrative agent. Every line is from
    claim_update_v3.jsonl (the catalog FinSight matches against). Financial facts are labelled from data/sec/processed (XBRL, computed by hand).
    Narrative facts (segment figures, buybacks, backlog, analyst estimates) are labelled from the company's own
    earnings release or announcement, published on or before the video; the source is named in `evidence`.
    The narrative corpus was not on the drafting machine: if a release below is NOT in the corpus, the
    narrative agent cannot see it, and the right label for that fact becomes INSUFFICIENT_EVIDENCE.
  * "REVIEW:" marks a borderline call for the second labeler.

After the team has checked the labels: `finsight eval --freeze` (locks the file), then run it once with --final.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_dev_gold import CATALOG, NONE, _with_routes, ask, fact, matched  # noqa: E402

OUT = ROOT / "finsight" / "eval" / "datasets" / "claims_heldout.jsonl"
S, P, C, I = "SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"
SEC = "labelled 30 Oct 2026 from data/sec/processed (XBRL, filed on or before the video); "
REL = "labelled 30 Oct 2026 from the company's release (published on or before the video); narrative corpus to confirm. "

# --- quote ids -------------------------------------------------------------------------------------------
AMD_Q2 = "GxKra1L2TCY-AMD-t137"
INTC_REV, INTC_EPS = "DzN-IOA-png-INTC-t95", "DzN-IOA-png-INTC-t96"
GOOG_NI, GOOG_OI = "4Ry3Jv_U8I8-GOOGL-t102", "4Ry3Jv_U8I8-GOOGL-t111"
GOOG_CLOUD, GOOG_BACKLOG = "4Ry3Jv_U8I8-GOOGL-t113", "4Ry3Jv_U8I8-GOOGL-t97"
NVDA_REV_31AUG, NVDA_REV_5SEP = "f_BmT1yaSQY-NVDA-t7", "csv55UtVMZM-NVDA-t7"
NVDA_OI = "f_BmT1yaSQY-NVDA-t11-operating_income"
NVDA_DC = "csv55UtVMZM-NVDA-t10"
NVDA_NI = "bBelRFgAqE0-NVDA-t153-net_income"
NVDA_BUYBACK = "bBelRFgAqE0-NVDA-t31"
NVDA_BUYBACK_SAME = ["bBelRFgAqE0-NVDA-t154"]  # the only other line with both figures
NVDA_ANALYST = "LKboBhfuC5Q-NVDA-t34"
META_REV, META_OI, META_OM = ("UcOlnec0QfA-META-t16-revenue", "UcOlnec0QfA-META-t16-operating_income",
                              "UcOlnec0QfA-META-t16-operating_margin")
META_FCF = "UcOlnec0QfA-META-t44"
AMD_DC = "DzN-IOA-png-AMD-t49"
NVDA_DC_Q1 = "GxKra1L2TCY-NVDA-t39"
GOOG_CLOUD_RUN = "k4Y3UctGWuI-GOOGL-t77"
NVDA_TTM_REV, NVDA_TTM_OI = "bBelRFgAqE0-NVDA-t45-revenue", "bBelRFgAqE0-NVDA-t45-operating_income"
NVDA_TTM_OCF, NVDA_TTM_FCF = "bBelRFgAqE0-NVDA-t48-operating_cash_flow", "bBelRFgAqE0-NVDA-t48-free_cash_flow"
GOOG_FCF = "4Ry3Jv_U8I8-GOOGL-t133"
NVDA_STREET = "H61QlAU5Ym8-NVDA-t60"
NVDA_BEATS = "H61QlAU5Ym8-NVDA-2-diluted_eps"
EV_NVDA_TTM = ("NVDA 10-K filed 2026-02-25 + 10-Q filed 2026-08-26: trailing 12 months to 2026-07-26 = FY2026 + H1 FY2027 "
               "- H1 FY2026.")

EV_NVDA_Q2 = ("NVDA 10-Q filed 2026-08-26: Q2 FY2027 (to 2026-07-26) revenue $96,221m vs $46,743m a year earlier "
              "(+105.9%) and $81,615m the quarter before (+17.9%).")
EV_META_TTM = ("META 10-K filed 2026-01-29 + 10-Q filed 2026-04-30: trailing 12 months to 2026-03-31 = FY2025 + Q1 2026 "
               "- Q1 2025.")
NVDA_BUYBACK_EV = ("NVIDIA release 2026-09-28 ('NVIDIA Announces a $150 Billion Share Repurchase Authorization "
                   "Increase'): additional $150B; $235B remaining, to be executed through fiscal 2028.")

ITEMS: list[dict] = [
    # -- financial agent ---------------------------------------------------------------------------------
    dict(id="ho-001", category="normal", case="both",
         claim="Alex Reffett said AMD's revenue came in at $11.5 billion last quarter, up 50%.",
         match=matched(AMD_Q2),
         facts=[fact("both", "The revenue came in at $11.5 billion", AMD_Q2, "$11.5 billion", S,
                     ["AMD 10-Q filed 2026-08-05 (same day as the video): Q2 2026 revenue $11,536m."], SEC),
                fact("both", "which was up 50% year-over-year", AMD_Q2, "up 50%", S,
                     ["Same 10-Q: $11,536m vs $7,685m a year earlier (+50.1%)."], SEC)]),
    dict(id="ho-002", category="normal", case="both",
         claim="Alex Reffett said Intel's earnings per share came in at 29 cents last quarter, up from 13 cents.",
         match=matched(INTC_EPS),
         facts=[fact("both", "And earnings per share came in at 29 cents, up from 13 cents last year", INTC_EPS,
                     "29 cents, up from 13 cents", C,
                     ["INTC 10-Q filed 2026-04-24: Q1 2026 diluted EPS -$0.73 vs -$0.19 a year earlier (GAAP)."],
                     SEC + "the rule book checks GAAP (XBRL) EPS: a loss, and lower. REVIEW: 29c and 13c look like "
                     "Intel's non-GAAP EPS; under the proposed section 12 rule (decision 2) this becomes "
                     "Partially supported.")]),
    dict(id="ho-003", category="normal", case="both",
         claim="Alex Reffett said Alphabet's net income nearly quadrupled to about $112 billion last quarter.",
         match=matched(GOOG_NI),
         facts=[fact("both", "Their net income jumped nearly 300% to $112 billion", GOOG_NI,
                     "nearly quadrupled to about $112 billion", S,
                     ["GOOGL 10-Q filed 2026-07-23: Q2 2026 net income $112,193m vs $28,196m (+297.9%, 3.98x)."],
                     SEC + "paraphrase: 'nearly quadrupled' = 'up nearly 300%'.")],
         notes="Paraphrase item: the user's wording ('quadrupled') shares no growth figure with the line."),
    dict(id="ho-004", category="misremembered", case="conflict",
         claim="Alex Reffett said Alphabet's operating income grew 300% last quarter.",
         match=matched(GOOG_OI),
         facts=[fact("conflict", "And their operating income grew by 30%", GOOG_OI, "300%", S,
                     ["GOOGL 10-Q filed 2026-07-23: Q2 2026 operating income $40,770m vs $31,271m (+30.4%)."],
                     SEC + "the influencer's version (30%) is checked; the user's 300% is his 'not the 300%'.")],
         notes="Misremembered: the user took the figure the influencer rejects. An earlier Reffett video "
               "(1K0OgUBolXc, 11 Jun) also says operating income grew 30%, but for Q1; only this line mentions 300%."),
    dict(id="ho-005", category="cutoff", case="quote_only", cutoff="2026-09-02",
         claim="Alex Reffett said Nvidia's revenue was $96.2 billion, up 106%. Cut-off: 2 Sep 2026",
         match=matched(NVDA_REV_31AUG),
         facts=[fact("both", "Nvidia reported record revenues of 96.2 billion dollars for the quarter", NVDA_REV_31AUG,
                     "$96.2 billion", S, [EV_NVDA_Q2], SEC),
                fact("quote_only", "which is up 18% quarter-over-quarter", NVDA_REV_31AUG, verdict=S,
                     evidence=[EV_NVDA_Q2], rule=SEC + "+17.9% rounds to 18%."),
                fact("both", "106% year-over-year", NVDA_REV_31AUG, "up 106%", S, [EV_NVDA_Q2], SEC)],
         notes="Said in two Reffett videos (31 Aug and 5 Sep 2026); with the cut-off only the 31 Aug video counts."),
    dict(id="ho-006", category="normal", case="quote_only",
         claim="Mark Roussin said Nvidia's operating income over the past year was nearly $200 billion.",
         match=matched(NVDA_TTM_OI, acceptable=[NVDA_TTM_REV]),
         facts=[fact("quote_only", "Revenue is reaching $33 billion", NVDA_TTM_REV, verdict=C,
                     evidence=[EV_NVDA_TTM + " Revenue $215,938m + $81,615m + $96,221m - $44,062m - $46,743m = "
                               "$302,969m (latest quarter alone $96,221m)."],
                     rule=SEC + "no period gives $33B. REVIEW: probably the transcript's rendering of '$303 billion'; "
                     "the stored words are what is checked."),
                fact("both", "operating income of nearly 200 billion", NVDA_TTM_OI, "nearly $200 billion", S,
                     [EV_NVDA_TTM + " Operating income $130,387m + $53,536m + $63,734m - $21,638m - $28,440m = "
                      "$197,579m."], SEC + "period: trailing 12 months ('over the past year').")],
         notes="The headline covers only operating income (Supported); the revenue figure in the same sentence is "
               "wrong as stored (Contradicted, 'also said')."),
    dict(id="ho-007", category="normal", case="both",
         claim="Mark Roussin said Nvidia generated $134 billion of operating cash flow and $127 billion of free cash "
               "flow over the past 12 months.",
         match=matched(NVDA_TTM_OCF, NVDA_TTM_FCF),
         facts=[fact("both", "The company generated operating cash flow of 134 billion the past 12 months",
                     NVDA_TTM_OCF, "$134 billion of operating cash flow", S,
                     [EV_NVDA_TTM + " Operating cash flow FY2026 $102,718m + H1 FY2027 $74,421m - H1 FY2026 $42,779m "
                      "= $134,360m."], SEC + "the 10-Q cash flow is year-to-date (H1), so the trailing figure needs "
                     "the H1 rows."),
                fact("both", "free cash flow of 127 billion", NVDA_TTM_FCF, "$127 billion of free cash flow", S,
                     [EV_NVDA_TTM + " Capex $6,042m + $4,434m - $3,122m = $7,354m; free cash flow $127,006m."], SEC)]),
    dict(id="ho-008", category="normal", case="both",
         claim="Alex Reffett said Alphabet's free cash flow was negative $5.9 billion last quarter.",
         match=matched(GOOG_FCF),
         facts=[fact("both", "their free cash flows, which came in at negative 5.9 billion dollars for the quarter",
                     GOOG_FCF, "negative $5.9 billion", S,
                     ["GOOGL 10-Q filed 2026-07-23: H1 2026 operating cash flow $84,859m - Q1 $45,790m = Q2 $39,069m; "
                      "capex $80,598m - $35,674m = $44,924m; Q2 free cash flow -$5,855m."],
                     SEC + "the quarter is H1 minus Q1 (10-Q cash flows are year-to-date).")]),

    dict(id="ho-009", category="normal", case="both",
         claim="Mark Roussin said Nvidia had less than $5 billion of free cash flow at the start of the pandemic.",
         match=matched("bBelRFgAqE0-NVDA-2"),
         facts=[fact("both", "At the start of the pandemic, Nvidia had free cash flow of less than $5 billion",
                     "bBelRFgAqE0-NVDA-2", "less than $5 billion", I,
                     rule="Fiscal 2020-21 is before the stored filings (the SEC store starts with periods ending in "
                          "2024; the corpus holds the latest 10-K and four 10-Qs).")],
         notes="Overclaim test (financial route): a figure from before the stored window. His 25 Aug video "
               "(H61QlAU5Ym8) says '4.6 billion', a different wording, so it is not the same statement."),
    dict(id="ho-010", category="normal", case="both",
         claim="Kevin Paffrath said Tesla's sales are down 16% year over year.",
         match=matched("j0mDc-ADqFU-TSLA-t150"),
         facts=[fact("both", "Tesla sales are down 16% year-over-year", "j0mDc-ADqFU-TSLA-t150", "down 16%", C,
                     ["TSLA 10-Q filed 2026-07-23: Q2 2026 revenue $28,236m vs $22,496m (+25.5%); FY2025 $94,827m."],
                     SEC + "'sales' read as revenue, latest quarter before the 13 Sep video: up, not down. REVIEW: he is "
                     "reading a news story about the US EV market, so he may mean unit sales in one region, which "
                     "the filings do not state; then the label is Insufficient evidence. Not Supported either way.")]),
    # -- narrative agent ---------------------------------------------------------------------------------
    dict(id="ho-011", category="normal", case="both",
         claim="Alex Reffett said Google Cloud revenue was $24.8 billion last quarter, up 82%.",
         match=matched(GOOG_CLOUD),
         facts=[fact("both", "Google Cloud revenue came in at $24.8 billion for the quarter", GOOG_CLOUD,
                     "$24.8 billion", S, ["Alphabet Q2 2026 release (22 Jul 2026): Google Cloud revenue $24.77B."],
                     REL + "segment revenue is not calculated (section 11)."),
                fact("both", "which is up 82% year-over-year", GOOG_CLOUD, "up 82%", S,
                     ["Same release: Google Cloud +82% year over year."], REL)]),
    dict(id="ho-012", category="normal", case="both",
         claim="Alex Reffett said AMD's data center revenue was $5.8 billion last quarter, up 57%.",
         match=matched(AMD_DC),
         facts=[fact("both", "Data center revenue came in at $5.8 billion", AMD_DC, "$5.8 billion", S,
                     ["AMD Q1 2026 release (5 May 2026): Data Center segment revenue $5.8B."], REL),
                fact("both", "which grew by a much higher 57%", AMD_DC, "up 57%", S,
                     ["Same release: Data Center up 57% year over year."], REL)]),
    dict(id="ho-013", category="partly_found", case="user_only",
         claim="Alex Reffett said Nvidia's data center revenue was $89 billion, up 117%, and its gaming revenue "
               "doubled.",
         match=matched(NVDA_DC),
         facts=[fact("both", "Nvidia's data center revenues came in at $89 billion", NVDA_DC, "$89 billion", S,
                     ["NVIDIA Q2 FY2027 release (26 Aug 2026): Data Center revenue $89.0B."], REL),
                fact("both", "which was up 117% year-over-year", NVDA_DC, "up 117%", S,
                     ["Same release: Data Center up 117% year over year (+18% quarter over quarter)."], REL),
                fact("user_only", user_span="its gaming revenue doubled")],
         notes="The user adds a fact the influencer never said: shown as 'not said', never checked."),
    dict(id="ho-014", category="normal", case="both",
         claim="Mark Roussin said Nvidia added another $150 billion to its buyback, bringing the total to $235 "
               "billion.",
         match=matched(NVDA_BUYBACK, acceptable=NVDA_BUYBACK_SAME),
         facts=[fact("both", "Nvidia's board approved an additional $150 billion under its existing program",
                     NVDA_BUYBACK, "another $150 billion", S, [NVDA_BUYBACK_EV], REL),
                fact("both", "the company now has $235 billion available for repurchases", NVDA_BUYBACK,
                     "the total to $235 billion", S, [NVDA_BUYBACK_EV], REL)],
         notes="REVIEW: the 28 Sep announcement is not in an earnings release or 10-Q; if the narrative corpus "
               "has no news or 8-K for it, the label becomes Insufficient evidence. "
               "He says it several times in the same video; t154 also gives both figures, so it is the same statement. "
               "t5 and t8 give only one figure each."),
    dict(id="ho-015", category="normal", case="both",
         claim="Mark Roussin said the average analyst estimate has Nvidia's earnings growing about 50% a year for the "
               "next two years.",
         match=matched(NVDA_ANALYST, acceptable=["pN4IO1TNXRU-NVDA-t16"]),
         facts=[fact("both", "an average analyst EPS estimate for the company this year and next growing earnings at "
                     "an average 50% clip each of the next 2 years", NVDA_ANALYST, "about 50% a year", I,
                     rule="Analyst estimates are not calculated and are not company statements (sections 11-12); a "
                          "forecast is never Supported.")],
         notes="A similar line is in his 7 Apr video (pN4IO1TNXRU), so that line is also accepted. Overclaim test."),
    dict(id="ho-016", category="normal", case="both",
         claim="Alex Reffett said Google Cloud's backlog grew from $462 billion to $514 billion.",
         match=matched(GOOG_BACKLOG),
         facts=[fact("both", "Google Cloud's backlog grew from $462 billion to $514 billion", GOOG_BACKLOG,
                     "from $462 billion to $514 billion", S,
                     ["Alphabet Q2 2026 release (22 Jul 2026): Google Cloud backlog $514B; Q1 2026 (29 Apr 2026): "
                      "$462B."], REL + "backlog is a company statement, not an XBRL figure.")],
         notes="REVIEW: backlog is stated in the earnings release and call; check it is in the narrative corpus. "
               "His 24 Sep video (BFJ4Khk8hHw) says the backlog 'is $514 billion' but not 'grew from $462 billion', "
               "so it is not the same statement."),

    dict(id="ho-017", category="normal", case="quote_only",
         claim="Alex Reffett said Nvidia's data center revenue was over $75 billion last quarter.",
         match=matched(NVDA_DC_Q1),
         facts=[fact("both", "Nvidia reported over $75 billion in data center revenues for the quarter", NVDA_DC_Q1,
                     "over $75 billion", S, ["NVIDIA Q1 FY2027 release (20 May 2026): Data Center revenue $75.2B."],
                     REL + "'latest earnings call' before the 5 Aug video = Q1 FY2027 (Q2 was reported 26 Aug)."),
                fact("quote_only", "roughly $60 billion from compute", NVDA_DC_Q1, verdict=S,
                     evidence=["Same release: Data Center compute revenue $60.4B."], rule=REL),
                fact("quote_only", "$15 billion from networking", NVDA_DC_Q1, verdict=S,
                     evidence=["Same release: Data Center networking revenue $14.8B ('roughly')."], rule=REL)]),
    dict(id="ho-018", category="normal", case="both",
         claim="Alex Reffett said Google Cloud is now making over $80 billion a year in revenue.",
         match=matched(GOOG_CLOUD_RUN),
         facts=[fact("both", "Google Cloud is now making over $80 billion a year in revenue", GOOG_CLOUD_RUN,
                     "over $80 billion a year", S,
                     ["Alphabet Q1 2026 results (29 Apr 2026): Google Cloud revenue $20.03B, x4 = $80.1B a year."],
                     REL + "a run rate (latest quarter x 4), the usual reading of 'now making ... a year'. REVIEW: the "
                     "strict reading (last four quarters added up) is below $80B, which would be Contradicted.")]),
    dict(id="ho-019", category="normal", case="both",
         claim="Mark Roussin said Wall Street expects Nvidia to report about $104 billion of revenue next quarter.",
         match=matched(NVDA_STREET, acceptable=["H61QlAU5Ym8-NVDA-t159"]),
         facts=[fact("both", "Wall Street currently expects roughly 104 billion for the next quarter", NVDA_STREET,
                     "about $104 billion of revenue next quarter", I,
                     rule="Analyst estimates are not stored and are not company statements (sections 11-12).")],
         notes="Overclaim test: the actual next-quarter result is irrelevant; the claim is about an expectation."),
    dict(id="ho-020", category="normal", case="both",
         claim="Mark Roussin said Nvidia has beaten EPS expectations in 18 of the past 20 quarters.",
         match=matched(NVDA_BEATS),
         facts=[fact("both", "EPS expectations they've beat in 18 of the past 20 quarters", NVDA_BEATS,
                     "18 of the past 20 quarters", I,
                     rule="Needs 20 quarters of analyst estimates; estimates are not stored (section 11).")],
         notes="Overclaim test, like dev's 'twelfth consecutive quarter'."),

]


def build(catalog: Path) -> list[dict]:
    lines = {}
    with open(catalog, encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            lines[row["sub_claim_id"]] = row
    influencers = {"Joseph Carlson": "joseph_carlson", "Mark Roussin": "mark_roussin", "Alex Reffett": "alex_reffett",
                   "Kevin Paffrath": "meet_kevin"}
    out = []
    for spec in ITEMS:
        match = spec.get("match")
        for quote_id in (match or {}).get("quote_ids", []) + (match or {}).get("acceptable_quote_ids", []):
            if quote_id not in lines:
                raise SystemExit(f"{spec['id']}: {quote_id} is not in the catalog")
        for row in spec.get("facts", []):
            if row.get("quote_id") and row["quote_span"] not in lines[row["quote_id"]]["text"]:
                raise SystemExit(f"{spec['id']}: quote_span not in {row['quote_id']}: {row['quote_span']!r}")
        who = next((key for name, key in influencers.items() if name in spec["claim"]), None)
        item = {
            "item_id": spec["id"],
            "split": "heldout",
            "category": spec["category"],
            "match_case": spec.get("case"),
            "claim": spec["claim"],
            "cutoff_date": spec.get("cutoff"),
            "influencer": who,
            "expect_refusal": bool(spec.get("refuse")),
            "expect_clarification": bool(spec.get("clarify") or (match or {}).get("status") in ("ASK", "NONE")),
            "gold_match": match,
            "gold_alignment": _with_routes(spec.get("facts", []), lines),
            "notes": spec.get("notes"),
            "gold_source": "Claude draft, 30 Oct 2026, from claim_update_v3 + data/sec/processed + company releases; "
                           "written before running FinSight on these items. Needs a second labeler, then --freeze.",
        }
        if match and match.get("status") == "MATCHED":
            first = lines[match["quote_ids"][0]]
            item["posts"] = [{"influencer": first["influencer"], "video_id": first["video_id"],
                              "upload_date": first["upload_date"], "title": first["title"]}]
        out.append({key: value for key, value in item.items() if value not in (None, [], "")})
    assert len({i["item_id"] for i in out}) == len(out), "duplicate item ids"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    items = build(args.catalog)
    args.out.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
    facts = [row for item in items for row in item.get("gold_alignment", []) if row["bucket"] != "user_only"]
    routes: dict[str, int] = {}
    for row in facts:
        routes[row.get("route", "?")] = routes.get(row.get("route", "?"), 0) + 1
    print(f"wrote {len(items)} items, {len(facts)} gold facts to {args.out}; routes: {routes}")


if __name__ == "__main__":
    main()
