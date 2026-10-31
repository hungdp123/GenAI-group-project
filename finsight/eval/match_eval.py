"""Score claim matching on its own: right line, right fact buckets, right verdict per fact.

    finsight eval --matcher                    # rules (free)
    FINSIGHT_MATCHER=llm finsight eval --matcher   # the LLM pick (paid mode; costs a little)

Three numbers, kept apart so a matching error is never counted as a verdict error:

  match accuracy      MATCHED: every gold quote_id chosen (or, for a one-line gold, an acceptable line instead),
                      nothing outside quote_ids + acceptable_quote_ids.
                      ASK: the system asked, and offered at least two of the gold video_ids.
                      NONE: the system said the line is not in the stored videos.
  alignment accuracy  per gold fact, the predicted fact with the same words is in the same bucket.
                      Only facts on lines the system actually chose are compared.
  verdict accuracy    per labeled fact the user asked about, the verdict of the sub-claim that checked it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from finsight.contracts.schemas import ClaimMatch, FinalAssessment
from finsight.eval.metrics import GoldFact, GoldMatch

_WORD = re.compile(r"[a-z0-9.%$]+")


def _words(text: str | None) -> set[str]:
    return set(_WORD.findall((text or "").lower().replace("​", "")))


def _same_words(left: str | None, right: str | None) -> bool:
    """Spans name the same words: one contains the other, or they share most words."""
    a, b = _words(left), _words(right)
    if not a or not b:
        return False
    return a <= b or b <= a or len(a & b) / min(len(a), len(b)) >= 0.6


def _figure(text: str | None) -> set[str]:
    return {word.rstrip(".") for word in _words(text) if any(ch.isdigit() for ch in word)}


def match_correct(gold: GoldMatch, predicted: ClaimMatch | None, question: str | None) -> bool:
    """`predicted` is the run's ClaimMatch (any status); None when the run stopped before matching."""
    status = predicted.status if predicted is not None else "STOPPED"
    if gold.status != status:
        return False
    if gold.status == "MATCHED":
        chosen = set(predicted.quote_ids if predicted else [])
        allowed = set(gold.quote_ids) | set(gold.acceptable_quote_ids)
        if not chosen or not chosen <= allowed:
            return False
        # Every gold line chosen; or, when the gold is one line, an acceptable line that states the same thing.
        return set(gold.quote_ids) <= chosen or (len(gold.quote_ids) == 1 and bool(gold.acceptable_quote_ids))
    if gold.status == "ASK":
        offered = {vid for vid in gold.video_ids if vid in (question or "")}
        return len(offered) >= min(2, len(gold.video_ids))
    return True


@dataclass
class AlignmentScore:
    correct: int
    total: int
    misses: list[str]


def score_alignment(gold: list[GoldFact], predicted: ClaimMatch | None) -> AlignmentScore:
    if predicted is None:
        return AlignmentScore(0, 0, [])
    chosen = set(predicted.quote_ids)
    facts = list(predicted.facts)
    correct = total = 0
    misses: list[str] = []
    used: set[int] = set()
    for fact in gold:
        if fact.bucket != "user_only" and fact.quote_id not in chosen:
            continue  # the system chose an acceptable other line: not comparable fact by fact
        total += 1
        hit = None
        for index, row in enumerate(facts):
            if index in used:
                continue
            if fact.bucket == "user_only":
                same = row.bucket == "user_only" and (
                    _same_words(fact.user_span, row.user_span) or (_figure(fact.user_span) & _figure(row.user_span)))
            else:
                same = _same_words(fact.quote_span, row.quote_span) and (
                    not _figure(fact.user_span) or not row.user_span
                    or bool(_figure(fact.user_span) & _figure(row.user_span)) or row.bucket == "quote_only")
            if same:
                hit = index
                if row.bucket == fact.bucket:
                    break
        if hit is not None and facts[hit].bucket == fact.bucket:
            correct += 1
            used.add(hit)
        else:
            got = facts[hit].bucket if hit is not None else "missing"
            misses.append(f"{fact.bucket} -> {got}: {(fact.quote_span or fact.user_span or '')[:60]}")
    return AlignmentScore(correct, total, misses)


def fact_verdicts(gold: list[GoldFact], assessment: FinalAssessment) -> list[tuple[str, str | None]]:
    """(gold, predicted) for each labeled fact the user asked about, via the sub-claim that checked it."""
    match = assessment.match
    if match is None:
        return []
    verdicts = {row.sub_claim.sub_claim_id: row.verdict.value for row in assessment.sub_claim_results}
    pairs = []
    for fact in gold:
        if not fact.asked or fact.gold_verdict is None:
            continue
        row = next((r for r in match.facts if r.bucket in ("both", "conflict")
                    and _same_words(fact.quote_span, r.quote_span)), None)
        pairs.append((fact.gold_verdict.value, verdicts.get(row.sub_claim_id or "") if row else None))
    return pairs


# -- matcher-only run (no agents, no filings) ---------------------------------------------------


def run_matcher_only(items, mode: str | None = None) -> list[dict]:
    """The front of the pipeline only: influencer, company, cut-off, candidate lines, match, align."""
    from finsight.agents.claim_extraction.build import claims_catalog_path
    from finsight.agents.claim_extraction.select import stated_claim
    from finsight.agents.claim_matching.catalog import lines_for, load_catalog
    from finsight.agents.claim_matching.matcher import match_claim
    from finsight.agents.claim_matching.subclaims import to_contract
    from finsight.agents.orchestrator import posts, routing
    from finsight.config import get_settings

    settings = get_settings()
    catalog = load_catalog(claims_catalog_path(settings.posts_dir))
    rows = []
    for item in items:
        if item.gold_match is None:
            continue
        influencer = routing.detect_influencer(item.claim)
        ticker = routing.detect_company(item.claim, settings.corpus_companies)
        cutoff, _ = routing.resolve_cutoff(item.cutoff_date, item.claim, date(2026, 9, 30))
        aliases = settings.corpus_companies.get(ticker or "", [])
        lines = [line for line in lines_for(catalog, influencer=influencer, ticker=ticker, cutoff=cutoff)
                 if line.upload_date is None or posts.WINDOW_START <= line.upload_date <= posts.WINDOW_END]
        focus = stated_claim(item.claim, influencer, aliases) or item.claim
        import time

        from finsight.llm.usage import track_usage

        started = time.perf_counter()
        with track_usage() as meter:
            outcome = match_claim(focus, lines, mode=mode, ticker=ticker)
        latency_ms = (time.perf_counter() - started) * 1000
        predicted = to_contract(outcome)
        alignment = score_alignment(item.gold_alignment, predicted if outcome.status == "MATCHED" else None)
        rows.append({
            "item_id": item.item_id, "case": item.match_case, "gold": item.gold_match.status,
            "predicted": outcome.status, "method": outcome.method,
            "correct": match_correct(item.gold_match, predicted, outcome.question),
            "alignment_correct": alignment.correct, "alignment_total": alignment.total,
            "misses": alignment.misses, "chosen": predicted.quote_ids, "gold_ids": item.gold_match.quote_ids,
            "candidates": len(lines),
            "input_tokens": meter.input_tokens, "output_tokens": meter.output_tokens,
            "cost_usd": meter.cost_usd, "latency_ms": latency_ms,
        })
    return rows


def matcher_report(rows: list[dict], header: str) -> str:
    n = len(rows) or 1
    facts = sum(row["alignment_total"] for row in rows)
    lines = [header, "",
             f"- Match accuracy: {sum(row['correct'] for row in rows)}/{len(rows)} "
             f"({sum(row['correct'] for row in rows) / n:.0%})",
             f"- Alignment accuracy: {sum(row['alignment_correct'] for row in rows)}/{facts} "
             f"({(sum(row['alignment_correct'] for row in rows) / facts) if facts else 0:.0%}) facts "
             "(on lines the matcher chose)",
             f"- Cost: ${sum(row.get('cost_usd', 0) for row in rows):.4f} for {len(rows)} items "
             f"({sum(row.get('input_tokens', 0) + row.get('output_tokens', 0) for row in rows):,} tokens); "
             f"mean latency {sum(row.get('latency_ms', 0) for row in rows) / n:.0f} ms per item",
             "", "| Case | Items | Match correct | Facts right |", "|---|---|---|---|"]
    cases: dict[str, list[dict]] = {}
    for row in rows:
        cases.setdefault(row["case"] or "-", []).append(row)
    for case, group in sorted(cases.items()):
        total = sum(row["alignment_total"] for row in group)
        right = sum(row["alignment_correct"] for row in group)
        lines.append(f"| {case} | {len(group)} | {sum(row['correct'] for row in group)}/{len(group)} | "
                     f"{right}/{total} |")
    lines += ["", "| Item | Case | Gold | Predicted | Match | Facts | Notes |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        note = "; ".join(row["misses"])
        if not row["correct"] and row["predicted"] == "MATCHED":
            note = f"chose {', '.join(row['chosen'])}; gold {', '.join(row['gold_ids'])}. " + note
        lines.append(f"| {row['item_id']} | {row['case']} | {row['gold']} | {row['predicted']} ({row['method']}) | "
                     f"{'ok' if row['correct'] else 'WRONG'} | {row['alignment_correct']}/{row['alignment_total']} | "
                     f"{note.replace('|', '/')} |")
    return "\n".join(lines) + "\n"
