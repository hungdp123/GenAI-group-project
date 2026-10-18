"""Experimental, source-preserving adaptive segmentation; no model calls.

Hard-valid complete plans alone receive soft scores. Semantic coherence is a
documented lexical proxy, not a semantic entailment or answer-quality judgment.
Canonical chunking defaults are unaffected.
"""
from __future__ import annotations

import bisect
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache


CRITERIA = ("structural_integrity", "context_completeness", "semantic_coherence", "length_efficiency")
STOP = set("a an the and or of to in for on with at is are was were be by as from that this it our we its their fiscal quarter year million billion percent".split())
FACT = re.compile(r"revenue|sales|income|margin|cash|earnings|subscribers?|active people|expenses", re.I)
FIELDS = {
    "metric": FACT,
    "unit": re.compile(r"\$|%|\b(?:million|billion|thousand|percent|USD|MM)\b", re.I),
    "period": re.compile(r"\b(?:20\d{2}|FY\s*\d{2,4}|Q[1-4]|[1-4]Q\d{2})\b", re.I),
    "basis": re.compile(r"\b(?:non[ -]?GAAP|GAAP|IFRS|constant currency)\b", re.I),
    "scope": re.compile(r"\b(?:consolidated|total|segment|worldwide|international|domestic)\b", re.I),
}


@dataclass
class AdaptiveConfig:
    token_targets: list[int] = field(default_factory=lambda: [128, 192, 256])
    boundary_policies: list[str] = field(default_factory=lambda: ["structure", "sentence", "token"])
    weights: dict[str, float] = field(default_factory=lambda: {k: 0.25 for k in CRITERIA})
    max_repeated_context_tokens: int = 64
    single_unit_coherence: float = 0.5
    structural_rewards: dict[str, float] = field(default_factory=lambda: {"paragraph_or_row": 1.0, "sentence": 0.8, "word": 0.25, "character": 0.0})

    def validate(self, limit: int) -> None:
        if not self.token_targets or any(not isinstance(v, int) or v < 16 or v > limit for v in self.token_targets):
            raise ValueError("Every token target must be an integer from 16 to the effective model limit")
        if not self.boundary_policies or any(p not in {"structure", "sentence", "token"} for p in self.boundary_policies):
            raise ValueError("Unknown boundary policy")
        if set(self.weights) != set(CRITERIA) or any(not math.isfinite(v) or v < 0 for v in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("Provide finite nonnegative weights for all four criteria, with positive total")
        if self.max_repeated_context_tokens < 0 or not 0 <= self.single_unit_coherence <= 1:
            raise ValueError("Invalid context budget or coherence convention")
        if set(self.structural_rewards) != {"paragraph_or_row", "sentence", "word", "character"} or any(not 0 <= v <= 1 for v in self.structural_rewards.values()):
            raise ValueError("Structural rewards must be in [0,1]")


@lru_cache(maxsize=16384)
def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])


def lexical_coherence(text: str, neutral: float) -> float:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    terms = [Counter(w.lower() for w in re.findall(r"[A-Za-z]{3,}", p) if w.lower() not in STOP) for p in parts if p.strip()]
    terms = [t for t in terms if t]
    if len(terms) < 2:
        return neutral
    scores = []
    for a, b in zip(terms, terms[1:]):
        denominator = math.sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))
        scores.append(sum(v*b.get(k, 0) for k, v in a.items()) / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def context_spans(block) -> list[list[int]]:
    text = block.text
    spans = []
    if text.startswith(block.section) and block.section not in {"Document text", "Front matter"}:
        spans.append([0, len(block.section)])
    if block.content_type in {"table", "chart"}:
        title = re.search(r"(?m)^### Chart:.*$", text)
        if title:
            spans.append([title.start(), title.end()])
        lines = list(re.finditer(r"(?m)^\|.*\|[ \t]*$", text))
        if lines:
            spans.append([lines[0].start(), lines[min(1, len(lines)-1)].end()])
        # Repeated context is copied literally, never inferred from publication date.
        for m in re.finditer(r"(?im)^(?:Unit scale:|Axis labels:|Period order:|Legend/series:).*$", text):
            spans.append([m.start(), m.end()])
    return sorted(set(map(tuple, spans)))


def build_candidate(block, tokenizer, target: int, policy: str, config: AdaptiveConfig, offsets: list[tuple[int, int]], required_context: list) -> list[dict]:
    text = block.text
    ends = [z for a, z in offsets if z > a]
    paragraph = {m.end() for m in re.finditer(r"\n\s*\n", text)}
    sentence = {m.end() for m in re.finditer(r"(?<=[.!?])\s+", text)}
    row = {m.end() for m in re.finditer(r"(?m)^\|.*\|[ \t]*\n?", text)}
    natural = sorted(paragraph | sentence | (row if block.content_type in {"table", "chart"} else set()))
    if policy == "sentence":
        natural = sorted(sentence | row)
    pieces, start = [], 0
    while start < len(text):
        available_context, context_tokens = [], 0
        if start:
            for a, z in required_context:
                if a >= start:
                    continue
                prefix = "\n\n".join(text[x:y] for x, y in available_context + [[a, z]])
                cost = count_tokens(tokenizer, prefix) - tokenizer.num_special_tokens_to_add(pair=False)
                if cost <= min(config.max_repeated_context_tokens, target // 2):
                    available_context.append([a, z])
                    context_tokens = cost
        prefix = "\n\n".join(text[a:z] for a, z in available_context)
        prefix = prefix + "\n\n" if prefix else ""
        body_budget = max(1, target - context_tokens - tokenizer.num_special_tokens_to_add(pair=False) - 3)
        token_start = bisect.bisect_right(ends, start)
        upper = ends[token_start + body_budget - 1] if ends and token_start + body_budget < len(ends) else len(text)
        upper = max(start + 1, min(upper, len(text)))
        if policy != "token":
            choices = natural[bisect.bisect_right(natural, start):bisect.bisect_right(natural, upper)]
            if choices:
                # Prefer a boundary near the capacity; structure candidates give
                # row/paragraph boundaries precedence when not very short.
                if policy == "structure":
                    preferred = [v for v in choices if v in paragraph | row and v >= start + (upper-start) * 0.5]
                    upper = preferred[-1] if preferred else choices[-1]
                else:
                    upper = choices[-1]
        # A split token can re-tokenize differently. Check the actual final input,
        # including repeated context; shrink by source characters if necessary.
        end = upper
        if count_tokens(tokenizer, prefix + text[start:upper]) > target:
            low, high = start + 1, upper
            while low < high:
                middle = (low + high + 1) // 2
                if count_tokens(tokenizer, prefix + text[start:middle]) <= target:
                    low = middle
                else:
                    high = middle - 1
            end = low
        while end > start + 1 and count_tokens(tokenizer, prefix + text[start:end]) > target:
            end -= 1
        # Avoid gratuitous midword cuts when a word boundary remains available.
        if end < len(text) and not text[end-1:end].isspace() and not text[end:end+1].isspace():
            word_ends = [m.start() for m in re.finditer(r"\s+", text[start:end])]
            if word_ends and word_ends[-1] > (end-start) * 0.5:
                end = start + word_ends[-1]
        chunk_text = prefix + text[start:end]
        tokens = count_tokens(tokenizer, chunk_text)
        pieces.append({"body_start": start, "body_end": end, "context_spans": available_context,
                       "text": chunk_text, "tokens": tokens})
        start = end
    if not pieces:
        pieces = [{"body_start": 0, "body_end": 0, "context_spans": [], "text": "", "tokens": count_tokens(tokenizer, "")}]
    return pieces


def hard_rules(block, pieces: list[dict], limit: int) -> dict:
    cursor = 0
    coverage = True
    traceable = bool(block.document_id and block.company_ticker and block.citation_span and block.local_path)
    for p in pieces:
        coverage &= p["body_start"] == cursor and p["body_end"] >= cursor
        cursor = p["body_end"]
        expected_prefix = "\n\n".join(block.text[a:z] for a, z in p["context_spans"])
        expected = (expected_prefix + "\n\n" if expected_prefix else "") + block.text[p["body_start"]:p["body_end"]]
        traceable &= p["text"] == expected and all(0 <= a <= z <= len(block.text) for a, z in p["context_spans"])
    return {"content_preservation": bool(coverage and cursor == len(block.text)), "source_traceability": bool(traceable),
            "token_limit": all(p["tokens"] <= limit for p in pieces)}


def score(block, pieces: list[dict], required_context: list, tokenizer, limit: int, config: AdaptiveConfig) -> tuple[dict, dict, float]:
    text = block.text
    paragraph_row = {m.end() for m in re.finditer(r"\n\s*\n|(?m:^\|.*\|[ \t]*\n?)", text)}
    sentences = {m.end() for m in re.finditer(r"(?<=[.!?])\s+", text)}
    boundaries = []
    context_scores, coherence = [], []
    for p in pieces:
        end = p["body_end"]
        if end < len(text):
            kind = "paragraph_or_row" if end in paragraph_row else "sentence" if end in sentences else "word" if text[end-1:end].isspace() or text[end:end+1].isspace() else "character"
            boundaries.append(config.structural_rewards[kind])
        body = text[p["body_start"]:end]
        coherence.append(lexical_coherence(body, config.single_unit_coherence))
        if required_context:
            intervals = p["context_spans"] + [[p["body_start"], end]]
            retained = sum(any(a <= x and z >= y for a, z in intervals) for x, y in required_context)
            context_scores.append(retained / len(required_context))
        elif FACT.search(body) and re.search(r"\d", body):
            applicable = [pattern for pattern in FIELDS.values() if pattern.search(text)]
            context_scores.append(sum(bool(pattern.search(p["text"])) for pattern in applicable) / len(applicable) if applicable else 1.0)
        else:
            context_scores.append(1.0)
    body_chars = len(text)
    repeated_chars = sum(z-a for p in pieces for a, z in p["context_spans"])
    utilization = sum(p["tokens"] for p in pieces) / (len(pieces) * limit)
    duplication = repeated_chars / max(1, body_chars + repeated_chars)
    values = {"structural_integrity": sum(boundaries)/len(boundaries) if boundaries else 1.0,
              "context_completeness": sum(context_scores)/len(context_scores),
              "semantic_coherence": sum(coherence)/len(coherence), "length_efficiency": utilization * (1-duplication)}
    features = {"internal_boundary_count": len(boundaries), "chunk_count": len(pieces), "token_utilization": utilization,
                "repeated_context_character_fraction": duplication, "required_context_span_count": len(required_context),
                "context_method": "literal header retention or source-applicable financial-field regex proxy",
                "coherence_method": "adjacent-unit bag-of-words cosine; no embedding/LLM; single unit neutral score"}
    weighted = sum(config.weights[k] * values[k] for k in CRITERIA) / sum(config.weights.values())
    return values, features, weighted


def choose_plan(block, tokenizer, limit: int, config: AdaptiveConfig) -> tuple[list[dict] | None, dict]:
    config.validate(limit)
    encoded = tokenizer(block.text, add_special_tokens=False, truncation=False, return_offsets_mapping=True, verbose=False)
    offsets = encoded["offset_mapping"]
    required_context = context_spans(block)
    candidates, unique = [], {}
    for target in config.token_targets:
        for policy in config.boundary_policies:
            if len(encoded["input_ids"]) + tokenizer.num_special_tokens_to_add(pair=False) <= target:
                pieces = [{"body_start": 0, "body_end": len(block.text), "context_spans": [], "text": block.text,
                           "tokens": count_tokens(tokenizer, block.text)}]
            else:
                pieces = build_candidate(block, tokenizer, target, policy, config, offsets, required_context)
            signature = tuple((p["body_start"], p["body_end"], tuple(map(tuple, p["context_spans"]))) for p in pieces)
            hard = hard_rules(block, pieces, limit)
            record = {"candidate_id": f"tokens_{target}_{policy}", "token_target": target, "boundary_policy": policy,
                      "hard_rules": hard, "hard_valid": all(hard.values()), "chunk_count": len(pieces),
                      "maximum_tokens": max(p["tokens"] for p in pieces), "scores": None, "features": None, "weighted_score": None}
            if record["hard_valid"]:
                if signature in unique:
                    values, features, weighted, _ = unique[signature]
                else:
                    values, features, weighted = score(block, pieces, required_context, tokenizer, limit, config)
                    unique[signature] = (values, features, weighted, pieces)
                record.update(scores=values, features=features, weighted_score=weighted)
            candidates.append((record, pieces))
    valid = [(r, p) for r, p in candidates if r["hard_valid"]]
    if not valid:
        return None, {"status": "no_hard_valid_plan", "candidates": [r for r, p in candidates], "selected_candidate": None,
                      "reason": "All candidates failed mandatory rules; source retained in frozen inputs; no silent drop."}
    record, selected = min(valid, key=lambda rp: (-rp[0]["weighted_score"], rp[0]["chunk_count"], -rp[0]["token_target"], rp[0]["candidate_id"]))
    return selected, {"status": "selected", "candidates": [r for r, p in candidates], "selected_candidate": record["candidate_id"],
                      "selected_scores": record["scores"], "selected_weighted_score": record["weighted_score"],
                      "reason": "Highest configured weighted score among hard-valid complete plans; ties: fewer chunks, larger target, candidate ID."}
