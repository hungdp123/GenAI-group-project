"""Trial the child-claim design on one segment. Not wired into the pipeline.

The model proposes children. This script then keeps a child only when the quote,
the figures, and any inherited citation are actually in the segment.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from finsight.llm.client import complete, data_block

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _norm(text: str) -> str:
    return _WS.sub(" ", text or "").strip().casefold()


@dataclass
class Slot:
    value: str | None
    origin: str
    source: str | None = None


@dataclass
class Figure:
    value: float
    raw: str
    unit: str | None
    qualifier: str | None = None
    origin: str = "explicit"
    source: str | None = None


@dataclass
class Child:
    quote: str
    subject: Slot
    measure: Slot
    figures: list[Figure] = field(default_factory=list)
    modality: str = "reported"
    other_companies: list[str] = field(default_factory=list)
    speaker: str = "host"
    problems: list[str] = field(default_factory=list)

    @property
    def kept(self) -> bool:
        return not self.problems


def _in_segment(fragment: str, segment: str) -> bool:
    return bool(fragment) and _norm(fragment) in _norm(segment)


def _number_in(raw: str, quote: str) -> bool:
    compact = _norm(quote).replace(",", "")
    token = raw.replace(",", "").strip()
    return token in compact


def rounding_tolerance(raw: str) -> float:
    """Half the last written place. '30' allows 0.5. '1.8' allows 0.05. '19' does not meet '30'."""
    token = raw.replace(",", "").strip()
    if "." in token:
        places = len(token.split(".", 1)[1])
        return 0.5 * 10 ** (-places)
    return 0.5


def figures_pair(left: Figure, right: Figure) -> bool:
    if (left.unit or "") != (right.unit or ""):
        return False
    allowance = max(rounding_tolerance(left.raw), rounding_tolerance(right.raw))
    return abs(left.value - right.value) <= allowance


def ground(segment: str, child: Child) -> Child:
    """Record why a proposed child cannot be kept. A clean child is unchanged."""
    problems: list[str] = []
    if not _in_segment(child.quote, segment):
        problems.append("quote is not verbatim in the segment")
    for figure in child.figures:
        if figure.origin == "explicit" and not _number_in(figure.raw, child.quote):
            problems.append(f"figure {figure.raw} is not in the quote")
        if figure.origin == "inherited" and not _in_segment(figure.source or "", segment):
            problems.append(f"figure {figure.raw} cites a source that is not in the segment")
    if child.measure.origin == "inherited":
        source = child.measure.source or ""
        if not _in_segment(source, segment):
            problems.append("inherited measure cites a source that is not in the segment")
        elif child.measure.value and _norm(child.measure.value) not in _norm(source):
            # The citation has to carry the measure words, not only the model’s label.
            cue = child.measure.value.split()[-1]
            if _norm(cue) not in _norm(source):
                problems.append("inherited measure is not written in the cited sentence")
    if child.measure.origin == "unresolved" and child.figures:
        problems.append("a figure with an unresolved measure is not a matchable claim")
    units = {figure.unit for figure in child.figures if figure.unit}
    if len(units) > 1:
        problems.append(f"figures disagree on unit: {sorted(units)}")
    if child.modality in {"hypothetical", "prediction", "derived"}:
        problems.append(f"modality {child.modality} cannot match a reported claim")
    child.problems = problems
    return child


_SYSTEM = """You split one transcript segment into child claims. Reply with JSON only, no markdown.
Shape: {"children": [ ... ]}.
Each child has these fields:
quote, subject {name, origin, source}, other_companies [name],
measure {name, origin, source}, figures [{value, raw, unit, qualifier, origin, source}],
modality, speaker.
origin is explicit, inherited, or unresolved.
An inherited slot’s source is the verbatim sentence the value comes from, copied from the segment.
The quote is verbatim words from the segment.
A number whose own quote does not name the measure inherits that measure from an earlier sentence in this segment.
Ford PE means forward P/E. A bare 19 or 30 next to a forward P/E is a P/E multiple, not dollars and not a percent.
If, would, and "they'd have to" are hypothetical or derived, not reported.
Do not emit a company the speaker refuses to discuss.
One child per company, measure, and figure."""


def _slot(raw: dict | None) -> Slot:
    raw = raw or {}
    return Slot(raw.get("name") or raw.get("value"), str(raw.get("origin") or "unresolved"), raw.get("source"))


def parse_children(payload: str) -> list[Child]:
    text = payload.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    data = json.loads(text)
    children = []
    for item in data.get("children") or []:
        figures = []
        for figure in item.get("figures") or []:
            figures.append(Figure(
                value=float(figure["value"]),
                raw=str(figure.get("raw") or figure["value"]),
                unit=figure.get("unit"),
                qualifier=figure.get("qualifier"),
                origin=str(figure.get("origin") or "explicit"),
                source=figure.get("source"),
            ))
        children.append(Child(
            quote=str(item.get("quote") or ""),
            subject=_slot(item.get("subject")),
            measure=_slot(item.get("measure")),
            figures=figures,
            modality=str(item.get("modality") or "reported"),
            other_companies=[str(name) for name in item.get("other_companies") or []],
            speaker=str(item.get("speaker") or "host"),
        ))
    return children


def segment_for_example() -> str:
    import json as _json
    from pathlib import Path

    root = Path("data/influencers/youtube")
    chosen = ""
    for path in root.rglob("*.jsonl"):
        if path.name == "missing_transcripts.jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if "SMFV5A8aq0c" not in line:
                continue
            row = _json.loads(line)
            if row.get("video_id") == "SMFV5A8aq0c" and len(row.get("transcript") or "") > len(chosen):
                chosen = row["transcript"]
    start = chosen.lower().find("and the first one is valuation")
    end = chosen.lower().find("the next thing is that i just believe")
    if start < 0 or end < 0:
        raise SystemExit("example segment not found")
    return chosen[start:end].strip()


def main() -> None:
    segment = segment_for_example()
    response = complete(
        data_block("segment", segment),
        system=_SYSTEM,
        max_tokens=4000,
        temperature=0.0,
    )
    print(f"model {response.model} cost ${response.cost_usd:.4f} latency {response.latency_ms:.0f} ms")
    children = [ground(segment, child) for child in parse_children(response.text)]
    reported = [child for child in children if child.kept]
    print(f"proposed {len(children)} kept {len(reported)}")
    for child in children:
        figures = ", ".join(f"{figure.raw} {figure.unit or ''}".strip() for figure in child.figures)
        status = "KEEP" if child.kept else "DROP"
        print(f"  {status} {child.subject.value} {child.measure.value} [{child.measure.origin}] {figures} {child.modality}")
        print(f"       {child.quote[:160]}")
        if child.measure.source:
            print(f"       source: {child.measure.source[:140]}")
        for problem in child.problems:
            print(f"       - {problem}")
    # The example’s two reported figures must not pair with each other.
    kept_figures = [figure for child in reported for figure in child.figures]
    if len(kept_figures) >= 2:
        paired = figures_pair(kept_figures[0], kept_figures[1])
        print(f"first two kept figures pair: {paired} ({kept_figures[0].raw} vs {kept_figures[1].raw})")


if __name__ == "__main__":
    main()
