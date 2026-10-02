"""Run per-agent evaluations and write a report. Owner: Person 5 (Hung).

    finsight eval --agents all
    finsight eval --agents auditor,financial

Writes results/{timestamp}/agents/summary.md plus <agent>.json with every case.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from finsight.config import AgentRegistry, get_settings
from finsight.eval.agent_eval import auditor, decomposer, financial, guardrails, narrative
from finsight.eval.agent_eval.common import AgentReport

EVALUATORS: dict[str, Callable[[AgentRegistry], AgentReport]] = {
    "decomposer": decomposer.evaluate,
    "financial": financial.evaluate,
    "narrative": narrative.evaluate_narrative,
    "retriever": narrative.evaluate_retriever,
    "auditor": auditor.evaluate,
    "advice_boundary": guardrails.evaluate_advice_boundary,
    "tier_policy": guardrails.evaluate_tier_policy,
}


def parse_names(spec: str) -> list[str]:
    names = list(EVALUATORS) if spec.strip().lower() == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in EVALUATORS]
    if unknown:
        raise ValueError(f"unknown agent(s) {unknown}; choose from {', '.join(EVALUATORS)} or 'all'")
    return names


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.0%}" if 0 <= value <= 1 else f"{value:.3f}"
    return str(value)


def render_report(report: AgentReport, max_failures: int = 8) -> str:
    lines = [f"## {report.agent} ({report.owner})", "", f"Implementation: `{report.implementation}`", "",
             "| Metric | Value |", "|---|---|"]
    lines += [f"| {name} | {_fmt(value)} |" for name, value in report.metrics.items()]
    failures = report.failures
    lines += ["", f"Cases passed: {len(report.cases) - len(failures)}/{len(report.cases)}"]
    if failures:
        lines.append("")
        lines += [f"- FAIL `{c.case_id}`: {c.detail}" for c in failures[:max_failures]]
        if len(failures) > max_failures:
            lines.append(f"- ... {len(failures) - max_failures} more in `{report.agent}.json`")
    lines += [f"\n_{note}_" for note in report.notes]
    return "\n".join(lines)


def run_agent_evals(names: list[str], out_root: Path = Path("results"),
                    registry: AgentRegistry | None = None) -> Path:
    registry = registry or AgentRegistry.from_config()
    out_dir = out_root / datetime.now().strftime("%Y%m%d-%H%M%S") / "agents"
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    sections = [f"# FinSight per-agent evaluation\n\nLLM mode: {settings.llm_provider} ({settings.llm_model}). "
                "Each agent is the implementation registered in `finsight/config.py`; stubs score low on "
                "purpose where they do no real checking.\n"]
    overview = ["| Agent | Owner | Cases passed |", "|---|---|---|"]
    for name in names:
        report = EVALUATORS[name](registry)
        (out_dir / f"{name}.json").write_text(report.to_json(), encoding="utf-8")
        overview.append(f"| {name} | {report.owner} | {len(report.cases) - len(report.failures)}/{len(report.cases)} |")
        sections.append(render_report(report))
    (out_dir / "summary.md").write_text("\n".join([sections[0], *overview, "", *sections[1:]]) + "\n",
                                        encoding="utf-8")
    return out_dir
