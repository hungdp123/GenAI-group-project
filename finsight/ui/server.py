"""Local testing server. Binds to localhost and serves the page plus a small JSON API.

    finsight ui
    finsight ui --port 8765
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from finsight.agents.orchestrator.graph import execute
from finsight.agents.orchestrator.stages import ROLES, StageReporter
from finsight.config import BOUNDARY_DISCLAIMER, COURSE_BUDGET_USD, get_settings
from finsight.contracts.schemas import DecisionLog
from finsight.llm.spend import ledger_summary
from finsight.logging.decision_log import format_trace
from finsight.ui.payload import _stats, build_view, cli_command

STATIC = Path(__file__).resolve().parent / "static"
_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()

EXAMPLES = [
    {
        "id": "advice",
        "label": "Personal advice",
        "note": "Stops at the advice gate. No evidence is gathered.",
        "claim": "Should I put my savings into Nvidia?",
    },
    {
        "id": "price",
        "label": "Price target",
        "note": "Price predictions are out of scope.",
        "claim": "Nvidia stock price will hit $500 next year.",
    },
    {
        "id": "macro",
        "label": "Market claim",
        "note": "Macro claims are refused before a company check.",
        "claim": "Will the S&P 500 hit 7000 by December?",
    },
    {
        "id": "company",
        "label": "Missing company",
        "note": "Asks which company. Answer in the box that appears.",
        "claim": "Joseph Carlson said revenue growth is just getting started.",
        "influencer": "Joseph Carlson",
    },
    {
        "id": "influencer",
        "label": "Missing influencer",
        "note": "Asks who said it. FinSight only checks stored posts.",
        "claim": "Nvidia revenue grew 20% last year.",
        "company": "NVDA",
    },
    {
        "id": "which-post",
        "label": "Which post",
        "note": "A broad question. FinSight lists stored posts and waits for a video id.",
        "claim": "Is Joseph Carlson right about Nvidia?",
    },
    {
        "id": "figure",
        "label": "Filing figure",
        "note": "A revenue sentence in the stored-post style. This one calls the live agents.",
        "claim": "Joseph Carlson said Nvidia reported revenues of 96.2 billion, up 106% year-over-year.",
    },
]


class StageCapture:
    """Records each pipeline stage as the run produces it. Safe to read while the run continues."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stages: list[dict[str, Any]] = []
        self._buf: list[str] = []
        self._reporter = StageReporter(emit=self._emit)

    def _emit(self, line: str) -> None:
        # The CLI wraps a long line. A continuation is indented further; join it back for the page.
        if line.startswith("          ") and self._buf:
            previous = self._buf[-1].rstrip()
            extra = line.strip()
            # A wrapped hyphen belongs to the word ("year-" + "over"), not a new word.
            joiner = "" if previous.endswith("-") else " "
            self._buf[-1] = f"{previous}{joiner}{extra}"
            return
        text = line.strip()
        if text:
            self._buf.append(text)

    def __call__(self, node: str, update: dict, recorder) -> None:
        self._buf = []
        self._reporter.emit = self._emit
        started = time.perf_counter()
        self._reporter(node, update, recorder)
        lines = [line.strip() for line in self._buf if line.strip()]
        stage = {
            "step": self._reporter.step,
            "node": node,
            "role": ROLES.get(node, ""),
            "lines": lines[1:] if lines else [],
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }
        with self._lock:
            self._stages.append(stage)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(stage) for stage in self._stages]


def parse_run_body(raw: dict[str, Any], *, paid: bool) -> dict[str, Any]:
    """Check the form payload. Raises ValueError, or PermissionError in paid mode without confirmation."""
    claim = raw.get("claim")
    if not isinstance(claim, str):
        raise ValueError("The claim has to be text.")
    claim = claim.strip()
    limit = get_settings().max_claim_chars
    if len(claim) > limit:
        raise ValueError(f"The claim is longer than {limit} characters.")
    cutoff = _blank(raw.get("cutoff"))
    if cutoff:
        try:
            date.fromisoformat(cutoff)
        except ValueError as exc:
            raise ValueError("The cut-off date must be YYYY-MM-DD.") from exc
    company = _blank(raw.get("company"))
    if company:
        company = company.upper()
        if _TOKEN.fullmatch(company) is None:
            raise ValueError("The ticker can only use letters, numbers, dots, and dashes.")
    influencer = _blank(raw.get("influencer"))
    post = _blank(raw.get("post"))
    if post and _TOKEN.fullmatch(post) is None:
        raise ValueError("The video id can only use letters, numbers, dots, and dashes.")
    if paid and not raw.get("confirm_paid"):
        raise PermissionError("This machine is in paid LLM mode. Confirm the run before it can spend credit.")
    return {"claim": claim, "cutoff": cutoff, "company": company, "influencer": influencer, "post": post}


def _blank(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def run_check(body: dict[str, Any], capture: StageCapture | None = None) -> dict[str, Any]:
    """Run one check and return the view the page renders."""
    capture = capture or StageCapture()
    out = execute(
        body["claim"],
        body.get("cutoff"),
        company_ticker=body.get("company"),
        influencer=body.get("influencer"),
        post_id=body.get("post"),
        on_stage=capture,
    )
    view = build_view(out)
    view["cli"] = cli_command(body)
    return view


def log_file(logs_dir: Path, request_id: str) -> Path:
    """Resolve a decision log inside logs_dir. Rejects anything that could leave that folder."""
    if _REQUEST_ID.fullmatch(request_id) is None or ".." in request_id:
        raise ValueError(request_id)
    root = logs_dir.resolve()
    path = (root / f"{request_id}.json").resolve()
    if path.parent != root:
        raise ValueError(request_id)
    return path


def list_logs(limit: int = 40) -> list[dict[str, Any]]:
    logs_dir = get_settings().logs_dir
    if not logs_dir.exists():
        return []
    files = sorted(logs_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    rows: list[dict[str, Any]] = []
    for path in files:
        if len(rows) >= limit or _REQUEST_ID.fullmatch(path.stem) is None:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        text = str(data.get("input") or "")
        calls = data.get("agent_calls") or []
        rows.append({
            "request_id": data.get("request_id") or path.stem,
            "status": data.get("status"),
            "cutoff_date": data.get("cutoff_date"),
            "input": text[:180],
            "latency_ms": data.get("total_latency_ms"),
            "agent_calls": len(calls) if isinstance(calls, list) else 0,
            "final_verdicts": data.get("final_verdicts") or {},
        })
    return rows


def load_log_view(request_id: str) -> dict[str, Any]:
    path = log_file(get_settings().logs_dir, request_id)
    if not path.exists():
        raise FileNotFoundError(request_id)
    text = path.read_text(encoding="utf-8")
    try:
        log = DecisionLog.model_validate_json(text)
    except ValidationError as exc:
        raw = json.loads(text)
        return {
            "decision_log": raw if isinstance(raw, dict) else {"raw": raw},
            "trace_text": "",
            "stats": {},
            "warning": f"This log does not match the current schema ({exc.error_count()} fields).",
        }
    return {"decision_log": log.model_dump(mode="json"), "trace_text": format_trace(log), "stats": _stats(log)}


def meta() -> dict[str, Any]:
    settings = get_settings()
    from finsight.agents.orchestrator.routing import INFLUENCERS

    totals = ledger_summary()
    spent = totals.get("ALL", {}).get("cost_usd", 0.0)
    companies = [
        {"ticker": ticker, "name": names[0]}
        for ticker, names in settings.corpus_companies.items()
        if names
    ]
    companies.sort(key=lambda row: row["name"].lower())
    paid = settings.llm_provider != "mock"
    return {
        "mode": {
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "paid": paid,
            "corpus_as_of": settings.corpus_as_of_date.isoformat(),
            "max_retries": settings.max_retries,
            "env_override": bool(os.getenv("FINSIGHT_LLM_PROVIDER")),
        },
        "companies": companies,
        "influencers": [aliases[0] for aliases in INFLUENCERS.values()],
        "examples": EXAMPLES,
        "disclaimer": BOUNDARY_DISCLAIMER,
        "spend": {
            "cost_usd": round(spent, 4),
            "budget_usd": COURSE_BUDGET_USD,
            "calls": int(totals.get("ALL", {}).get("calls", 0)),
        },
        "logs_dir": str(settings.logs_dir),
    }


def start_job(body: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {
        "id": job_id,
        "status": "running",
        "capture": StageCapture(),
        "result": None,
        "error": None,
        "started": time.perf_counter(),
    }
    with _LOCK:
        _JOBS[job_id] = job
        finished = [item for item in _JOBS.values() if item["status"] != "running"]
        overflow = len(_JOBS) - 40
        for item in sorted(finished, key=lambda row: row["started"])[: max(overflow, 0)]:
            _JOBS.pop(item["id"], None)
    threading.Thread(target=_worker, args=(job, body), daemon=True, name=f"finsight-ui-{job_id}").start()
    return job_id


def job_snapshot(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return None
    return {
        "id": job["id"],
        "status": job["status"],
        "stages": job["capture"].snapshot(),
        "result": job["result"],
        "error": job["error"],
        "elapsed_ms": round((time.perf_counter() - job["started"]) * 1000, 1),
    }


def _worker(job: dict[str, Any], body: dict[str, Any]) -> None:
    try:
        job["result"] = run_check(body, job["capture"])
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001 - the page shows the failure instead of dropping the job
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[ui] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        path = urlparse(self.path).path
        if path in _FILES:
            self._file(*_FILES[path])
            return
        if path == "/api/meta":
            self._json(200, meta())
            return
        if path == "/api/logs":
            self._json(200, {"logs": list_logs()})
            return
        if path.startswith("/api/logs/"):
            self._saved_log(path.removeprefix("/api/logs/"))
            return
        if path.startswith("/api/jobs/"):
            self._job(path.removeprefix("/api/jobs/"))
            return
        self._json(404, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        if urlparse(self.path).path != "/api/run":
            self._json(404, {"error": "Not found."})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 100_000:
            self._json(400, {"error": "The request body is empty or too large."})
            return
        try:
            raw = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._json(400, {"error": "The request body is not JSON."})
            return
        if not isinstance(raw, dict):
            self._json(400, {"error": "The request body must be an object."})
            return
        try:
            body = parse_run_body(raw, paid=get_settings().llm_provider != "mock")
        except PermissionError as exc:
            self._json(400, {"error": str(exc), "paid": True})
            return
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(202, {"job_id": start_job(body)})

    def _saved_log(self, request_id: str) -> None:
        try:
            self._json(200, load_log_view(request_id))
        except ValueError:
            self._json(400, {"error": "That request id is not valid."})
        except FileNotFoundError:
            self._json(404, {"error": f"No decision log for {request_id}."})

    def _job(self, job_id: str) -> None:
        if _TOKEN.fullmatch(job_id) is None:
            self._json(400, {"error": "That job id is not valid."})
            return
        snapshot = job_snapshot(job_id)
        if snapshot is None:
            self._json(404, {"error": "That run is no longer in this session."})
            return
        self._json(200, snapshot)

    def _file(self, name: str, content_type: str) -> None:
        path = STATIC / name
        if not path.is_file():
            self._json(500, {"error": f"Missing UI file {name}."})
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        raise SystemExit(f"Could not listen on {host}:{port} ({exc}).") from exc
    print(f"FinSight testing UI: http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
