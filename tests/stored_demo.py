"""A stored Company X post for tests that need a finished check.

The running app does not invent claims. These tests write the demo sentences
into the claim file and ask whether that post is right. The financial and
narrative stand-ins supply the evidence, because CI has no filing corpus.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from finsight.agents.decomposition.stub import StubDecomposer
from finsight.agents.financial.stub import StubFinancialAgent
from finsight.agents.narrative.demo.stub import StubNarrativeAgent
from finsight.config import AgentRegistry

from demo import DEMO_CLAIM, DEMO_CUTOFF

DEMO_PROMPT = "Is Joseph Carlson right about Company X?"


def demo_registry(**overrides) -> AgentRegistry:
    agents = {"financial": StubFinancialAgent(), "narrative": StubNarrativeAgent()}
    agents.update(overrides)
    return AgentRegistry.from_config(**agents)


def prepared_demo(monkeypatch, tmp_path: Path, *, transcript: str = DEMO_CLAIM, pin_impl: bool = False) -> None:
    posts = tmp_path / "stored_posts"
    posts.mkdir(exist_ok=True)
    (posts / "posts.jsonl").write_text(
        json.dumps({
            "influencer": "joseph_carlson",
            "video_id": "demo-0",
            "title": "Company X",
            "description": "Company X",
            "upload_date": "2025-07-31",
            "transcript": transcript,
            "channel": "Joseph Carlson",
        })
        + "\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims"
    claims.mkdir(exist_ok=True)
    rows = []
    for claim in StubDecomposer().decompose(transcript, DEMO_CUTOFF):
        rows.append(json.dumps({
            "video_id": "demo-0",
            "company_ticker": claim.company_ticker or "CMPX",
            "sub_claim_id": claim.sub_claim_id,
            "text": claim.text,
            "sub_claim_type": claim.sub_claim_type.value,
            "is_quantitative": claim.is_quantitative,
        }))
    (claims / "claims.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("FINSIGHT_POSTS_DIR", str(posts))
    if pin_impl:
        monkeypatch.setenv("FINSIGHT_IMPL_FINANCIAL", "finsight.agents.financial.stub:StubFinancialAgent")
        monkeypatch.setenv("FINSIGHT_IMPL_NARRATIVE", "finsight.agents.narrative.demo.stub:StubNarrativeAgent")


def store_fixture_claims(video_id: str, transcript: str) -> None:
    """Add stored sentences for one video already in the test posts directory."""
    posts = Path(os.environ["FINSIGHT_POSTS_DIR"])
    claims = posts.parent / "claims"
    claims.mkdir(exist_ok=True)
    rows = []
    for claim in StubDecomposer().decompose(transcript, DEMO_CUTOFF):
        rows.append(json.dumps({
            "video_id": video_id,
            "company_ticker": claim.company_ticker or "CMPX",
            "sub_claim_id": claim.sub_claim_id,
            "text": claim.text,
            "sub_claim_type": claim.sub_claim_type.value,
            "is_quantitative": claim.is_quantitative,
        }))
    (claims / "claims.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
