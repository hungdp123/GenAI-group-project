"""Older import path. The advice screen is AdviceBoundaryAgent in agent.py."""

from finsight.agents.advice_boundary.agent import AdviceBoundaryAgent as StubAdviceBoundary
from finsight.agents.advice_boundary.agent import without_advice

__all__ = ["StubAdviceBoundary", "without_advice"]
