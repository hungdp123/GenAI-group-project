"""
Financial evidence schema
"""

from __future__ import annotations
from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field

Concept = Literal["Revenue", "CostOfRevenue", "GrossProfit", "OperatingIncome", "NetIncome",
                  "OperatingCashFlow", "CapitalExpenditure", "DilutedShares", "DilutedEPS",
                  "TotalAssets", "TotalLiabilities", "LiabilitiesAndEquity", "ShareholdersEquity", "CashAndEquivalents",
                  "CurrentAssets", "CurrentLiabilities", "Inventory"]
Span = Literal["quarter", "annual"]
Metric = Literal["gross_margin", "operating_margin", "net_margin", "free_cash_flow", "fcf_margin",
                 "liabilities_to_equity", "current_ratio", "cash_ratio"]
Status = Literal["ok", "insufficient_data", "out_of_scope"]


class FinancialFact(BaseModel):
    """One reported XBRL value that is traceable to the filing it came from."""
    ticker: str
    cik: int
    concept: str                      # canonical name, e.g. "Revenue"
    xbrl_tag: str                     # us-gaap tag actually used
    value: float
    unit: str = "USD"
    period_start: Optional[date] = None   # None for instant facts
    period_end: date
    period_class: Literal["I", "Q", "H1", "9M", "FY"]   # I = instant; others = duration length
    form: str                         # 10-K / 10-Q
    accession: str
    filed: date                       # filing date: what the cut-off filter is applied to
    source_url: str
    source_tier: Literal[1] = 1


class SeriesPoint(BaseModel):
    """One period in a trend series."""
    period_end: date
    value: float
    derived: bool = False
    evidence_id: str                  # the single-period evidence this point came from


class FinancialEvidence(BaseModel):
    """Result of one deterministic tool call."""
    evidence_id: str                  # stable hash of tool + arguments
    tool: str
    ticker: str
    metric: str                       # e.g. "Revenue", "Revenue_growth_yoy", "gross_margin"
    status: Status
    value: Optional[float] = None
    unit: Literal["USD", "ratio", "shares", "USD/shares"] = "USD"   # ratio is a decimal: 0.14 = 14%
    span: Optional[str] = None
    period_end: Optional[date] = None
    cutoff_date: Optional[date] = None      # as requested; None = latest in corpus
    as_of_date: Optional[date] = None       # latest filing date among the inputs
    formula: str = ""                       # human-readable with numbers substituted
    inputs: list[FinancialFact] = Field(default_factory=list)   # every fact used = the citations
    derived: bool = False                   # True if any input was computed (e.g. Q4 = FY - 9M)
    source_tier: Literal[1] = 1
    reason: Optional[str] = None            # set when status != ok
    warnings: list[str] = Field(default_factory=list)
    # trend tool only
    series: list[SeriesPoint] = Field(default_factory=list)      # oldest first
    direction: Optional[Literal["up", "down", "flat", "mixed"]] = None


class ValueCheck(BaseModel):
    """Auditor execution-comparison which is stated value vs computed value."""
    stated: float
    computed: Optional[float]
    abs_diff: Optional[float] = None
    rel_diff: Optional[float] = None
    tolerance: float
    match: bool
    evidence_id: str
