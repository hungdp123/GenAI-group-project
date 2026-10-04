from datetime import date
from finsight.agents.financial.agent import XbrlFinancialAgent
from finsight.contracts.schemas import SubClaim, SubClaimType

G = SubClaimType.HISTORICAL_GROWTH
P = SubClaimType.PROFITABILITY_CASH_LEVERAGE
D = SubClaimType.DEMAND_ATTRIBUTION
CUTOFF = date(2025, 7, 31)

CASES = [  # (ticker, type, claim text, cut-off)
    ("NVDA", G, "NVIDIA's revenue grew strongly.", CUTOFF),
    ("NVDA", G, "NVIDIA's revenue grew strongly.", date(2025, 5, 1)),
    ("KSS", G, "Kohl's revenue is growing.", CUTOFF),
    ("NVDA", P, "NVIDIA's gross margin is expanding.", CUTOFF),
    ("TSLA", P, "Tesla's free cash flow is improving.", CUTOFF),
    ("GME", P, "GameStop's share count keeps rising.", CUTOFF),
    ("OPEN", P, "Opendoor's inventory is falling.", CUTOFF),
    ("AMC", P, "AMC's gross margin is improving.", CUTOFF),
    ("MCD", P, "McDonald's leverage is falling.", CUTOFF),
    ("ZZZZ", G, "This company's revenue grew.", CUTOFF),
    ("NVDA", D, "NVIDIA's growth is driven by AI demand.", CUTOFF),
]

agent = XbrlFinancialAgent()
for ticker, kind, text, cutoff in CASES:
    items = agent.gather(
        SubClaim(sub_claim_id="sc1", text=text, sub_claim_type=kind,
                 company_ticker=ticker, is_quantitative=True),
        cutoff, None)
    print(f"\n[{ticker}] {text}  (cut-off {cutoff})")
    if not items:
        print("   -> no evidence returned")
    for i in items:
        leak = "  LOOK-AHEAD!" if i.publication_date > cutoff else ""
        print(f"   -> {i.stance.value:11} {i.metric_name} = {i.computed_value}  filed {i.publication_date}{leak}")
        print(f"      {i.content}")
        print(f"      {i.url}")