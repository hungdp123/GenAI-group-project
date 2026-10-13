""" Concepts based on us-gaap tags

Only concepts whose tags have one clear, consistent meaning across companies are listed.

For a given period the first tag in the list that has a value wins as companies change tags overtime.

Each entry: (kind, tags, unit, additive)
  kind      "duration" (with start and end) or "instant" (as of; with end only)
  unit      the Company Facts unit key the values are read from
  additive  True if a quarter may be derived by subtracting year-to-date figures.
            Weighted-average share counts and per-share figures are not additive.
"""

CONCEPTS = {
    # duration
    "Revenue": ("duration", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                             "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"], "USD", True),
    "CostOfRevenue": ("duration", ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"], "USD", True),
    "GrossProfit": ("duration", ["GrossProfit"], "USD", True),
    "OperatingIncome": ("duration", ["OperatingIncomeLoss"], "USD", True),
    "NetIncome": ("duration", ["NetIncomeLoss", "ProfitLoss"], "USD", True),
    "OperatingCashFlow": ("duration", ["NetCashProvidedByUsedInOperatingActivities"], "USD", True),
    "CapitalExpenditure": ("duration", ["PaymentsToAcquirePropertyPlantAndEquipment",
                                        "PaymentsToAcquireProductiveAssets"], "USD", True),
    "DilutedShares": ("duration", ["WeightedAverageNumberOfDilutedSharesOutstanding"], "shares", False),
    "DilutedEPS": ("duration", ["EarningsPerShareDiluted"], "USD/shares", False),
    # instant
    "TotalAssets": ("instant", ["Assets"], "USD", True),
    "TotalLiabilities": ("instant", ["Liabilities"], "USD", True),
    "LiabilitiesAndEquity": ("instant", ["LiabilitiesAndStockholdersEquity"], "USD", True),
    "ShareholdersEquity": ("instant", ["StockholdersEquity",
                                       "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
                           "USD", True),
    "CashAndEquivalents": ("instant", ["CashAndCashEquivalentsAtCarryingValue"], "USD", True),
    "CurrentAssets": ("instant", ["AssetsCurrent"], "USD", True),
    "CurrentLiabilities": ("instant", ["LiabilitiesCurrent"], "USD", True),
    "Inventory": ("instant", ["InventoryNet", "InventoryRealEstate"], "USD", True),
}
TAG_TO_CONCEPT = {t: c for c, v in CONCEPTS.items() for t in v[1]}


def kind(concept: str) -> str:
    return CONCEPTS[concept][0]


def priority(concept: str, tag: str) -> int:
    return CONCEPTS[concept][1].index(tag)


def unit(concept: str) -> str:
    return CONCEPTS[concept][2]


def additive(concept: str) -> bool:
    return CONCEPTS[concept][3]
