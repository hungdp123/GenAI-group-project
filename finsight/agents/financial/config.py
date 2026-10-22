"""
Companies in project scope
"""

from datetime import date

# ticker -> industry
COMPANIES = {
    "AAPL": "Consumer electronics",
    "GOOGL": "Internet / advertising",
    "MSFT": "Software",
    "NFLX": "Streaming",
    "NVDA": "Semiconductors",
    "TSLA": "Automotive",
    "AMZN": "E-commerce / cloud",
    "META": "Social media / advertising",
    "MELI": "E-commerce / fintech",
    "SBUX": "Restaurants",
    "PLTR": "Software",
    "INTC": "Semiconductors",
    "INTU": "Software",
    "CRM": "Software",
    "ORCL": "Software",
    "AMD": "Semiconductors",
}
# Only needed if a ticker is missing from SEC's list: ticker -> CIK
CIK_OVERRIDES = {}

MIN_PERIOD_END = date(2024, 1, 1)   # drop facts for periods ending before this
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
SOURCE_TIER = 1 # XBRL facts from filings are always Tier 1
