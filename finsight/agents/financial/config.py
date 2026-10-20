"""
Companies in scope
Edit: companies and period based on what the team wants
"""

from datetime import date

# ticker -> industry
COMPANIES = {
    "AAPL": "Consumer electronics",
    "GOOGL": "Internet / advertising",      # Alphabet (Google)
    "MSFT": "Software",
    "NKE": "Apparel / footwear",
    "NVDA": "Semiconductors",
    "MCD": "Restaurants",
    "TSLA": "Automotive",
    "NFLX": "Streaming",
    "OPEN": "Real estate technology",       # Opendoor
    "KSS": "Department stores",             # Kohl's
    "DNUT": "Food",                         # Krispy Kreme
    "GPRO": "Consumer electronics",         # GoPro
    "BIRD": "Apparel / footwear",           # Allbirds
    "GME": "Specialty retail",              # GameStop
    "AMC": "Entertainment",                 # AMC Entertainment
}
# Only needed if a ticker is missing from SEC's list: ticker -> CIK
CIK_OVERRIDES = {}

MIN_PERIOD_END = date(2022, 1, 1)   # drop facts for periods ending before this
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
SOURCE_TIER = 1                      # XBRL facts from filings are always Tier 1
