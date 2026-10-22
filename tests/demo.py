"""Shared demo inputs for tests."""
from datetime import date

DEMO_CLAIM = (
    "Joseph Carlson said Company X's revenue growth proves it is the biggest winner "
    "from AI infrastructure demand, and that this growth is just getting started. "
    "Cut-off date: 31 Jul 2025"
)
# Stored posts the tests check. Each transcript is the sentence the decomposer should see.
FIXTURE_POSTS = [
    DEMO_CLAIM,
    "Joseph Carlson said Company X's revenue growth is driven by AI demand.",
    "Joseph Carlson said Company X is getting more profitable: its margins keep expanding.",
    "Joseph Carlson said Company X is a really interesting company.",
    "Its revenue growth is driven by AI infrastructure demand and is just getting started.",
    "Its revenue growth is just getting started.",
]
DEMO_CUTOFF = date(2025, 7, 31)
