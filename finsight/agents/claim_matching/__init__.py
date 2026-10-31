"""Claim matching: find the influencer line a prompt refers to, then line up the facts.

filter (catalog) -> rank (facts + BM25) -> pick (rules, or an LLM choosing listed ids) -> align (both / conflict /
quote_only / user_only). See docs/claim_matching.md.
"""
