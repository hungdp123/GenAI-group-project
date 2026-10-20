# Basic Narrative RAG numerical consistency check

Run date: 18 October 2026  
Provider: local Ollama, `qwen3-vl:4b-instruct`  
Retrieval: BM25, `top-k 3` per support/counter query  
Generation limit: 1,500 tokens; Ollama context: 20,000 tokens

## Finding and changes

The shared verdict enum in `finsight/contracts/schemas.py` uses
`CONTRADICTED`; the Basic RAG response contract now also accepts `REFUTED` for
clear conflicts in a single factual assertion. Its prompt defines the
distinction: a reliable value for the same company, metric, reporting period,
unit and stated accounting basis refutes a conflicting number. Numeric
closeness alone is not a match.

Basic RAG applies a deterministic check only to a narrow set of clear numeric
claims (operating/gross/net margin percentages and DAP values) when the claim
has one metric, one value and an explicit quarter or month. Matching uses
half-up rounding at the coarser precision displayed by the claim and cited
evidence. For example, a claim of `34.0%` matches evidence displayed as `34%`,
while `34.5%` does not. Multi-part and ambiguous claims are not collapsed into
this single-value check; the prompt reserves `PARTIALLY_SUPPORTED` for
multi-part claims with different outcomes across their independently verifiable
parts.

The checker reads cited Tier 1/2 evidence and corrects the verdict and
explanation only when the cited passage provides a clear, period-matched value.
If a model quote has a damaged punctuation character but identifies the exact
structured metric/year/value cell, the code replaces it with the corresponding
verbatim source substring; the ordinary exact-substring citation validator then
runs unchanged. This does not treat arbitrary approximate quotes as valid.

## Rerun results

| Claim | Expected | Final verdict | Technical validation | Ground-truth match | Report |
|---|---|---|---|---|---|
| Alphabet operating margin is 34% in Q1 2025 | `SUPPORTED` | `SUPPORTED` | Pass: `status=OK`, exact citation passed, numeric check `APPLIED_MATCH` | Yes | `data/narrative/experiments/narrative_basic_rag/GOOGL_margin_34pct_consistency.json` |
| Alphabet operating margin is 36% in Q1 2025 | `REFUTED` | `REFUTED` | Pass: `status=OK`, exact citation passed, numeric check `APPLIED_CONFLICT`; cited 34% | Yes | `data/narrative/experiments/narrative_basic_rag/GOOGL_margin_36pct_consistency.json` |
| META DAP was 3.27 billion on average for June 2024 | `SUPPORTED` | `SUPPORTED` | Pass: `status=OK`, exact citation passed, numeric check `APPLIED_MATCH` | Yes | `data/narrative/experiments/narrative_basic_rag/META_DAP_327B_consistency.json` |

The two GOOGL claims used cutoff `2026-06-01`; META used `2025-07-31`. The
deterministic check selected a Q1 earnings-release row for GOOGL and the official
Q2 2024 earnings release for META. Generation ran locally and incurred no API
charge. Each report contains model token counts and the consistency-check trace.

## Regression tests

`python -m pytest tests/test_narrative_basic_rag.py -q`: **7 passed**. Coverage
includes matching and conflicting Q1 operating-margin values, half-up rounding,
META DAP, malformed quote repair to an exact source substring, and leaving a
multi-part claim out of the single-number resolver.

Full suite at the same revision: **204 passed, 14 xfailed, 1 failed**. The
unrelated failure is `tests/test_docling_charts.py::test_ollama_provider_uses_local_schema_and_image`; its fixture writes
`b"fake-png"`, which Pillow rejects as an invalid image. No chart-processing
code was changed for this task.

## Scope and limitation

These checks establish technical schema/citation validity and agreement with
the three stated ground-truth labels. They do not establish broad Basic RAG
accuracy. The deterministic consistency check intentionally abstains when
metric, period, basis, or cited numeric structure cannot be identified
unambiguously. Full semantic evaluation on reviewed claims remains pending.
