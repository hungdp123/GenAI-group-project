# PDF and chart processing

The processing command uses current retrieval-eligible canonical manifests. It leaves official
PDFs unchanged and records their SHA-256 hashes. Visual comparisons and retrieval experiments
are later work; extracted fields are not marked as visually verified.

## Run

Audit every eligible PDF without model calls:

```powershell
python -m finsight.agents.narrative.corpus.pdf.pdf_chart_processing --all --audit-only
```

Extract PDF text, tables, and chart information:

```powershell
python -m finsight.agents.narrative.corpus.pdf.pdf_chart_processing --all --ollama-model qwen3-vl:2b-instruct-q4_K_M --chart-timeout 180
```

The local 2B instruct model avoids the thinking-only responses observed with the installed
`qwen3-vl:4b` thinking variant and uses less memory than `gemma3:12b`. Each result retains the
actual model name. Other installed Ollama vision models can be passed through `--ollama-model`.
Install dependencies with `python -m pip install -e ".[document-ai,retrieval]"` and install the
chosen local model with `ollama pull MODEL_NAME` if necessary.

### OpenAI API configuration

The OpenAI adapter accepts `--openai-model gpt-5.4`. It reads `OPENAI_API_KEY` first,
then `openai_api_key` from the project's gitignored `config.yaml` using the shared key
loader. Run from the project root so the default relative config path resolves there;
`FINSIGHT_SECRETS_FILE` can select a different config file. An explicitly supplied
Python provider `api_key` argument takes precedence over both.

Create `config.yaml` from `config.example.yaml` if it does not already exist. Add your
key in your local configuration:

```yaml
openai_api_key: "YOUR_OPENAI_API_KEY"
llm_provider: mock
```

`--openai-model` explicitly selects paid chart extraction even when the general
application's `llm_provider` remains `mock`. Resume the remaining work and promote:

```powershell
python -u -m finsight.agents.narrative.corpus.pdf.pdf_chart_processing --all --openai-model gpt-5.4 --chart-workers 1 --chart-timeout 300 --promote --rebuild-vectors
```

Successful source-matched results from Ollama remain cached with their original model
metadata; OpenAI handles pending or failed chart extractions. Do not add `--force` to
this resume command. Use `--ticker CRM` instead of `--all` to limit the batch to CRM.

The adapter sends chart images and nearby extracted text through the Responses API;
Docling/OCR processing stays local. It uses the shared key loader only, not the shared
spend guard or usage ledger, and does not enforce a dollar spending cap.
`--chart-max-output-tokens` limits response generation, not total API spend.
Adding config-file support did not run extraction or send paid OpenAI requests.

### Automatic OpenAI usage and estimated cost

For OpenAI runs started after the usage-tracking update, the processor automatically
prints one combined token/cost summary when it finishes, including when an exception
or Ctrl+C reaches the finalizer. This covers all selected companies and PDFs in that
invocation. No extra option is needed on your extraction command.

Only new provider requests are recorded. Reusing a chart cache or a completed PDF
does not add its old tokens to the run total. A chart skipped locally after a fatal
billing error is also not counted as a new API request. Returned response usage is
recorded before checking for missing parsed chart output. SDK/network/parse errors
without usage are recorded as unknown cost, not zero cost.

Each run writes:

```text
data/narrative/experiments/pdf_chart_processing/usage/<RUN_ID>/
  calls.jsonl       # One record per new provider request; updated after each call
  summary.json      # Combined tokens, cost estimate, missing-usage counts and status
```

The exact summary path is printed at startup and completion. The summary is updated
after every recorded call, so completed calls remain recorded if processing stops.
A force-killed process can still lose an in-flight request's usage.

Show cumulative recorded usage across all selected documents and runs without
starting extraction or making API calls:

```powershell
python -m finsight.agents.narrative.corpus.pdf.pdf_chart_processing --all --usage
```

For one company:

```powershell
python -m finsight.agents.narrative.corpus.pdf.pdf_chart_processing --ticker GOOGL --usage
```

The totals start with the tracking update; prior extractions and
`check_openai_key.py` calls are not backfilled. `--experiment NAME` selects that
experiment's separate usage history. The summary's `cost_estimate_complete` is
false if any request lacks usage or verified pricing; `estimated_cost_usd` then
represents only the priced portion.

GPT-5.4 estimates use standard-tier input/cached-input/output rates of
$2.50/$0.25/$15.00 per million tokens, checked on 18 October 2026 against the
[official model documentation](https://developers.openai.com/api/docs/models/gpt-5.4).
Other models, processing tiers and requests above 272,000 input tokens retain token
counts but have no cost estimate until their pricing is configured. Output usage
includes any reported reasoning tokens; input usage includes the chart image tokens.
These local estimates do not show shared account spending or remaining credit and
do not enforce a spending cap. Use the OpenAI dashboard for authoritative billing.
The existing `finsight usage` shared ledger remains separate from this chart ledger.

Use `--ticker INTC` instead of `--all` for one company. Repeat `--ticker` for several companies.
Use `--document-id DOC_ID` to restrict the document set and `--pages 7-10` for a page selection.
The default is one chart worker. `--force` bypasses caches.
For a dense chart whose JSON was cut off, increase `--chart-max-output-tokens` and, for
Ollama, `--ollama-context`. These default to 4,000 output tokens and an 8,192-token context
for Ollama (6,000 output tokens for OpenAI). Successful cached records are still reused.

## Routing and outputs

Pages are classified as digitally readable, image-only, requiring OCR, chart-heavy, or
table-heavy. Native text handles ordinary readable pages. Contiguous ranges needing layout
analysis go to Docling with OCR disabled; zero-text and sparse image-bearing ranges use
RapidOCR. The converter reuses its loaded models. Built-in Granite chart extraction stays off.

Docling picture geometry, labels, and financial text identify chart regions. Fallback regions
are limited to OCR or presentation pages and reject long financial prose. Every selected chart
is cropped from the original PDF, then read by the configured vision provider. Displayed value
strings, units, signs, accounting basis where stated, period order, legends, axes, trends, and
uncertainties remain traceable to the source page and crop. Missing fields stay unknown.

Artifacts live under:

```text
data/narrative/processed/<TICKER>/experiments/pdf_chart_processing/
  audit_report.json          # All selected pages; no inference
  experiment_report.json    # Progress and final extraction/coverage report
  <DOCUMENT_ID>/
    page_audit.json
    audit.json               # Per-document extraction details
    blocks.jsonl             # Page-cited narrative, chart, and table blocks
    ranges/pNNN-NNN/         # Docling range exports and per-chart caches/crops
```

For readable extracted content, open a range's `document.md` or
`enhanced/document.enhanced.md`. The latter includes successfully extracted chart tables.
Structured chart fields are in `enhanced/charts/*.json`; the matching PNG files show the
source crops. `blocks.jsonl` contains the page-cited blocks used for canonical integration.

Routing and candidate detection use heuristics. A result with zero detected chart candidates
does not establish that the original PDF contains no charts; detection coverage and extracted
values can be compared with the originals during the later experiments.

The command checkpoints after each document and each completed extraction range. Completed
chart caches survive interruptions. Rerunning reuses matching source/range caches and successful
chart records only when crop hashes match. Prior successful trial records may be reused for the
identical official PDF, with their original provider/model retained.

The company-level progress report counts completed documents. While a document is running,
its newest chart records may already be on disk before the company totals update. Use the
terminal's per-chart messages or that document's `enhanced/charts/` files for current activity.

`status: partial` means one or more extraction operations failed. Inspect `documents_with_errors`
and `coverage_limitations`. `unsupported_pages` lists pages for which the richer extraction
produced no usable content; individual chart failures are reported separately. Successful OCR
text remains available when a chart is unreadable.

## Promote and rebuild

Promote the recorded page blocks and rebuild BM25 without rerunning extraction:

```powershell
python -m finsight.agents.narrative.corpus.pdf.pdf_chart_processing --all --promote-only --rebuild-vectors
```

Or append `--promote --rebuild-vectors` to an extraction command. Use `--document-id` to promote
selected documents. Structural extraction warnings do not require manual approval in this step.

Promotion stores hashed artifacts in `<TICKER>/enrichments/` and records them in
`<TICKER>/pdf_enrichment.json`. Only current eligible documents whose source hashes match can
be promoted. Current manifest provenance, including explicitly estimated dates, is carried into
the blocks and chunks. Full ordinary page blocks are replaced by their enriched page blocks;
text within successfully extracted numeric chart/table regions is not repeated in the narrative.
Charts without numeric values retain their Docling text alongside descriptive fields.

Canonical rebuilds retain experiments and promoted artifacts. Generated files are staged and
published with rollback copies. The effective corpus fingerprint includes both eligible source
metadata and promoted extraction/coverage hashes. This invalidates stale vectors and dense
caches. Qdrant rebuilds use the enriched chunks and preserve page/chart/citation metadata.

Coverage limitations are stored in the processing report and index. `CorpusRetriever` exposes
them through `coverage_limitations(cutoff_date)`, filtered by company and publication cutoff.

## Stopping and resuming

Read progress without starting another extraction job:

```powershell
python -m finsight.agents.narrative.corpus.pdf.pdf_chart_processing --all --status
```

Use Ctrl+C in the terminal running the processor, then rerun the same command to resume from
caches. The experiment report records interruptions that reach the Python handler. If Python
is force-terminated, the last per-chart caches and completed range blocks remain available.
Do not launch two processors against the same company's experiment directory simultaneously.
