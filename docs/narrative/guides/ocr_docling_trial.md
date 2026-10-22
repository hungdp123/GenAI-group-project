# OCR, Docling, and financial-chart extraction

Date: 2026-10-15

## Purpose

The existing narrative ingestion pipeline uses `pypdf` to extract the embedded text layer from PDFs. This works for digitally generated PDFs, but it cannot recover text from image-only slides, scanned pages, charts, or text rendered as graphics.

This trial evaluated Docling as a fallback for PDFs that produce no text or very little text through the existing parser. The production corpus and indexes were not modified during the trial.

## Trial document

The test used Intel's 19-page Q1 2025 earnings presentation:

```text
data/narrative/raw/INTC/official_ir/
2025-04-24_earnings_presentation_Q1_2025_Earnings_Deck_INTC.com.pdf
```

Document ID: `INTC-IR-2b85d5c638b905e1`

The existing `pypdf` parser reported:

- 0 processed blocks
- 0 extracted words
- 19 pages with no extractable text

The PDF is image-based and contains title slides, photographs, dense legal text, KPI cards, charts, and financial tables.

## Environment and installation

For the original trial, Docling was installed in an isolated workspace virtual environment:

```powershell
uv venv tmp/docling-venv --system-site-packages
uv pip install --python tmp/docling-venv/Scripts/python.exe docling
```

Installed Docling version: `2.139.0`

The generated `docling.exe` wrapper was blocked by Windows Application Control. The Python API worked normally and was used instead.

The isolated environment occupies approximately 1.1 GB. Docling's cached layout and standard models occupy approximately another 0.49 GB under the Hugging Face cache.

## OCR and layout configuration

The successful test used Docling's standard PDF pipeline with:

- RapidOCR using ONNX Runtime
- full-page OCR
- layout analysis
- table-structure recognition
- a 300-second document timeout
- CPU execution

The key configuration was:

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

options = PdfPipelineOptions()
options.do_ocr = True
options.do_table_structure = True
options.document_timeout = 300
options.ocr_options = RapidOcrOptions(mode=OcrMode.FULL_PAGE)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options),
    }
)
result = converter.convert(source_pdf, page_range=(1, 19))
```

The reusable trial script is located at:

```text
tmp/pdfs/docling_trial/run_trial.py
```

Because this is under `tmp/`, it should be treated as an experimental script rather than production code.

## Commands used

The first run tested pages 1-3:

```powershell
tmp/docling-venv/Scripts/python.exe `
  tmp/pdfs/docling_trial/run_trial.py `
  "data/narrative/raw/INTC/official_ir/2025-04-24_earnings_presentation_Q1_2025_Earnings_Deck_INTC.com.pdf" `
  tmp/pdfs/docling_trial/output `
  --first-page 1 `
  --last-page 3
```

The complete OCR/layout run used:

```powershell
tmp/docling-venv/Scripts/python.exe `
  tmp/pdfs/docling_trial/run_trial.py `
  "data/narrative/raw/INTC/official_ir/2025-04-24_earnings_presentation_Q1_2025_Earnings_Deck_INTC.com.pdf" `
  tmp/pdfs/docling_trial/output_full `
  --first-page 1 `
  --last-page 19
```

## OCR/layout results

The complete run finished successfully with no Docling conversion errors.

| Metric | Result |
| --- | ---: |
| Pages processed | 19 |
| Markdown characters recovered | 23,489 |
| Structured text elements | 450 |
| Tables detected | 6 |
| Pictures detected | 11 |
| Conversion errors | 0 |

Generated outputs:

```text
tmp/pdfs/docling_trial/output_full/
  2025-04-24_earnings_presentation_Q1_2025_Earnings_Deck_INTC.com.md
  2025-04-24_earnings_presentation_Q1_2025_Earnings_Deck_INTC.com.json
```

The JSON retains structured document elements and page provenance. The Markdown is convenient for review and downstream text chunking.

## Visual verification

The source pages were rendered and inspected rather than evaluating only the extracted text. A contact sheet of all 19 pages and full-size renders of pages 1-3 were created under:

```text
tmp/pdfs/docling_trial/rendered/
```

Observed strengths:

- Correctly recovered the cover title and date.
- Correctly recovered speaker names and job titles.
- Recovered most dense forward-looking-statement text.
- Recovered KPI-card headings and headline financial values.
- Converted several appendix tables into structured Markdown tables.
- Preserved headings and generally useful reading order.

Observed limitations:

- Dense small text contained occasional garbled sequences.
- `AI` was sometimes recognized as `Al`.
- Some numeric values were misread, for example `1L678`.
- A complex chart-like 2x2 grid triggered a table-layout warning and lost structure.
- Some large multi-section tables were flattened into a single Markdown cell.
- OCR recovered chart labels and numbers but did not reliably preserve the relationships represented by bars, lines, or legends.
- Decorative images were detected as pictures but were not semantically described.

OCR-derived financial values should therefore be treated as lower-confidence evidence and checked against the earnings release, SEC filing, or rendered source page before use in an investment claim.

## Chart-extraction experiment

Docling's optional semantic chart extraction was tested on pages 6-10 using:

```python
options.do_chart_extraction = True
```

This feature downloaded and loaded IBM Granite Vision 4.1 4B. On the available CPU-only machine it:

- used approximately 11.6 GB of RAM;
- downloaded approximately 7.46 GB of model data;
- produced no completed output after approximately 30 minutes for five pages.

The chart run was stopped. No partial result was added to the corpus.

This mode is not suitable as the default bulk-processing path on the current machine. It may be reconsidered for a small number of important chart pages when a suitable GPU or remote inference service is available.

## Enhanced chart pipeline

### Existing implementation review

The original experiment in `tmp/pdfs/docling_trial/run_trial.py` correctly enables full-page RapidOCR
and table-structure extraction, then exports Markdown and JSON. That behavior remains the source of truth
for text, tables, reading order, and page provenance.

The missing functionality was downstream of parsing:

- no production module detected financial charts or distinguished them from photos and logos;
- no code cropped Docling bounding boxes from the original PDF;
- no configurable Vision LLM returned a strict, auditable chart schema;
- Markdown image placeholders had no chart identity, making blind replacement unsafe;
- extraction failures, confidence, and per-value source references were not represented.

The enhanced implementation is additive. It can consume existing Docling Markdown and JSON, or run the
same Docling OCR/table parse itself. Docling picture references are traversed in document-body order and
mapped to Markdown `<!-- image -->` placeholders. Only accepted, successfully extracted charts are
replaced. Chunking, embedding, and indexes remain outside this pipeline.

### Implementation modules

- `finsight/agents/narrative/corpus/pdf/chart_models.py`: strict schemas for source regions, chart data, extraction
  status, usage, and validation.
- `finsight/agents/narrative/corpus/pdf/chart_vision.py`: provider protocol and configurable OpenAI Responses API
  adapter using image input and Structured Outputs.
- `finsight/agents/narrative/corpus/pdf/docling_charts.py`: detection, crop rendering, extraction, caching,
  validation, Markdown/JSON merging, reports, and command-line interface.

| Function | Purpose |
|---|---|
| `parse_pdf_with_docling` | Preserves OCR/table parsing and Markdown/JSON export; built-in Granite chart extraction remains off. |
| `detect_chart_candidates` | Uses picture labels, bounding-box area, and financial OCR evidence. |
| `_detect_fallback_regions` | Finds strong chart-like OCR clusters when no picture was emitted; skips table pages. |
| `crop_chart_region` | Converts Docling coordinates and renders a padded 240-DPI crop from the source PDF. |
| `OpenAIChartVisionProvider.extract` | Sends the crop and OCR context to a configurable vision model with a strict schema. |
| `validate_chart_extraction` | Adds trusted source refs and checks duplicates, periods, numbers, confidence, and OCR. |
| `merge_chart_markdown` | Replaces only successful chart placeholders; failed and unrelated images remain. |
| `merge_chart_json` | Preserves Docling JSON and adds a namespaced chart layer plus picture links. |
| `write_validation_reports` | Writes a JSON audit report and value-level reviewer CSV. |

Every point preserves `value_text`, such as `-$2.4` or `45.1%`, and a normalized value, such as `-2.4`
or `45.1`. A trusted `source_ref`, generated outside the model, identifies the document, page, chart,
and bounding box. The chart crop SHA-256 is also stored.

### Dependencies and configuration

The production-ready optional dependencies are now declared in `pyproject.toml`:

```powershell
python -m pip install -e ".[document-ai]"
$env:OPENAI_API_KEY = "..."
$env:FINSIGHT_CHART_VISION_MODEL = "gpt-5.4"
```

`--vision-model` overrides the environment variable, allowing compatible models to be compared without
code changes. `--workers` controls concurrency. Successful responses are cached using the crop hash,
provider, and model; `--force` bypasses the cache. Failed responses retry on a later run. A non-retryable
credit error trips a provider circuit breaker to avoid unnecessary calls.

For a no-API-cost local alternative, install Ollama and use a local vision model. This machine already
had `gemma3:12b`, so no additional model download was required:

```powershell
ollama list
```

Then replace the provider/model arguments in the command below with:

```powershell
--provider ollama --vision-model gemma3:12b --workers 1
```

The Ollama adapter sends the same image, OCR context, and Pydantic JSON schema to the local
`http://localhost:11434/api/chat` endpoint. No document content is sent to OpenAI. Set
`FINSIGHT_OLLAMA_URL` or pass `--ollama-url` if Ollama is hosted elsewhere. Local inference has no API
charge but consumes local RAM/GPU resources and can be slow on CPU. Use one worker for the installed
12B model on this hardware; two concurrent requests were slower and stalled the command wrapper.

### Run with the existing Intel exports

```powershell
python -m finsight.agents.narrative.corpus.pdf.docling_charts `
  "data/narrative/raw/INTC/official_ir/2025-04-24_earnings_presentation_Q1_2025_Earnings_Deck_INTC.com.pdf" `
  "data/narrative/processed/INTC/intel_q1_2025_enhanced" `
  --docling-json "tmp/pdfs/docling_trial/output_full/2025-04-24_earnings_presentation_Q1_2025_Earnings_Deck_INTC.com.json" `
  --markdown "tmp/pdfs/docling_trial/output_full/2025-04-24_earnings_presentation_Q1_2025_Earnings_Deck_INTC.com.md" `
  --chart-pages 7-10 `
  --provider openai `
  --vision-model gpt-5.4 `
  --dpi 240 `
  --workers 2
```

Omit `--docling-json` and `--markdown` to run Docling first. Use `--provider none` for detection and
cropping without API calls; image placeholders remain intact in that mode.

The outputs are:

- `<document>.enhanced.md`;
- `<document>.enhanced.json`;
- PNG crops and per-chart cache JSON under `charts/`;
- `chart_validation_report.json`;
- `chart_validation_values.csv`;
- `chart_output_comparison.json`.

Abbreviated, manually verified page-7 output examples are available in:

- `docs/narrative/examples/intel_q1_2025_chart.example.md`;
- `docs/narrative/examples/intel_q1_2025_chart.example.json`.

### Intel pages 7-10 test

Detection against the existing Intel JSON found exactly five target regions and rejected the separate
page-10 photograph:

| Page | Chart ID | Content | Result |
|---:|---|---|---|
| 7 | `p007-picture-005` | Revenue, non-GAAP GM%, non-GAAP operating income/loss | accepted |
| 8 | `p008-picture-006` | CCG | accepted |
| 8 | `p008-picture-007` | DCAI | accepted |
| 9 | `p009-picture-008` | Intel Foundry | accepted |
| 10 | `p010-picture-009` | All Other | accepted |
| 10 | `#/pictures/10` | Photograph | rejected and preserved |

All accepted regions rendered correctly at 240 DPI. The following values were manually read from the
crops and are ordered Q1'24, Q2'24, Q3'24, Q4'24, Q1'25.

| Page/chart | Metric | Unit/basis | Expected displayed values |
|---|---|---|---|
| 7 | Revenue | USD B | `$12.7`, `$12.8`, `$13.3`, `$14.3`, `$12.7` |
| 7 | GM% | non-GAAP percent | `45.1%`, `38.7%`, `18.0%`, `42.1%`, `39.2%` |
| 7 | Op Income/Loss | non-GAAP USD B | `$0.7`, `$0.0`, `-$2.4`, `$1.4`, `$0.7` |
| 8 CCG | Revenue | USD B | `$8.3`, `$8.1`, `$8.2`, `$8.8`, `$7.6` |
| 8 CCG | Op Income | USD B | `$2.8`, `$2.6`, `$2.9`, `$3.2`, `$2.4` |
| 8 CCG | OM% | Percent | `34.1%`, `32.4%`, `36.0%`, `36.4%`, `30.9%` |
| 8 DCAI | Revenue | USD B | `$3.8`, `$3.8`, `$4.1`, `$4.4`, `$4.1` |
| 8 DCAI | Op Income | USD B | `$0.4`, `$0.2`, `$0.4`, `$0.4`, `$0.6` |
| 8 DCAI | OM% | Percent | `10.9%`, `6.4%`, `9.2%`, `8.6%`, `13.9%` |
| 9 Foundry | Revenue | USD B | `$4.4`, `$4.3`, `$4.3`, `$4.3`, `$4.7` |
| 9 Foundry | Op Income/Loss | USD B | `-$2.4`, `-$2.8`, `-$5.8`, `-$2.2`, `-$2.3` |
| 9 Foundry | OM% | Percent | `-56.0%`, `-65.4%`, `-133.6%`, `-51.8%`, `-49.7%` |
| 10 All Other | Revenue | USD M | `$643`, `$881`, `$964`, `$1,113`, `$943` |
| 10 All Other | Op Income/Loss | USD M | `-$170`, `-$46`, `$5`, `$154`, `$103` |
| 10 All Other | OM% | Percent | `-26.4%`, `-5.2%`, `0.5%`, `13.8%`, `10.9%` |

Page-8 OCR captured DCAI values but not in visual series order. This demonstrates why the image/legend
relationship must be read rather than trusting raw OCR order.

### Validation procedure

1. Run the pipeline and open `chart_validation_values.csv`.
2. Open each `crop_path` and cited PDF page side by side.
3. Check title/type, legend mapping, unit scale, accounting basis, and period order.
4. Check every displayed value, including minus signs, parentheses, commas, percent signs, and decimal
   precision. Never approve an inferred value.
5. Resolve every `null`, ambiguity, low-confidence warning, duplicate, or OCR mismatch. OCR mismatch is
   a review signal rather than automatic rejection because the image may be clearer than OCR.
6. Add reviewer initials/date or approval in the CSV's `approved` column and retain it with the outputs.
7. Run `python -m pytest tests/test_docling_charts.py -q`.

### Original versus enhanced output

The original Intel Markdown has 11 image placeholders; five are target charts. A successful live run
should replace those five, preserve six unrelated images, and recover 75 value points across pages 7-10.

The paid OpenAI test on 2026-10-15 was blocked because the configured account returned
`credit_balance_exhausted`. The same crops were then processed locally with Ollama `gemma3:12b`:

| Measure | Original | Local Ollama run |
|---|---:|---:|
| Image placeholders | 11 | 6 |
| Detected financial chart regions | 0 structured | 5 |
| Successful Vision extractions | 0 | 5 |
| Failed or skipped charts | n/a | 0 |
| Recovered value points | 0 | 75 |
| Pages with recovered charts | none | 7, 8, 9, 10 |
| Unrelated images preserved | 11 placeholders | 6 placeholders |

All 75 local-model values were compared with the rendered chart crops and matched the manual reference
above. The model correctly preserved negative signs, percent signs, currency symbols, commas, decimal
precision, and period order. It conservatively marked accounting basis as unknown where the chart did
not state GAAP/non-GAAP and retained manual-review warnings for a few units or OCR mismatches.

The final local outputs are under `tmp/pdfs/docling_chart_ollama_test/`. Successful records are cached,
so rebuilding the combined enhanced Markdown and JSON takes only a few seconds.

## Recommended production design

Use a routed extraction strategy rather than replacing `pypdf` for every PDF:

1. Run the existing `pypdf` parser first.
2. Measure extracted words per page.
3. Keep native extraction for normal text-bearing pages.
4. Route zero-text or abnormally low-text PDFs/pages through Docling OCR.
5. Use full-page OCR for fully image-based documents.
6. Use PDF-aware layout-region OCR for mixed PDFs where native text should be retained.
7. Preserve page numbers, bounding-box provenance where available, extraction method, and OCR warnings.
8. Mark OCR-derived evidence with an extraction-quality flag.
9. Keep Docling's local Granite chart extraction disabled by default; use the modular remote Vision path
   for selected chart candidates.
10. Require source-page verification for numerical claims derived from OCR or complex tables.

Before rebuilding production indexes, add a labeled test set containing representative scanned text, KPI cards, tables, and charts. Compare retrieval quality and numeric accuracy between native extraction and the Docling fallback.

## Status

- Docling OCR/layout trial: successful
- Full image-only Intel deck recovered: successful
- Visual inspection: completed
- Modular Docling/chart pipeline: implemented
- Existing chunking/index integration: intentionally unchanged
- Local Ollama Vision extraction: successful for all five target charts
- Paid OpenAI extraction: blocked by exhausted API credits, but no longer required for this document
- Existing processed corpora/indexes: unchanged
- Bulk OCR of all warning pages: not run

## References

- [Docling overview](https://docling.ai/)
- [Docling CLI options](https://github.com/docling-project/docling/blob/main/docs/reference/cli.md)
- [Docling pipeline options](https://docling-project.github.io/docling/reference/pipeline_options/)
- [Docling OCR FAQ](https://docling-project.github.io/docling/faq/)
- [OpenAI Responses API](https://developers.openai.com/api/reference/responses/overview)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)
