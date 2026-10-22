# FinSight Financial Agent

| File | What it does |
|---|---|
| `config.py` | List companies within scope as well as the minimum period covered for the data pulled |
| `pull.py` | Download SEC Company Facts for the companies in `config.py` |
| `normalize.py` | Flattens the downloaded company files |
| `concepts.py` | Concepts mapped to us-gaap tags |
| `coverage.py` | Report of which concepts and metrics are available per company |
| `store.py` | Looks facts up. The cut-off filter (`filed <= cutoff`) is used here |
| `tools.py` | Where financial calculations happen |
| `schema.py` | Internal result format (`FinancialEvidence`); `agent.py` converts it to the team's `EvidenceItem` |
| `planner.py` | The reasoning loop: the model reads the claim and decides, one action at a time, which figures to compute; each result is shown to it before its next step |
| `claim_reader.py` | Reads the claim: which figures the post line states, for which company and period, narrowed to what the user's prompt asks about |
| `agent.py` | `XbrlFinancialAgent.gather(sub_claim, cutoff_date, feedback) -> list[EvidenceItem]`: computes each figure the reader found |
|---|---|
| `test_calculations.py` | Checks whether the calculation logic is correct |
| `test_metrics.py` | Checks whether the tags pulled from the SEC files make sense |
| `test_agent.py` | Checks how the agent reads claims and which figures it reports (based on the eval items) |
| `test_planner.py` | Checks the reasoning loop with a scripted model: actions carried out, results fed back, guardrails, fallback |
| `test_scope.py` | Checks that sub-claims sharing one sentence each report only their own measure, and that the model must read before it acts |
| `run_eval.py` | Runs the agent on the dev eval items (the lines from `pre_agent_steps.json`, the prompts from `claims_dev_final.json`) and prints the results next to the expected ones |
| `run_claims.py` | Runs the agent over every stored line in `claims_updated.jsonl` (cut-off = post date) and writes `agent_run.csv` |


## Getting the data (run from the repo root)

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"  # SEC requires a contact
python -m finsight.agents.financial.pull  # writes data/sec/raw and data/sec/processed
python -m finsight.agents.financial.pull --offline # when concepts.py and config.py have been edited 
and facts.csv and filings.csv need to be rebuild
python -m finsight.agents.financial.coverage --cutoff 2026-07-31 # produces the coverage report
python -m finsight.agents.financial.test_metrics --cutoff 2026-07-31
python -m finsight.agents.financial.test_calculations
python -m finsight.agents.financial.test_agent
python -m finsight.agents.financial.test_planner
python -m finsight.agents.financial.test_scope
python -m finsight.agents.financial.run_eval --steps pre_agent_steps.json --items claims_dev_final.json --trace
python -m finsight.agents.financial.run_claims --claims data/influencers/claims/claims_updated.jsonl --with-number
```

## 1. What the agent does

Someone makes a claim such as "NVIDIA's revenue grew 70%". The agent works out what the claim states, looks up the real numbers the company filed with the SEC, does the arithmetic in Python, and reports them with the filing they came from.

It **does not give a verdict**. Every piece of evidence it returns has stance NEUTRAL. Deciding whether the claim is right (did revenue grow? is 70% close enough?) is the job of the Auditor and the Overall assessment. Supporting and counter-evidence from report text comes from the Narrative Evidence Agent.

### Input (from the orchestrator)

`gather(sub_claim, cutoff_date, feedback) -> list[EvidenceItem]`

| Field | What it must contain | Example |
|---|---|---|
| `sub_claim.sub_claim_id` | Unique per sub-claim | `"q11-1"` |
| `sub_claim.company_ticker` | One of the 16 tickers in `config.py` | `"GOOGL"` |
| `sub_claim.text` | **One claim**: one company, one measure named in words, with the period and stated figure if the influencer gave them | `"Alphabet's revenue grew 22% year over year to $110 billion last quarter"` |
| `cutoff_date` | Resolved by the orchestrator; `None` = latest data | `date(2026, 4, 30)` |
| `feedback` | `None`, or the Auditor's reason on the retry | `"period is quarter ended 2026-06-30 but the claim names Q1 2026"` |
| `sub_claim.sub_claim_type` | Always present (required by `SubClaim`). Shown to the model as a hint only: it comes from keywords and is often wrong | `HISTORICAL_GROWTH` |
| `sub_claim.measure` (optional) | When set, limits what this sub-claim may report. Set on a section 6 expansion (`profitability`, `cash_generation`, `liquidity`, `leverage`) and on a growth child split off a margin sentence (`operating_income`, `diluted_eps`, `operating_cash_flow`) | `"liquidity"` |
| `sub_claim.parent_span` (optional) | The phrase an expansion came from; shown to the model | `"a very strong financial position"` |
| `sub_claim.user_prompt`, `sub_claim.period` (optional) | Not in `SubClaim` yet (schema 0.5 forbids extra fields). Used when present; `period` is the last resort when the words name none | `"trailing 12-month"` |

One sentence can arrive as several sub-claims with the same `text` (an expansion, or a margin sentence plus its growth child). `measure` tells them apart, so each reports only its own figure. The type alone never limits what is reported: on `claims_updated.jsonl` a type filter dropped a stated measure in 401 of 1,733 financial sub-claims.

### How it understands the claim

An LLM reads the claim and plans, Python computes, and code enforces the rules.

| Step | Who | What happens |
|---|---|---|
| 1. Understand | The model (team LLM client, `planner.py`) | Its first action is always `understand`: from the words alone it lists each statement about the company (measure, period, stated figure, direction, in this sub-claim's scope or not). Code refuses any other action before it. Only then is the rule-based reading shown, as a cross-check, so the rules cannot steer the reading |
| 2. Act | The model | Picks one action: `report_growth`, `report_level`, `report_streak`, `list_periods`, `skip` or `finish`, with a short "thought" saying why. A measure outside the sub-claim's `measure` scope is refused |
| 3. Compute | Python (`tools.py`) | Carries out the action. The formulas, the cut-off, the window and the segment guardrail are enforced here; the model never writes a number |
| 4. Observe | The model | Sees the result ("OK: +21.8% ...", "NO FIGURE: outside the window", "REFUSED: segment") and chooses the next action, until it finishes |
| 5. Retry | The model | On a retry, the Auditor's feedback is in its input, so it can choose a different period or measure |
| Fallback | Rules (`claim_reader.py`) | When no model answers in the action format (the mock provider, no API key, the spend limit), the rules decide |

`agent.traces[sub_claim_id]` holds the model's steps (thought, action, result); `agent.readings[sub_claim_id]` holds its `understand` parts; `agent.modes[sub_claim_id]` says `llm` or `rules`. Untrusted text (the claim, the prompt, the feedback) only goes inside the client's data blocks.

**What it reads from the words** (the model's instructions and the rule-based reading follow the same rules)

| Reads | Example |
|---|---|
| The measure | "operating profit" → operating income; "earnings" → net income; "margins" → operating then gross margin; "profitability" → operating and net margin; "Google grew 24%" → revenue |
| Another company's figure is skipped | "Alphabet's operating margins are 34% and AMD's are around 17%" → Alphabet only |
| Left to the narrative agent, with a note | A segment or product ("Google Cloud revenues"), a run rate, gross profit, adjusted / non-GAAP figures, net income excluding gains, a debt amount, a forecast ("earnings are set to grow 75% next year") |
| Ignored words | "earnings multiple of 20 times", "reports earnings on Tuesday", "taking profits", share prices |

**Measures** (knowledge_layer.yaml v0.6)

| Measure | What it reports |
|---|---|
| Revenue, operating income, operating cash flow, diluted EPS, net income | Growth, year on year |
| Gross, operating and net margin; free cash flow; free cash flow margin; current ratio; cash ratio | The figure and the same figure a year earlier. Free cash flow also gets its growth rate when the claim states a percentage change ("down 91%") |
| Debt to equity | Nothing: debt is not mapped in the SEC data |

**Which period**

| Wording | What the agent reports | Example |
|---|---|---|
| No period named | The latest fiscal year vs the year before. The trailing twelve months go in the notes as context only (team ruling, 18 Oct) | "Netflix generates $12 billion in free cash flow" |
| Trailing twelve months | The trailing twelve months vs the twelve months a year earlier | "over the last 12 months", "TTM" |
| "Recent" or "quarter" | The latest quarter vs the same quarter a year earlier | "last quarter", "has recently turned negative" |
| A margin or ratio stated in the present | The latest quarter | "Alphabet's operating margin is 34%" |
| "Last two quarters" | Each of the two latest quarters vs a year earlier | |
| A named period | That period | "Q1 2026", "Q4 of 2025", "the March quarter", "Q1 FY2026", "fiscal 2025" |
| A streak within the window | One growth item per quarter | "four straight quarters of double-digit growth" |

### Output (to the Auditor)

A list of `EvidenceItem`s for the claim's measure; usually one.

| Field | Value |
|---|---|
| `evidence_id` | `XBRL-<ticker>-<metric_name>-<period end>` |
| `stance` | `NEUTRAL` |
| `content` | Growth: `Revenue was $109,896m in the quarter ended 2026-03-31 vs $90,234m in the quarter ended 2025-03-31 (+21.8% YoY).` Margin: `Operating margin was 40.6% in the quarter ended 2026-03-31 vs 41.5% in the quarter ended 2025-03-31.` Ratio: `Current ratio was 1.50x as at 2026-06-30 vs 2.00x as at 2025-06-30.` |
| `source_tier`, `doc_type` | `TIER_1`; `10-Q` or `10-K` (amendments are reported as the base form) |
| `company` | The ticker (`COMPANY_FIELD` in `agent.py`; switch to `"name"` if the Auditor wants "Alphabet") |
| `period` | `quarter ended …`, `year ended …`, `twelve months ended …`, `as at …` |
| `publication_date`, `url`, `citation_span` | The filing date, the SEC filing index, `XBRL us-gaap:…; filings: 10-Q …` |
| `computed_value`, `metric_name` | Growth as a decimal with `<measure>_growth_yoy`; levels with `gross_margin`, `operating_margin`, `net_margin`, `free_cash_flow`, `fcf_margin`, `current_ratio`, `cash_ratio` |
| `comparison_value`, `comparison_period` | Growth: the amount for the period. Level: the figure a year earlier |

**When it returns nothing, or skips part of a claim**, `agent.reasons[sub_claim_id]` says why (the orchestrator copies it to `no_evidence_reason`):

| Case | Example |
|---|---|
| Outside the window: the latest 10-K plus the latest four quarters filed by the cut-off | "Q1 2025" when the latest quarter is Q2 2026; "twelfth straight quarter" |
| A period not filed by the cut-off | "the June 2026 quarter" with a 1 Apr 2026 cut-off |
| "The latest quarter" is unclear: a newer quarter has ended and its results are probably out, but its filing is not in the data | Meta with a 29 Apr 2026 cut-off (10-Q filed on the 30th) |
| Not calculated here | Segment, run rate, gross profit, adjusted figures, forecasts, debt to equity |
| Context figures | "Context only (not evidence): Revenue was … in the twelve months ended …" |

## 2. Where the numbers come from

| Term | What it is | Everyday comparison |
|---|---|---|
| 10-K and 10-Q | The annual and quarterly reports a company files with the SEC. They contain text and numbers. |
| XBRL | Labels attached to every number in those reports so a computer can read them. |
| Company Facts | One file per company that the SEC builds by collecting all those labelled numbers. |

The agent uses Company Facts. It keeps only numbers from four report types: 10-K, 10-Q, and corrected versions of each (10-K/A and 10-Q/A).

**What this choice leaves out**

- **Only US accounting rules (US GAAP).** A foreign company that reports under international rules, such as SAP or Shell, would return nothing.
- **Banks and insurers don't fit.** Their reports have no "gross profit" or "current assets" so several calculations would come back empty.

## 3. Concepts and tags

Companies don't all use the same label for the same thing. One company calls its sales `Revenues`; another uses `RevenueFromContractWithCustomerExcludingAssessedTax`. A **concept** is our own name for the idea and **tags** are the SEC labels that count as that idea.

Example, the line for inventory in `concepts.py`:

```python
"Inventory": ("instant", ["InventoryNet", "InventoryRealEstate"], "USD", True),
```

| Part | Value here | Meaning |
|---|---|---|
| Name | `Inventory` | The name the tools and teammates use. |
| Kind | `instant` | Insant means as of date (only has end date) while durationg is over a period of time (has start and end date) |
| Tags | `InventoryNet`, `InventoryRealEstate` | The SEC labels that count as inventory. The first one that has a value is used. |
| Unit | `USD` | Dollars. Other units are shares, and dollars per share. |
| Additive | `True` | A missing quarter may be worked out by subtraction. This is `False` for share counts and EPS. |

Opendoor is why there are two tags. Its inventory is houses, filed under `InventoryRealEstate`. Before that tag was added, Opendoor's inventory came back empty.

**How much to trust each concept**

| Confidence | Concepts | Why |
|---|---|---|
| Highest | Total assets, liabilities and equity total, current assets, current liabilities, operating cash flow, operating income, gross profit, inventory, diluted shares, diluted EPS | One label each, with one meaning. |
| Good, with a known wrinkle | Net income, shareholders' equity, total liabilities, cash | The first-choice label is solid. The backup labels for net income and equity include minority owners' share, so they are slightly different numbers. |
| Weakest | Revenue, cost of revenue, capital expenditure | Each has several valid labels, and the order they are tried in is a judgement call. |

There are 17 concepts. Debt, interest expense and short-term investments are deliberately left out because companies label them too differently to trust without checking each company.

## 4. The cut-off date and restatements

Every question comes with a cut-off date. The agent only uses reports filed on or before that date so it never "knows" something the user couldn't have known.

Example: asked about NVIDIA with a cut-off of 31 July 2025, the agent used the quarter that ended in April. The July quarter existed but its report wasn't filed until late August so it was ignored.

**Restatements.** Companies sometimes correct an old number in a later report. The rule is: use the most recently filed version as long as it was filed by the cut-off.

| Report | Filed on | Revenue for Jan to Mar 2024 |
|---|---|---|
| Original | 25 Apr 2024 | 100 |
| A year later, with the old quarter corrected | 25 Apr 2025 | 101 |

- Cut-off of 31 Dec 2024: the answer is **100**. The correction didn't exist yet.
- Cut-off of 31 Jul 2025: the answer is **101**.

So the same period can give different answers at different cut-offs. That is correct but it can look like a mistake when two runs are compared.

**What the rule does not catch**

- **A correction filed under a different label.** After Smartbird (ticker BIRD) changed its business, it began reporting cash flow under a "continuing operations" label. The old and new figures are not recognised as two versions of one number.
- **A changed financial year.** If a company moves its year-end, the corrected periods have different dates and are treated as separate periods.
- **No flag.** The result doesn't say that a number was corrected or what it was originally.

## 5. Quarters that are never reported on their own

Companies never report the fourth quarter by itself and cash flow is only reported as a running total for the year. The agent works these out by subtraction and marks the result as "derived".

Example: fourth quarter = full year minus the first nine months.

Share counts and earnings per share are never worked out this way. They are averages so subtracting them gives nonsense. A fourth-quarter share count simply comes back as "not available".

## 6. Awkward situations the code handles

**Dates and filings**

| Situation | In plain terms | Example |
|---|---|---|
| Report filed exactly on the cut-off date | It counts. | Cut-off 31 Jul, report filed 31 Jul: used. |
| Cut-off typed in an odd format | Rejected with an error instead of guessing. | "31 Jul 2025" is refused; "2025-07-31" is accepted. |
| Two reports filed on the same day | The one with the higher filing number is treated as later. | An original and a correction filed the same day. |
| The newest data is old | A warning is added. | Cut-off in mid-2026 but the latest report is from mid-2025. |

**Comparing two periods**

| Situation | In plain terms | Example |
|---|---|---|
| A year with 53 weeks | The matching period is still found with a warning that one is a week longer. | A 14-week quarter compared with a 13-week one. |
| The company changed its year-end | No comparison is made. | Better no answer than a wrong one. |
| The company switched labels between years | The growth is calculated with a warning. | Alphabet's revenue moved between two labels in 2025. |
| Last year's number was zero or negative | No growth rate is given. | Growth from a loss of 20 to a profit of 50 is not a meaningful percentage. |
| Share count suddenly doubles or halves | A warning about a possible stock split. | A 10-for-1 split makes the count look ten times larger. |

**Mixed versions of numbers**

| Situation | In plain terms | Example |
|---|---|---|
| Working out a quarter by subtraction | Both numbers are taken as they stood on the same date so they match. | Tesla: a corrected first-half figure (2,790m) must not be subtracted from an uncorrected nine-month total (4,774m). |
| Only a later version of one number exists | Still calculated with a warning. | |
| One number in a ratio was corrected and the other wasn't | A warning is added. | Revenue corrected later, gross profit not. |

**The data files**

| Situation | In plain terms | Example |
|---|---|---|
| The label list was edited but the data wasn't rebuilt | Rows with labels that are no longer listed are ignored. | Fix: run `pull --offline`. |
| Broken rows in the SEC file | Dropped and counted during the pull. | A value that isn't a number or a report dated before its period ended. |
| A company files a number on a different scale | A per-company multiplier in `config.py`. | McDonald's files 718 for its share count where others would file 718,000,000. |

The McDonald's multiplier is supported by the code but not switched on yet: `VALUE_SCALE` still needs adding to `config.py`.

## 7. What the agent cannot do

1. **No product or segment numbers.** Company Facts only has company-wide totals. It can't see NVIDIA's data-centre revenue or GameStop's collectibles sales.
2. **No company-defined measures.** Adjusted EBITDA, comparable sales, subscribers, attendance and homes sold are not in this data at all.
3. **No share price.** So no valuation measures such as price-to-earnings.
4. **Gross profit is missing for some companies.** AMC and Krispy Kreme don't report it. McDonald's quarterly figure looks like only part of its costs and shouldn't be trusted.
5. **Debt is not mapped.** A `LEVERAGE` sub-claim (debt to equity) gets no evidence, with the reason given. The rule book caps it at Insufficient Evidence.
6. **Reading the claim can go wrong.** The model can misread a claim, and the rules (used without a model) can miss unusual wording. `run_eval.py --trace` shows what was found and why for each line. A segment word just before a measure ("cloud revenue", "Services margin") makes the figure a segment one, and a plain "Q1 2026" is read as the calendar quarter even for a company with a different fiscal year.
7. **Stock splits are flagged, not adjusted.**
8. **The cut-off uses the date only.** A report filed late in the evening on the cut-off day still counts.

Items 1 and 2 are the job of the Narrative Evidence Agent which reads the report text.

## 8. Where things stand

| Item | Status |
|---|---|
| Calculation tests (`test_calculations.py`) | Done: 46/46 pass (including trailing twelve months) |
| Data checks (`test_metrics.py`) | Done: 0 failures; MELI revenue warnings are expected (total vs contract revenue) |
| Planner tests (`test_planner.py`) | Done: 15/15 pass with a scripted model (each script starts with `understand`). Not yet run with a real model |
| Scope tests (`test_scope.py`) | Done: 9/9 pass. Sub-claims built with the team's claim builder; real lines from eval items 26 and 34 |
| Agent tests (`test_agent.py`) | Done: 56/56 pass (reading claims, measures, periods, trailing twelve months, window, the output shapes agreed with the Auditor) |
| Numbers checked by hand against the filings | Done: Alphabet 21.8% and 24.2%, Meta 33.1%, Microsoft 18.3%, Nvidia full-year operating margin 62.4% to 60.4% all match |
| `comparison_value` / `comparison_period` on EvidenceItem | Agreed with the team, but not yet in the repo's `contracts/schemas.py` (0.5): without them the agent drops the values silently and 6 agent tests fail. Needs Hung |
| Auditor uses `computed_value` and `comparison_value`; the Financial agent's stance is always NEUTRAL | Agreed |
| Sub-claims arrive as one claim each (`sub_claim_id`, `company_ticker`, `text`) with `cutoff_date` and `feedback` | To agree with Hung and Amelia |
| Auditor confirms: `company` as ticker or name, `doc_type` 10-Q/10-K, free cash flow in dollars, `current_ratio` / `cash_ratio` / `free_cash_flow_growth_yoy` names, a `no_evidence_reason` field | To agree with the Auditor owner and Hung |
| Agent registered in the pipeline | Not yet: needs Hung to add the companies and register it |
| Style check (`ruff`) | 20 trailing-space errors to fix with `ruff check finsight tests --fix --unsafe-fixes` |