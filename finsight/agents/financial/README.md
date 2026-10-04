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
| `agent.py` | `XbrlFinancialAgent.gather(sub_claim, cutoff_date, feedback) -> list[EvidenceItem]`: the real agent |

## Getting the data (run from the repo root)

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"  # SEC requires a contact
python -m finsight.agents.financial.pull  # writes data/sec/raw and data/sec/processed
python -m finsight.agents.financial.coverage --cutoff 2025-07-31 # produces the coverage report
```

## 1. What the agent does

Someone makes a claim such as "NVIDIA's revenue grew strongly". The agent looks up the real numbers the company filed with the SEC, does the arithmetic in Python, and says whether the numbers support or challenge the claim.

Example of what it returns:

> **SUPPORTING.** Revenue was $44.06bn in the quarter ended 2025-04-27 vs $26.04bn a year earlier: +69.2% year on year.

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
5. **Debt is not mapped.** "Debt" or "leverage" in a claim is answered with total liabilities divided by equity. For companies with negative equity, such as McDonald's and AMC, there is no answer.
6. **Wording rules are simple.** The agent picks what to calculate from keywords. It doesn't understand "is not growing" or cautious wording.
7. **Stock splits are flagged, not adjusted.**
8. **The cut-off uses the date only.** A report filed late in the evening on the cut-off day still counts.

Items 1 and 2 are the job of the Narrative Evidence Agent which reads the report text.

## 8. Where things stand

| Item | Status |
|---|---|
| Double check checks.py, tools.py, agents.py | Not done yet |
| Numbers checked by hand against the filings | Not done yet |
| McDonald's share-count multiplier | Not switched on yet |
| Agent registered in the pipeline | Not yet: needs Hung to add the companies and register it |
| Style check (`ruff`) | 20 trailing-space errors to fix with `ruff check finsight tests --fix --unsafe-fixes` |