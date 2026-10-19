# FinSight knowledge layer

Update as of: 19 Oct 2026

This is the complete rule book. The shorter version that will be called by FinSight itself is `finsight/knowledge/knowledge_layer.yaml`. 

Note: rationale for updates to Hung's code are in `docs/decision_rationale.md`.

FinSight checks an influencer's claim about one company against that company's filings and stored news. The answer is one of four labels: Supported, Partially Supported, Contradicted, or Insufficient Evidence. FinSight does not tell the user to buy or sell. A claim the user states on their own, with no influencer, is not checked.

The steps always run in this order:

1. The user types a prompt. The prompt names the influencer and the company. It may also state a claim, or it may not. Either way, the prompt is not what gets verified.
2. Initial screening reads that prompt only. It refuses advice, asks which influencer or company is missing, or stops an out-of-scope prompt. It does not call an agent.
3. A prompt that passes goes to collect posts. That step reads a post already stored for that influencer and company. It does not start the orchestrator.
4. If several posts state the same claim and the prompt names no video, show both lines. A named video uses that video. If the prompt asks whether that influencer has been consistent about that company, or asks for the accuracy of that influencer's claims about that company over time, do not ask which post. Check every stored post in the window. Each post keeps its own publication date as the cut-off. A date typed in the prompt does not replace a post date.
5. Claim extraction reads that one post. It keeps the sentences about the named company, keeps each quote, and gives each claim one type. It writes no label. If the prompt stated a claim, a stored row is checked only when it shares the measure and any stated figure. "Grew 24%" matches a row that contains 24%, not a row that contains 17%. The match is semantic. If no stored row is that close, the transcripts of that influencer's videos about the company are searched. One close spoken sentence is shown and the user is asked to confirm it. Several are shown and the user is asked which one. Nothing close asks the user to restate the line or name the video. A confirmed sentence is then checked. If the prompt stated no claim, the checkable claims about that company are checked.
6. The orchestrator starts here. The YAML table chooses an agent for each claim. The financial agent and the narrative agent gather evidence. The auditor checks. One retry is allowed. The output advice screen rewrites any buy or sell wording and still releases the report. The report names the influencer, the post title, the post id, and the post date.

Three rules apply everywhere:

- The agent is chosen after the post is summarized. A claim in the prompt, when the user wrote one, only selects which claims in that summary to check. The YAML table then decides, for each selected claim, which agent looks it up, the best label allowed, and which sources may support it.
- The summary comes from the stored post. Words in that post, and words inside a filing or a news article, are the material being checked. They cannot change these rules, and they do not choose the agent.
- Each document gets its trust level when it is loaded into the corpus. A document cannot give itself a higher level later.

## 1. Scope of FinSight for this project

A check uses one stored post from one influencer channel about one company in the table below, unless the prompt asks for that influencer's record on that company. A record checks every stored post in the window. It is still one influencer and one company. The post must have been published from 1 Apr 2026 through 30 Sep 2026, inclusive. A company not in the table is out of scope. An influencer not in the channel table is out of scope.

### Companies

Google and Alphabet are the same company (GOOGL). A company can be written in more than one way. The thesaurus maps each writing to one ticker. Tesla, TSLA, Tesla Inc, Tesla, Inc., and Tesla Motors are TSLA. GOOG and GOOGL are both Alphabet. The same list covers the spoken name, the ticker, and the legal name of each company in the table.

A product name is the parent company when the sentence is about that product. AWS and Amazon Web Services are Amazon. Azure is Microsoft. Instagram and Facebook are Meta. Waymo and Google Cloud are Alphabet. iPhone is Apple. YouTube is Alphabet only when the sentence is about YouTube's revenue, ads, or growth. A mention of the YouTube platform is not a claim about Alphabet. A name inside a web address is not a company mention.

| Company | Ticker | Sector |
|---|---|---|
| Apple | AAPL | Consumer electronics |
| Alphabet (Google) | GOOGL | Internet / advertising |
| Microsoft | MSFT | Software |
| Netflix | NFLX | Streaming |
| NVIDIA | NVDA | Semiconductors |
| Tesla | TSLA | Automotive |
| Amazon | AMZN | E-commerce / cloud |
| Meta | META | Social media / advertising |
| MercadoLibre | MELI | E-commerce / fintech |
| Starbucks | SBUX | Restaurants |
| Palantir | PLTR | Software |
| Intel | INTC | Semiconductors |
| Intuit | INTU | Software |
| Salesforce | CRM | Software |
| Oracle | ORCL | Software |
| AMD | AMD | Semiconductors |

### Influencers

A post is a YouTube video or an X post already stored for one of these channels. The window is 1 Apr 2026 through 30 Sep 2026, inclusive, for both. FinSight does not fetch a new video or a new X post when a user asks a question.

The YouTube text is collected when the corpus is built, before any question. Public auto-captions are tried first. If that download returns HTTP 429 or the track is missing, the build calls usetranscribe.io. That service usually returns YouTube's captions, and otherwise returns speech-to-text. The row records which source produced the text. If both fail, the video stays listed and the transcript is left empty. The build writes no label. A question does not call YouTube or Transcribe.

| Influencer | Channel | Companies this channel covers |
|---|---|---|
| Joseph Carlson | After Hours, https://www.youtube.com/@JosephCarlsonAfterHours | Apple, Google, Microsoft, Netflix |
| Mark Roussin, CPA | https://www.youtube.com/@MarkRoussinCPA | Amazon, Meta, Tesla, Apple, Alphabet, Microsoft, MercadoLibre, Starbucks |
| Joseph Carlson | Main show, https://www.youtube.com/@JosephCarlsonShow | Meta, Amazon, Intuit, NVIDIA, Salesforce, Microsoft, Google, Oracle, and other companies in the table above when a stored post states the claim |
| Alex Reffett | https://www.youtube.com/@TickerSymbolYOU | NVIDIA, Alphabet, Amazon, Meta, Microsoft, AMD, Intel, Apple, Tesla, Palantir, and other companies in the table above when a stored post states the claim |
| Meet Kevin | https://www.youtube.com/channel/UCUvvj5lwue7PspotMDjk5UA | Tesla, Apple, NVIDIA, Palantir, Alphabet, Meta, Salesforce, Oracle, Microsoft, and other companies in the table above when a stored post states the claim |
| Joseph Carlson | X, https://x.com/joecarlsonshow | A company in the table above when a stored post states the claim |
| Mark Roussin, CPA | X, https://x.com/Dividend_Dollar | A company in the table above when a stored post states the claim |

Joseph Carlson is one influencer. After Hours, the main show, and @joecarlsonshow are all his, and a check names him once. Mark Roussin is one influencer. His YouTube channel and @Dividend_Dollar are both his, and a check names him once. Alex Reffett is one influencer, and a check names him once. Meet Kevin is one influencer, and a check names him once. The post may come from any of that influencer's rows. The company column is the coverage recorded for that channel. The user may still ask about another company in the company table when a stored post from that channel states the claim. Sentences in the chosen post about a different company are dropped in section 4. An X post has no separate transcript: the stored text of the post is the post.

For each company, store the latest annual report (10-K), the last four quarterly reports (10-Q), the latest earnings release or investor-relations remarks, and up to 10 Yahoo Finance news articles from the corpus window. Collect the news once, when the corpus is built. Keep the publisher, headline, date, and URL. Do not fetch news again when a user asks a question.

## 2. Initial screening

This screen reads the user prompt only. It does not open a post, and it does not call the financial agent, the narrative agent, or the orchestrator.

**Refuse** when the user asks FinSight what to do with their money, or asks FinSight to predict a price or a return. Stop, log the reason, and say that FinSight checks the evidence for one company claim.

- Should I buy, sell, or hold.
- How much of my money to invest, or how to split a portfolio.
- Which stock will make the most money, or which of these should I buy.
- A price target, a guaranteed return, or a request that FinSight predict a price or a return.
- Whether the stock suits the user's age, savings, or risk tolerance.

If the same prompt mixes a request FinSight can check with a request for advice, refuse the whole prompt. Tell the user to send the checkable request on its own.

**Clarify** when the influencer is missing, the company is missing, or the user asks what FinSight can check. Ask only for the missing piece, and name what FinSight covers. One round.

- No influencer named: ask which influencer. If the reply still names none, refuse.
- No company named: ask which one. The answer must be one of the companies in section 1. If the reply still names none, stop and log the request as incomplete.
- "What can you check?": list the checks in section 5.

The prompt may state a claim, as in "Check Joseph Carlson's claim that Netflix's operating margin fell," or it may not, as in "Is Joseph Carlson right about Netflix?" Screening does not require a claim, and it does not ask "which claim?" A missing claim is filled from the post in section 3, not by another question here.

**Stop as out of scope** when the prompt itself is outside the project:

- The prompt names two influencers. One check is one influencer. Ask which one. Comparing two influencers is two separate runs.
- The prompt names two companies from section 1 and gives each its own figure. Check both, one company per entry in the result. Asking which of those companies to choose is still out of scope.
- The company is not in section 1.
- The whole prompt is about the economy, a market index, a peer group, or a price, and names no company in section 1.

**Pass to collect posts** when the prompt names one influencer from section 1 and one company from section 1. "Check Joseph Carlson's claim that Netflix's operating margin fell over the last two quarters" passes. "Does this look financially strong?" and "Is the finfluencer right?" pass only once the influencer and the company are named. Passing does not answer the prompt. The next step is section 3. The orchestrator has not started.

## 3. Collecting the post

Collect posts is a pipeline, not an agent. It runs only after section 2 passes.

It reads posts already stored for that influencer's channel in section 1, and keeps those that mention the company. The post must fall in 1 Apr 2026 through 30 Sep 2026, inclusive. It does not open YouTube or X, it does not call Transcribe, and it does not fetch a new video or a new X post for the question. A post that is not already stored is not fetched during the question. A URL in the prompt does not start a download. If that URL matches a stored post, it selects that post. If it does not, refuse and say the post is not in the store.

- No stored post mentions that company: refuse. Say that this influencer has no stored post about that company.
- A video is listed, but the spoken text is not stored yet: refuse with a different reason. Say that the video is listed and the transcript is not available yet. Do not treat that as "this influencer never covered the company."
- Several posts state the same claim, and the prompt names no video: show both lines, with the date, the title, and the video id. The user resubmits with the video id. A named video uses that video.
- The prompt asks whether this influencer has been consistent about this company, or asks for the accuracy of this influencer's claims about this company over time: do not ask which post. Check every stored post in the window that has text. A URL or a post id in the prompt still selects that one post.
- One post fits, or the user gave a post id: that post is the one to check.

A record has no single overall label. Each post is still one check, with that post's publication date as its cut-off.

Consistency compares the influencer's words across those posts. It looks at the same measure. The posts are not consistent when one says that measure rose and another says it fell. They are consistent when no measure flips. A post that states no direction is skipped for that measure. A filing label is not a direction. A post can be Supported and another Contradicted without that, by itself, making the wording inconsistent. The consistency report says it does not compare price, rank, or a shift from bullish to cautious.

Accuracy counts the labels. Report how many posts were Supported, Partially Supported, Contradicted, and Insufficient Evidence. Of the posts that were Supported or Contradicted, say how many were Supported. Partially Supported and Insufficient Evidence are listed and are not counted as right or wrong. The report names every post: date, post id, title, and label. Each label uses filings filed on or before that post's date. A forecast is not scored against what happened later. Contradicted means that sentence does not match the filing. It does not say the person is wrong.

The cut-off date is that post's publication date. A date the user typed does not replace it. Say the influencer, the post title, the post id, and that date in the report. For a video, the post id is the video id. For an X post, the post id is the X post id. A filing or remark published on the cut-off date is included. One published after it is left out. The period a report covers is not the cut-off date. For an SEC filing, the publication date is the date it was filed.

Only this chosen post is handed to claim extraction. The orchestrator has not started.

## 4. Extracting the claims

Claim extraction starts here. It keeps the sentences about the company named in the prompt. A portfolio video often names several companies. A sentence about another company is dropped, the same way a price line is set aside in section 7. It is not stamped with the company the user asked about. Record every dropped quote.

A comparison stays when it is about the named company, such as "bigger than Intel" or "the biggest winner." That becomes a comparative claim in section 5. Filings for the second company are not loaded. A sentence whose subject is the other company is dropped. The subject is the company the sentence is about. A name that only follows than, versus, or ahead of is the comparison, not the subject. "They" is kept only when that same sentence names one company and that company is the one in the prompt. A list of three or more companies, with no revenue, profit, margin, or cash figure, is dropped.

Who said it. Keep a sentence the host states. Drop "analysts say," a guest line, a sponsorship, and a quoted repost. The transcripts stored for this project do not mark the speaker. The report says the speaker was not separated.

What counts as a claim. Drop a question. Drop an "if" that is paired with would, could, or might. Drop a sentence the same video takes back, and drop the sentence it takes back. Do not try to detect sarcasm. A sentence the post marks as a joke is left out and listed as not checked.

A garbled number or a garbled ticker is Insufficient Evidence. It is not Contradicted. A time written in the transcript, such as [02:14], is kept with the quote. A time is not invented. The report includes the video URL. If the transcript has no time, the report says it cannot point to a moment in the video.

A figure that appears only on a chart, a slide, or an image is not in the stored text. The report says so. This project does not read the screen.

If the prompt stated a claim, match it to stored rows by meaning. The row must share the measure, and a stated figure must be in that row. "Revenue doubled" matches a kept sentence such as "revenues of 96.2 billion, up 106% year-over-year." "Grew 24%" does not match "search revenue increased by 17%." If no stored row is that close, search the transcripts of that influencer's videos about the company. One close spoken sentence is shown, and the user confirms it before it is checked. Several close sentences are shown, and the user picks one. If nothing is close, say the line is not in the stored videos and ask the user to restate it or name the video. Do not check a nearby sentence, and do not check the other claims. If the prompt stated no claim, check the kept claims about that company.

The cap of 8 applies to the claims selected above. If that set would pass 8, keep claims in this order and drop the rest: comparative, continuation, demand attribution, historical growth, net income growth, then the four children of a company-wide strong or weak phrase, then business risk. The report lists every quote the cap left out.

For example, a kept summary "Revenue growth proves it is the biggest winner from AI demand, and this is just getting started" becomes four claims: historical growth, demand attribution, comparative, continuation. A fact and an opinion about that fact stay in separate claims.

Keep the post's words. Do not make a claim stronger, weaker, or add a claim from the model's own knowledge. Each claim stores the quote it came from in the post.

This step runs when the corpus is built, once per post. It writes one record per kept sentence: post id, company, quote, type, and the expansion fields in section 8. It writes no label. A question reads that record and does not extract the post again. The record is a table of sentences. It is not the vector database. The vector database holds filing passages for the narrative agent. Until a post has this record, a question may extract it once. After the record exists, the question uses the record.

The only sentences the system may add are the expansions in section 6. Each added claim stores the original phrase and the name of the rule that expanded it.

A continuation sentence in the post, such as "growth is just getting started," is a continuation claim in section 5. It is checked against what management said. The same sentence typed by the user, with no stored post, never reaches this step.

An advice sentence inside the post, such as "you should buy this," is dropped. It does not refuse the post. The other claims are still checked. Advice in the user prompt was already refused in section 2.

A price, a macro remark, or a ranking inside the post stays in the summary. Section 7 says how that phrase is recorded. If that kind of phrase was the whole user prompt, section 2 already refused it.

Every claim uses the post date from section 3 as its cut-off. A claim about a period after that date, or about more years than the corpus holds (the latest annual report plus the last four quarters), is Insufficient Evidence. It is not looked up again. If the year-ago period is missing from the stored facts, the report says the comparison filing is not in the corpus. That is a gap in the files, and the label stays Insufficient Evidence.

## 5. Types of claim

Each small claim from the summary gets exactly one type from the table below. Advice in the user prompt was refused in section 2. An advice sentence inside the post was dropped in section 4, so it does not become a claim.

Words such as strong, weak, healthy, solid, poor, or stretched are not types. They take their meaning from what they describe. Section 6 has the tests.

- The word describes the whole company and names no measure: "financially strong," "weak fundamentals," "a healthy company," "in trouble." Section 6 splits it into profitability, cash generation, liquidity, and leverage.
- The word describes one measure: "strong revenue growth," "weak margins," "solid liquidity," "leverage is under control." It stays one claim of that measure's type. Strong, healthy, solid, and improving use that measure's favorable test. Weak, poor, falling, and strained use the opposite.

If the same type, measure, and period appears twice, keep one claim. Keep the earliest quote from the post, and record the duplicate as dropped.

| Type | What the post says | Example | Who looks it up | Best label |
|---|---|---|---|---|
| Historical growth | Past change in total revenue | "Revenue grew about 14% year over year." | Financial | Supported, from a filing only |
| Net income growth | Past change in accounting profit | "Profits grew." | Financial | Supported, from a filing only |
| Profitability | Operating margin or net margin, level and direction | "Margins are healthy and not falling." | Financial | Supported, from a filing only |
| Cash generation | Free cash flow, level and direction | "The company generates cash." | Financial | Supported, from a filing only |
| Liquidity | Current ratio, level and direction | "Liquidity is solid." | Financial | Supported, from a filing only |
| Leverage | Debt-to-equity, direction only | "Leverage is under control." | Financial | Supported, from a filing only |
| Demand attribution | The post names the cause of a result, such as AI demand or a new product. The check is whether the company itself said that. | "Growth is from AI infrastructure demand." | Narrative | Supported only when a filing gives that cause's share of the result and names no other cause. Any weaker company statement is Partially Supported. |
| Continuation | A forward-looking claim about demand or guidance | "This growth is just getting started." | Narrative | Partially Supported. The source is management's own remarks. |
| Comparative | A ranking against other companies | "Biggest winner." | Nobody. No peer filings. | Insufficient Evidence |
| Business risk | A risk named in the post | "Customer concentration is a risk." | Narrative | Supported only when a 10-K or 10-Q risk factor states it. An earnings-call remark can at most be Partially Supported. |
| Not in the table | A company fact that matches no row above | "Gross margin improved." "They bought back stock." "Attendance rose." | Narrative, in the documents stored for that company | Supported when a filing states it. An earnings remark can at most be Partially Supported. |

A sentence that fits none of the rows will be checked against the documents stored for that company, then determined whether Out of Scope or not. For example:

- A filing passage that states the claim can Support it.
- An earnings remark or investor-relations comment can at most Partially Support it.
- News may be shown. It does not move the label.
- No new calculation is invented. If the claim needs a number section 11 does not calculate, and no stored passage states that number, the label is Insufficient Evidence.
- If the filings and the company's own remarks say nothing about the claim, then it is Out of scope. The label is Insufficient Evidence.

The other sentences in the same post are still checked. Do not ask the user to rephrase only because a sentence is missing from the table.

A ranking against another company does not get this search. Peer filings are not stored, so it stays Insufficient Evidence. A price or a return that is the whole user prompt was refused in section 2. Inside a post, a price is Insufficient Evidence, because the stored filings are not used to judge a price.

## 6. Vague phrases

Use this table. Do not invent a new expansion.

| Phrase | Becomes |
|---|---|
| Company-wide, no measure named: financially strong, strong fundamentals, healthy company, good company, financially weak, weak fundamentals, poor company, in trouble | Four child claims: profitability, cash generation, liquidity, leverage. The original phrase is kept as the parent. |
| The same kind of word on one measure: strong growth, weak margins, solid cash flow, liquidity is tight, leverage is under control | One claim of that measure's type. No extra claims. |
| Turnaround is working, fundamentals are improving, credible path to a turnaround | The measures in section 11 that the post names, plus management's comments as continuation, demand attribution, or business risk. |
| Just getting started, will continue, sustainable growth | Continuation |
| Biggest winner, best, most exposed, leader | Comparative |
| From AI, because of AI demand | Demand attribution |
| Profits grew, earnings grew | Net income growth |
| Is the finfluencer right? | The claims summarized from the stored post. The answer is about those claims. |

**What strong and weak mean.** There is no peer group, so there is no test such as "margin above the industry." Compare the latest reported fiscal year with the year before, using periods on or before the cut-off. "Latest year" means that pair. It is not a calendar year and not a quarter, unless the post named a quarter.

Strong, healthy, solid, good, and improving mean the favorable test below. Weak, poor, strained, falling, tight, and in trouble mean that same test failed. A level word with no direction, such as "leverage is high" or "a lot of debt," is Insufficient Evidence. There is no cutoff for how high is too high.

| Measure | Favorable when | In plain words |
|---|---|---|
| Historical growth, net income growth | The later figure is higher than the earlier one. | "Strong" or "explosive" means it rose. The size of the rise is not tested. "Weak" means flat or down. |
| Profitability | Latest operating margin is above zero, and the change from last year is zero or positive. | The company made an operating profit, and the margin did not fall. |
| Cash generation | Latest free cash flow is above zero, and the change from last year is zero or positive. | Cash left after spending on assets, and that cash did not fall. |
| Liquidity | Latest current ratio is at least 1, and the change from last year is zero or positive. | Short-term assets cover short-term bills. A ratio of 1 means they match. This is a definition, not a quality ranking. |
| Leverage | Debt-to-equity did not rise from last year. | Only the direction is used. A positive ratio is not treated as good or bad. "Under control" means it did not rise. "Stretched" means it rose. |

When the phrase names no measure, combine the four children into one label for the parent. For a weak, poor, or in-trouble parent, read "favorable" below as "unfavorable," and the other way around.

- All four favorable: Supported.
- Two or three favorable, and none missing: Partially Supported.
- Zero or one favorable, and none missing: Contradicted.
- One missing, and the other three unfavorable: Contradicted.
- One missing, and at least two of the other three favorable: Partially Supported.
- One missing, and fewer than two favorable, and the other three are not all unfavorable: Insufficient Evidence.
- Two or more missing: Insufficient Evidence.

## 7. Phrases that are saved but not researched

- A pure opinion with nothing to check ("I love this stock") is left out. The output lists that quote and says it had nothing to check.
- A ranking inside an otherwise checkable post is saved as a comparative claim, so the report shows Insufficient Evidence. Peer filings are not loaded.
- A price inside an otherwise checkable post is saved and labeled Insufficient Evidence. Filings are not used to judge a price. If that phrase is the entire user prompt, section 2 refuses it instead.
- A company outside section 1, named in the user prompt, is refused in section 2. Named only inside the post, the sentence is dropped in section 4. It does not become a claim, and it does not refuse the rest of the post.

## 8. What is written down for each claim

| Field | Meaning |
|---|---|
| claim_id | Id for this claim inside the request |
| text | The claim, in the post's words |
| source_span | The quote in the stored post |
| parent_span | Set only when section 6 expanded a vague phrase. This is the quote the report shows. |
| expansion_rule | Set only for an expanded child, for example `financially_strong` |
| measure | The measure that child tests, such as profitability, cash generation, liquidity, or leverage. For a turnaround, the measure the post names. |
| quote_time | A time in the stored post, such as 02:14. Empty when the post has no time. |
| type | One type from section 5 |
| cut_off | The post's publication date from section 3 |
| collector | From section 5: Financial, Narrative, or none |

## 9. Where evidence can come from

Every stored passage has a trust level from the day it was loaded. A passage with no level cannot be quoted. The stored influencer post has no level.

| Level | Source | What it may do |
|---|---|---|
| 1 | SEC filings and their tagged numbers: 10-K, 10-Q, the filed management discussion, and the risk factors | Support a past fact when the calculated figure matches the value, the direction, and the period. |
| 2 | The company's earnings release or investor-relations remarks | Partially Support a forward-looking claim. "Expects" stays "expects." The same remark can Partially Support a business-risk claim, or a claim that is not in the type table, when the remark states it. It cannot support a past number that section 11 calculates. |
| 3 | Yahoo Finance company news, up to 10 articles, loaded with the corpus | Context only. It may be shown. It does not move a label up or down. |
| None | The stored influencer post | The claim under test. A number that appears only here is not evidence. |

When a filing and a press release or a news article disagree, keep the filing's number and direction.

A quote must be an exact span of a passage stored at load time. A quote that cannot be found there is blocked.

If the earnings release is in the corpus and the 10-Q is not yet, report the press-release figure as what the company said. The best label is Partially Supported until the filing confirms it.

A risk-factor passage retrieved while checking some other claim counts as counter-evidence for that claim. It can keep that claim from reaching Supported. It does not replace a Tier 1 number.

## 10. Verification

Two checks happen before the user sees an answer. Both run after the agents have gathered evidence for the summarized claims.

**Auditor.** For each claim, check three things. The number matches section 11, when the claim uses one of those measures. The quote is an exact span of a stored passage. The source level is one section 9 allows for that type. Section 12 says when the label may be Supported. If this check fails, section 13 is the one retry.

**Last screen.** The Advice Boundary Agent reads the drafted report. If the draft tells the user to buy, sell, hold, or allocate money, or if it names a price target or guarantees a return, rewrite the draft and delete those sentences. The claim results stay. This screen does not refuse the request. A refusal happens earlier: the input screen in section 2, or collect posts in section 3 when no usable post is stored.

The report names the influencer, the post title, the post id, and the post date used as the cut-off. If the prompt stated a claim and no stored row is close enough, the reply quotes the closest spoken sentence and asks the user to confirm it, or says the line is not in the stored videos. An evidence answer also includes one sentence: this checks the evidence for the stated claim and is not a personal recommendation. A record report names every post instead of one, and it has no single overall label. It includes the consistency sentence, the accuracy count, or both, as section 3 describes.

When section 6 expands a phrase, the report shows that expansion. The log is not a substitute. The overall label uses the parent, so the children are not also counted as separate claims.

- Company-wide, no measure named. One block for the parent. It quotes `parent_span` and names `expansion_rule`. When `quote_time` is stored, the block includes that time. It then gives each child: the measure, the plain-words test from section 6, the filing figure already calculated, the period dates, and the child's label. Then the section 6 rollup, in one sentence, such as "3 of 4 met the test, so the phrase is Partially Supported." Then one limits sentence: the size of the figures was not compared with other companies, and leverage was tested on direction only.
- One measure, such as "strong growth." The report is that one claim. It says the size of the change was not tested, as section 11 already requires. It does not invent the four tests.
- "Turnaround is working." The block lists the measures the post names, each with its figure and label. It does not add a measure the post did not name.

The plain-words test is the result the user reads. "Favorable" is not the wording of the report. The advice screen reads this block with the rest of the draft.

## 11. The numbers we calculate

Only these calculations may support a financial claim. The numbers come from the SEC's tagged filing data for that company. Python calculates the ratio. The model does not. Use the filing whose filed date is on or before the cut-off. If a number was restated, use the latest version filed on or before the cut-off.

| Measure | Inputs | Formula |
|---|---|---|
| Revenue growth | Total revenue | Later revenue ÷ earlier revenue − 1 |
| Net income growth | Accounting profit | Later profit ÷ earlier profit − 1 |
| Operating margin | Operating income, revenue | Operating income ÷ revenue |
| Net margin | Profit, revenue | Profit ÷ revenue |
| Free cash flow | Cash from operations, capital spending | Cash from operations minus capital spending. Treat capital spending as money going out. If the tagged number is already negative, do not subtract it a second time. |
| Current ratio | Short-term assets, short-term bills | Short-term assets ÷ short-term bills. A result of 1 means the assets cover the bills. |
| Debt-to-equity | Short-term debt, long-term debt, shareholders' equity | (Short-term debt + long-term debt) ÷ equity. For the "financially strong" test, only the direction from last year is used. |

Use total revenue and total profit, not a segment. A segment growth rate presented as if it were the whole company is Insufficient Evidence. A segment figure stated as that segment, such as "AWS grew 17%," is checked as that segment. It is not sent through the total-revenue calculation. The report says it is about that segment, not the whole company. The lookup is a stored passage that states the figure. If no passage states it, the label is Insufficient Evidence.

**Which period.**

- No period named: latest fiscal year against the year before.
- "Recent" or "quarter": latest quarter against the same quarter a year earlier.
- "Last two quarters": the two latest quarters that end on or before the cut-off, each against the same quarter a year earlier, unless the post said quarter over quarter.
- Cash flow in a quarterly report is year-to-date. A single quarter's free cash flow is this year-to-date figure minus the prior quarter's year-to-date figure in the same fiscal year. The first quarter's year-to-date figure is the quarter.
- The fourth quarter has no report of its own. It is the full year minus the year-to-date figure through the third quarter. If either figure is missing, the fourth quarter is unavailable.

**Awkward inputs.**

- Equity is zero or negative: debt-to-equity is Insufficient Evidence.
- One or both debt tags are missing: debt-to-equity is Insufficient Evidence. A missing tag is not zero.
- Both debt tags are present and sum to zero: total debt is zero.
- The earlier revenue or profit is zero or negative: a percentage change is undefined, so the percentage claim is Insufficient Evidence. If both amounts exist, the direction can still be stated.

**When a stated number matches.**

- A percentage matches when it is within 0.5 percentage points of the calculated figure, and the direction matches. "About 14%" matches 14.3%. A decline does not match.
- "Revenue did not grow" is Supported when the calculated direction is flat or down.
- "Doubled" matches when the later amount is from 1.9 times to 2.1 times the earlier amount.
- "Double-digit" matches when the size of the change is at least 10%.
- On a growth claim, "strong" and "explosive" mean the direction only, as in section 6. The output says the size was not tested.
- A dollar amount matches after rounding to the precision the post stated. "$2.1 billion" matches a calculated amount from $2.05 billion to $2.15 billion.

**Not calculated here.** A missing tag or a missing comparison period makes that calculated claim Insufficient Evidence. The gap is not filled from news, from the stored post, or from memory.

Gross margin, adjusted EBITDA, earnings per share, buybacks, dividends, a cash balance, homes sold, and attendance are not in the table above, so they are not calculated. They are still checked under section 5, by searching the stored documents. The label is Insufficient Evidence only when those documents do not state the claim. A segment figure presented as the whole company stays Insufficient Evidence.

## 12. The four labels

| Label | Meaning |
|---|---|
| Supported | The evidence allowed for that type states the claim, including the value and the direction when the claim has them, and no contrary passage was ignored. |
| Partially Supported | The evidence agrees in part, the claim is forward-looking, more than one cause is credited, or the size of the change was not tested. The report names which of those four reasons applies: partial match, forward-looking, more than one cause, or size not tested. |
| Contradicted | A financial claim disagrees with the filing in value or direction. A narrative claim is Contradicted only when a filing says the opposite. A missing passage is not Contradicted. The label is about that sentence and that filing. The report does not say the person is wrong. |
| Insufficient Evidence | The stored documents cannot decide. This includes a claim they do not mention, a period outside the window, a missing tag, and a ranking against peers. |

| Type | Supported when | Otherwise |
|---|---|---|
| Historical growth, net income growth, profitability, cash, liquidity, leverage | The filing matches the value, the direction, and the period in section 11 | Missing period or missing tag: Insufficient Evidence |
| Demand attribution | A filing gives that driver's share of growth and names no other driver | Any weaker company statement: Partially Supported. A number with no driver: Insufficient Evidence |
| Continuation | Never | Management says "expects" or gives guidance: Partially Supported. "Will" or "proves" overstates that. |
| Comparative | Never | Insufficient Evidence. Peer filings are not stored, so there is no search. |
| Not in the table | A stored filing states the claim | An earnings remark that states it: Partially Supported. Nothing in the stored filings or remarks: Insufficient Evidence. That is the out-of-scope judgment. |
| Business risk | The 10-K or 10-Q risk factors state that risk | An earnings remark that states it: Partially Supported. Otherwise the passage can only count as counter-evidence on another claim. |

For a narrative claim, both supporting and contrary passages are collected. A contrary passage from a filing or from the company's own remarks keeps the claim from reaching Supported. A contrary remark does not overturn a filing's number. News counts as neither support nor challenge.

Words such as "proves," "certainly," or "guaranteed" cannot sit on a Partially Supported fact. "Just getting started" is a continuation claim. It does not upgrade a past fact.

## 13. Retry (if the auditor's check failed)

If the auditor's check in section 10 fails, retry one time. The second attempt does not call the other agent. A wrong number is rewritten from the figure already calculated. The narrative agent is called again only when a passage is missing.

Skip the second look when another pass cannot help: a comparative claim, a price inside a post, a missing filing tag, a period after the cut-off, a history longer than the corpus window, or a ratio section 11 cannot define. A claim that is not in the type table still gets its search. It is out of scope only after that search finds nothing.

| What went wrong | What the second look does |
|---|---|
| The stated number or direction disagrees with the filing | Rewrite the sentence to the calculated figure. Leave the calculated figure as it is. |
| The quoted passage does not support the sentence | Rewrite the sentence to match the passage, or drop the quote. |
| There is no quote, or the quote is not an exact span of a stored passage | Attach an exact span, or drop the sentence. |
| A press release or a news article was used to support a past fact | Replace the quote with the filing, or lower the label. |
| The label is higher than that type allows | Lower the label to the cap in section 5. |
| A contrary passage was ignored | State that passage, and set the label no higher than Partially Supported. |
| The wording is stronger than the evidence | Rewrite to the weaker wording. |
| A narrative passage is missing | The narrative agent searches once more, aimed at the gap. |

Across the whole request, stop after 8 second looks or 4 extra searches, and stop if the token budget for the request is used up. Unfinished claims are Insufficient Evidence. The overall label in section 14 uses the claims that finished. The report says the run stopped because a cap was hit.

If the second look still fails, that claim is Insufficient Evidence and still counts in the overall label.

## 14. The overall label

Combine the claims after the parent rollup in section 6. The report shows that parent once, with the tests underneath, as section 10 says. A refused request has no overall label. The report lists any contradicted claims.

| Situation | Overall |
|---|---|
| Every claim Supported | Supported |
| At least one Supported or Partially Supported, and at least one Partially Supported or Insufficient Evidence, and none Contradicted | Partially Supported |
| At least one Contradicted, and at least one Supported or Partially Supported | Partially Supported |
| At least one Contradicted, and every other claim Insufficient Evidence | Contradicted |
| Every claim Insufficient Evidence | Insufficient Evidence |
| Nothing left to check | No overall label. The report says nothing in the post could be checked. |

One Supported claim next to contradicted claims is still Partially Supported overall.

The report also says three limits. The label is a document check of the sentence against the filing, and it can be wrong. The project window is 1 Apr 2026 through 30 Sep 2026. Extending that window is a change to this rule book, with filings through the new date. The check is in English. A transcript in another language is outside this project, and a question does not translate it.

A record names the post it is checking. Each post keeps the one retry already allowed. That retry repeats the same check and does not start a second agent. If a post cannot be finished, the record says it is incomplete.

## 15. The log

Every step adds a record: the user prompt, whether it stated a claim, the influencer, the company, the post that was collected (post id, title, and publication date), the cut-off (that post date, even if the prompt named a different date), the screening decision and why, the collect-posts decision and why, quotes dropped because they were about another company, each claim with its type, quote, label, and source level, any second look, other dropped quotes, the overall label, and the tokens and time used.
