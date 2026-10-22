# Decision rationale

Owner: Irine (Person 4), for Hung (Person 5).
Date: 5 Oct 2026.

These rows replace `docs/to_resolve.md`. The choice below is the one to implement. `docs/knowledge_layer.md` and `finsight/knowledge/knowledge_layer.yaml` already follow it. The orchestrator code in `docs/decisions.md` does not, until Hung updates it.

| ID | Choice | Why |
|---|---|---|
| D1 | Knowledge layer. One Supported or Partially Supported claim alongside a Contradicted one stays Partially Supported. One Supported and two Contradicted is Partially Supported. The report lists the contradicted claims. | The reader should see a mixed thesis as mixed. The code's "more than half contradicted" rule hides a supported fact behind a Contradicted headline. |
| D22 | Knowledge layer. `retry_collectors` is empty. A wrong number is rewritten from the computed record. The narrative agent is called again only when a passage is missing. Continuation is not retried with numbers. | In code, a retry is a second evidence collection. Calling the narrative agent cannot repair a revenue figure. |
| D21 | Knowledge layer. Skip the retry for a missing XBRL tag, a comparative claim, an out-of-scope claim, or a period outside the corpus window. Retry only when the missing evidence might be in the corpus. | A second pass cannot create a missing tag or a peer filing, and it spends a model call. |
| D8 | Knowledge layer. If the draft says buy, sell, hold, allocate, names a price target, or guarantees a return, rewrite the draft and delete those sentences. Release the claim results. Do not refuse the request. | The user asked for an evidence check. Refusing the whole report hides that check because of one sentence. |
| D9 | Knowledge layer. If the user gives no cut-off, use the publication date of that company's newest document. | One shared corpus date can include another company's later filing, or drop this company's newest 10-Q. |
| D6 | Knowledge layer. Refuse when the user asks FinSight for a price or a return. A price mentioned inside a pasted thesis is an Out of scope sub-claim, and the revenue claim is still checked. | Realistic finfluencer posts mention both. Refusing the whole request drops the checkable claim. |
| D16 | Knowledge layer. A business-risk claim is Supported only when a 10-K or 10-Q risk factor states it. A Tier 2 remark can at most Partially Support it. The same ceiling applies to a claim that is not in the type table. Tier 1 or Tier 2 counter-evidence still blocks Supported. | An earnings-call sentence should not be treated as a filed risk factor. The orchestrator lowers a Supported label to Partially Supported when the only support is Tier 2. |
