const state = {
  meta: null, view: null, log: null, stepQuery: "", lastStageCount: -1,
  company: "", influencer: "", post: "",
};

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function quote(text) {
  return `'${String(text).replaceAll("'", "'\"'\"'")}'`;
}

function verdictClass(verdict) {
  return verdict ? `v-${verdict}` : "";
}

function headline(view) {
  const assessment = view.assessment;
  if (assessment.overall_verdict) {
    return { text: view.overall_label || assessment.verdict_label, cls: verdictClass(assessment.overall_verdict) };
  }
  const labels = {
    REFUSED: "Refused",
    NEEDS_CLARIFICATION: "Needs an answer",
    FAILED: "Failed",
    COMPLETED: "Completed",
  };
  return { text: labels[assessment.status] || assessment.status, cls: `s-${assessment.status}` };
}

function stepTone(lines) {
  const text = lines.join(" ").toUpperCase();
  if (text.includes("STOP") || text.includes("GAVE UP") || text.includes("FLAGGED")) return "is-stop";
  if (/\bFAIL/.test(text)) return "is-fail";
  if (text.includes("RETRY")) return "is-retry";
  if (text.includes("CONTINUE") || text.includes("RELEASE") || text.includes("ACCEPTED")) return "is-ok";
  return "";
}

function lineClass(line) {
  const text = line.toUpperCase();
  if (text.includes("STOP") || text.includes("GAVE UP")) return "line-stop";
  if (/\bFAIL/.test(text)) return "line-fail";
  if (text.includes("RETRY")) return "line-retry";
  return "";
}

function cliPreview() {
  const parts = ["finsight run", quote($("claim").value), "--stages"];
  if ($("cutoff").value) parts.push("--cutoff", $("cutoff").value);
  if (state.company) parts.push("--company", state.company);
  if (state.influencer) parts.push("--influencer", quote(state.influencer));
  if (state.post) parts.push("--post", state.post);
  $("cliPreview").textContent = parts.join(" ");
}

async function init() {
  const response = await fetch("/api/meta");
  state.meta = await response.json();
  renderMode();
  const asOf = state.meta.mode.corpus_as_of;
  if (asOf) {
    const [year, month, day] = asOf.split("-");
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const label = `${Number(day)} ${months[Number(month) - 1]} ${year}`;
    $("cutoffHint").textContent = `Enter a cut-off to choose which posts are eligible. Leave it blank and a stored post is checked as of the day it was published. If that post has no date, the check uses the latest filing in the corpus (${label}).`;
  }
  cliPreview();
  $("cutoff").addEventListener("input", cliPreview);
  $("claim").addEventListener("input", () => {
    state.company = "";
    state.influencer = "";
    state.post = "";
    cliPreview();
  });
  $("runBtn").addEventListener("click", run);
  $("clearBtn").addEventListener("click", clearForm);
  $("claim").addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      run();
    }
  });
  $("technical").addEventListener("change", () => {
    document.body.classList.toggle("is-technical", $("technical").checked);
    localStorage.setItem("finsight-technical", $("technical").checked ? "1" : "0");
  });
  if (localStorage.getItem("finsight-technical") === "1") {
    $("technical").checked = true;
    document.body.classList.add("is-technical");
  }
  $("preview").addEventListener("click", onPreviewClick);
}

function renderMode() {
  const mode = state.meta.mode;
  const chip = $("modeChip");
  const spend = state.meta.spend;
  const override = mode.env_override ? " · shell override" : "";
  chip.classList.toggle("is-paid", mode.paid);
  chip.textContent = mode.paid
    ? `PAID ${mode.provider} · ${mode.model}${override}`
    : `MOCK · free · corpus ${mode.corpus_as_of}${override}`;
  chip.title = `Retries ${mode.max_retries}. Ledger $${spend.cost_usd.toFixed(4)} of $${spend.budget_usd.toFixed(0)} on this machine.`;
  $("paidRow").hidden = !mode.paid;
}

function clearForm() {
  $("claim").value = "";
  $("cutoff").value = "";
  state.company = "";
  state.influencer = "";
  state.post = "";
  cliPreview();
  $("formStatus").textContent = "Cleared.";
}

async function run() {
  const body = {
    claim: $("claim").value,
    cutoff: $("cutoff").value,
    company: state.company,
    influencer: state.influencer,
    post: state.post,
    confirm_paid: $("confirmPaid").checked,
  };
  if (state.meta.mode.paid && !body.confirm_paid) {
    $("formStatus").textContent = "Tick the confirmation first. A paid run can spend course credit.";
    return;
  }
  $("runBtn").disabled = true;
  $("empty").hidden = true;
  $("result").innerHTML = "";
  $("trace").innerHTML = "";
  state.lastStageCount = -1;
  state.view = null;
  $("formStatus").textContent = "Running… steps will appear as each stage finishes.";
  renderStages([], 0, true);
  let response;
  try {
    response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    $("formStatus").textContent = "The page could not reach the local server.";
    $("runBtn").disabled = false;
    return;
  }
  const data = await response.json();
  if (!response.ok) {
    $("formStatus").textContent = data.error || "The run was rejected.";
    $("runBtn").disabled = false;
    return;
  }
  poll(data.job_id);
}

async function poll(jobId) {
  let data;
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    data = await response.json();
    if (!response.ok) throw new Error(data.error || "Lost the run.");
  } catch (error) {
    $("formStatus").textContent = error.message;
    $("runBtn").disabled = false;
    return;
  }
  renderStages(data.stages || [], data.elapsed_ms, data.status === "running");
  if (data.status === "running") {
    $("formStatus").textContent = `Running… ${data.stages.length} stage${data.stages.length === 1 ? "" : "s"}, ${Math.round(data.elapsed_ms)} ms.`;
    setTimeout(() => poll(jobId), 400);
    return;
  }
  $("runBtn").disabled = false;
  if (data.status === "error") {
    $("formStatus").textContent = data.error || "The run failed.";
    return;
  }
  state.view = data.result;
  state.log = data.result.decision_log;
  renderResult(data.result);
  renderTrace(data.result.decision_log, data.result.stats, data.result.trace_text, data.result.warning);
  const assessment = data.result.assessment;
  $("formStatus").textContent = `${assessment.status} · ${assessment.request_id} · ${Math.round(data.elapsed_ms)} ms.`;
}

function renderStages(stages, elapsed, running) {
  if (stages.length === state.lastStageCount && $("timeline")) {
    const note = $("stepNote");
    if (note) note.textContent = stageNote(stages, elapsed, running);
    return;
  }
  state.lastStageCount = stages.length;
  const visible = stages.filter((stage) => stageMatches(stage, state.stepQuery));
  $("steps").innerHTML = `
    <section class="card" id="stepCard">
      <div class="section-head">
        <h2 id="steps-title">Steps</h2>
        <span class="ops__note" id="stepNote">${esc(stageNote(stages, elapsed, running))}</span>
        <input id="stepFilter" placeholder="Filter steps" value="${esc(state.stepQuery)}" />
      </div>
      <ol class="timeline" id="timeline">
        ${visible.map(stageHtml).join("") || `<li class="ops__note">${running ? "Waiting for the first stage." : "No stages match."}</li>`}
      </ol>
    </section>`;
  $("stepFilter").addEventListener("input", (event) => {
    state.stepQuery = event.target.value;
    state.lastStageCount = -1;
    renderStages(stages, elapsed, running);
    $("stepFilter").focus();
  });
}

function stageNote(stages, elapsed, running) {
  const time = elapsed ? `${Math.round(elapsed)} ms` : "";
  const tail = running ? "running" : "finished";
  return `${stages.length} stage${stages.length === 1 ? "" : "s"} · ${tail}${time ? ` · ${time}` : ""}`;
}

function stageMatches(stage, query) {
  if (!query) return true;
  const haystack = `${stage.node} ${stage.role} ${stage.lines.join(" ")}`.toLowerCase();
  return haystack.includes(query.toLowerCase());
}

function stageHtml(stage) {
  const lines = stage.lines.map((line) => `<p class="${lineClass(line)}">${esc(line)}</p>`).join("");
  const timing = stage.ms == null ? "" : `<span class="step__ms">${stage.ms} ms</span>`;
  return `
    <li class="step ${stepTone(stage.lines)}">
      <span class="step__n">${esc(stage.step)}</span>
      <div>
        <div class="step__head">
          <span class="step__node">${esc(stage.node)}</span>
          <span class="step__role">${esc(stage.role)}</span>
          ${timing}
        </div>
        ${lines}
      </div>
    </li>`;
}

function renderResult(view) {
  const assessment = view.assessment;
  const head = headline(view);
  const gates = (view.decision_log.gate_decisions || []).map((gate) => `
    <span class="tag ${gate.flagged ? "against" : "for"}">${esc(gate.gate)} ${gate.flagged ? "flagged" : "pass"}</span>
  `).join("");
  const violations = (view.decision_log.temporal_violations || []).map((item) => `
    <p class="violation">Cut-off leak: ${esc(item.evidence_id)} for ${esc(item.sub_claim_id)} was published ${esc(item.publication_date)}, after ${esc(item.cutoff_date)}. ${esc(item.produced_by)} produced it and it was dropped.</p>
  `).join("");
  $("result").innerHTML = `
    <article class="card">
      <div class="result-top">
        <span class="verdict ${head.cls}">${esc(head.text)}</span>
        <span class="status">${esc(assessment.status)}</span>
        <span class="rid">${esc(assessment.request_id)}</span>
        <button type="button" class="text-btn" data-copy="${esc(assessment.request_id)}">Copy id</button>
        <button type="button" class="text-btn" data-download="assessment">Assessment JSON</button>
        <button type="button" class="text-btn" data-download="log">Decision log JSON</button>
      </div>
      <div class="jump-row">
        <a href="#stepCard">Steps</a>
        <a href="#checks">Checks and citations</a>
        <a href="#traceCard">Trace</a>
      </div>
      ${assessment.checked_post ? `<p class="post-line">${esc(assessment.checked_post)}</p>` : ""}
      <p class="summary">${esc(assessment.summary || assessment.refusal_reason || assessment.clarification_question || "No summary.")}</p>
      ${assessment.as_of_date ? `<p class="reason">Information up to ${esc(assessment.as_of_date)}${view.cutoff_note ? ` — ${esc(view.cutoff_note)}` : ""}.</p>` : ""}
      ${assessment.refusal_reason && !(assessment.summary || "").includes(assessment.refusal_reason) ? `<p class="reason">${esc(assessment.refusal_reason)}</p>` : ""}
      <div class="gate-row">${gates}</div>
      <div class="stat-row">${statHtml(view.stats)}</div>
      ${violations}
      <p class="disclaimer">${esc(assessment.boundary_disclaimer)}</p>
      ${assessment.status === "COMPLETED" ? `<p class="disclaimer">${esc(view.limits || "")}</p>` : ""}
      ${view.cli ? `<p class="cli-label">Repeat this run</p><pre class="cli">${esc(view.cli)}</pre>` : ""}
    </article>
    ${clarificationHtml(view)}
    ${recordHtml(view.record)}
    <section class="card" id="checks">
      <h2>Checks and citations</h2>
      <p class="card__hint">Each sub-claim keeps the passage or filing figure behind it. A missing citation is shown on purpose, so a gap is visible while testing.</p>
      ${checksHtml(view)}
    </section>`;
  $("empty").hidden = true;
}

function statHtml(stats) {
  if (!stats) return "";
  const bits = [
    ["calls", stats.agent_calls],
    ["tools", stats.tool_calls],
    ["retries", stats.retries],
    ["tokens", stats.tokens],
    ["cost", `$${Number(stats.cost_usd || 0).toFixed(4)}`],
    ["latency", `${Math.round(stats.latency_ms || 0)} ms`],
    ["cut-off leaks", stats.temporal_violations],
  ];
  return bits.map(([label, value]) => `<span class="stat">${esc(label)} ${esc(value)}</span>`).join("");
}

function clarificationHtml(view) {
  const question = view.assessment.clarification_question;
  if (view.assessment.status !== "NEEDS_CLARIFICATION" || !question) return "";
  const choices = (view.post_choices || []).map((choice) => `
    <button type="button" class="choice" data-fill-post="${esc(choice.video_id)}">
      ${esc(choice.video_id)}
      <small>${esc(choice.date)} ${esc(choice.title)}</small>
    </button>`).join("");
  const companies = view.answer_slot === "company"
    ? state.meta.companies.map((row) => `
      <button type="button" class="choice" data-fill-company="${esc(row.ticker)}">${esc(row.name)} <small>${esc(row.ticker)}</small></button>`).join("")
    : "";
  const influencers = view.answer_slot === "influencer"
    ? state.meta.influencers.map((name) => `
      <button type="button" class="choice" data-fill-influencer="${esc(name)}">${esc(name)}</button>`).join("")
    : "";
  const slotLabel = { post: "Video id", company: "Ticker", influencer: "Influencer", claim: "Revised claim" }[view.answer_slot] || "Answer";
  return `
    <section class="card clarify">
      <h2>FinSight needs an answer</h2>
      <p class="summary">${esc(question)}</p>
      <div class="choices">${choices}${companies}${influencers}</div>
      <label class="field"><span>${esc(slotLabel)}</span><input id="answer" /></label>
      <button type="button" class="run-btn" id="sendAnswer">Send answer and run again</button>
    </section>`;
}

function recordHtml(record) {
  if (!record || !record.posts || !record.posts.length) return "";
  const unfinished = record.unfinished && record.unfinished.length
    ? `<p class="line-stop">Unfinished posts: ${esc(record.unfinished.join(", "))}</p>` : "";
  const posts = record.posts.map((post, index) => `
    <details ${index === 0 ? "open" : ""}>
      <summary><strong>${esc(post.label)}</strong> · ${esc(post.upload_date || "undated")} · ${esc(post.video_id)} · ${esc(post.title || "")}</summary>
      <p class="reason">Cut-off used for this post: ${esc(post.cutoff_date || "unknown")}.</p>
      ${post.view ? checksHtml(post.view) : `<p class="ops__note">This post has no decision log.</p>`}
    </details>`).join("");
  return `
    <section class="card">
      <h2>Posts in this record</h2>
      <p>${esc(record.consistency || "")}</p>
      <p>${esc(record.accuracy || "")}</p>
      ${unfinished}
      ${posts}
    </section>`;
}

function checksHtml(view) {
  const rows = Object.fromEntries((view.assessment.sub_claim_results || []).map((row) => [row.sub_claim.sub_claim_id, row]));
  const units = view.units && view.units.length
    ? view.units
    : (view.assessment.sub_claim_results || []).map((row) => ({
      heading: row.sub_claim.text,
      verdict: row.verdict,
      label: row.verdict_label,
      partial_reason: null,
      is_parent: false,
      ids: [row.sub_claim.sub_claim_id],
    }));
  if (!units.length) {
    return `<p class="ops__note">No sub-claim was checked. The status above is the whole result.</p>`;
  }
  const gaps = (view.assessment.evidence_gaps || []).map((gap) => `<p class="reason">Gap: ${esc(gap)}</p>`).join("");
  return gaps + units.map((unit) => {
    const children = unit.ids.map((id) => rows[id]).filter(Boolean);
    const reason = unit.partial_reason ? ` · ${unit.partial_reason}` : "";
    return `
      <article class="check">
        <div class="check__head">
          <p class="check__text">${esc(unit.heading)}</p>
          <span class="verdict ${verdictClass(unit.verdict)}">${esc(unit.label)}${esc(reason)}</span>
        </div>
        ${children.map(checkBody).join("")}
      </article>`;
  }).join("");
}

function checkBody(row) {
  const claim = row.sub_claim;
  const audit = row.audit || {};
  const tier = row.highest_tier_used || "—";
  return `
    <div class="tech">
      <p>${esc(claim.sub_claim_id)} · ${esc(claim.sub_claim_type)}${claim.measure ? ` · ${esc(claim.measure)}` : ""} · tier ${esc(tier)} · retries ${esc(row.retries_used)}</p>
      ${claim.source_span ? `<p>Quote: ${esc(claim.source_span)}</p>` : ""}
      ${row.rationale ? `<p>Audit: ${esc(row.rationale)}</p>` : ""}
      ${audit.review_note ? `<p>Review: ${esc(audit.review_note)}</p>` : ""}
      ${row.capped_by ? `<p>Capped: ${esc(row.capped_by)}</p>` : ""}
      ${audit.numeric_label ? `<p>Number: ${esc(audit.numeric_label)}</p>` : ""}
      ${stanceLines(audit)}
      ${(audit.failure_reasons || []).length ? `<p>Failures: ${esc(audit.failure_reasons.join("; "))}</p>` : ""}
    </div>
    ${row.explanation ? `<p class="why">${esc(row.explanation)}</p>` : ""}
    ${evidenceGroup(row.supporting, "Evidence for", "for")}
    ${evidenceGroup(row.challenging, "Evidence against", "against")}
    ${!row.supporting.length && !row.challenging.length ? `<p class="where missing">No evidence was attached to ${esc(claim.sub_claim_id)}.</p>` : ""}`;
}

function stanceLines(audit) {
  const stances = audit.evidence_stances || {};
  const reasons = audit.failure_reasons || [];
  return Object.keys(stances).map((id) => {
    const why = reasons.find((reason) => reason.startsWith(`${id}: `));
    const note = why ? why.slice(id.length + 2) : "";
    return `<p>Stance: ${esc(stances[id])}${note ? ` — ${esc(note)}` : ""}</p><p class="where">${esc(id)}</p>`;
  }).join("");
}

function evidenceGroup(items, title, tone) {
  if (!items || !items.length) return "";
  return `<h3><span class="tag ${tone}">${esc(title)}</span></h3>${items.map(evidenceCard).join("")}`;
}

function evidenceCard(item) {
  const link = typeof item.url === "string" && /^https?:\/\//.test(item.url)
    ? `<p class="link"><span>Link</span> <a href="${esc(item.url)}" target="_blank" rel="noopener">${esc(item.url)}</a></p>`
    : "";
  const where = item.citation_span
    ? `<p class="where"><span>Citation</span> ${esc(item.citation_span)}</p>`
    : `<p class="where missing"><span>Citation</span> none recorded</p>`;
  const bits = [];
  if (item.period) bits.push(`period ${item.period}`);
  if (item.period_source) bits.push(`from ${item.period_source}`);
  if (item.computed_label) bits.push(`computed ${item.computed_label}`);
  if (item.comparison_label) bits.push(`compared with ${item.comparison_label}`);
  if (item.publication_date_is_estimated) bits.push("publication date estimated");
  const meta = bits.length ? `<p class="meta"><span>Detail</span> ${esc(bits.join(" · "))}</p>` : "";
  return `
    <article class="evidence">
      <blockquote>${esc(item.content)}</blockquote>
      <p class="cite">${esc(item.source_label || item.doc_type || "Source not labelled")}</p>
      ${where}
      ${link}
      ${meta}
      <div class="tech">
        <p>${esc(item.evidence_id)} · ${esc(item.stance)} · ${esc(item.produced_by)} · ${esc(item.metric_name || "")}</p>
      </div>
    </article>`;
}

function renderTrace(log, stats, traceText, warning) {
  if (!log) {
    $("trace").innerHTML = "";
    return;
  }
  const calls = log.agent_calls || [];
  const maxMs = Math.max(1, ...calls.map((call) => call.latency_ms || 0));
  const callRows = calls.map((call) => `
    <tr>
      <td>${esc((call.started_at || "").slice(11, 23))}</td>
      <td>${esc(call.agent)}</td>
      <td>${esc(call.sub_claim_id || "—")}</td>
      <td><div class="bar-track"><div class="bar" style="width:${Math.max(4, (call.latency_ms / maxMs) * 90)}px"></div></div> ${Math.round(call.latency_ms)} ms</td>
      <td>${esc((call.input_tokens || 0) + (call.output_tokens || 0))}</td>
      <td>${esc(call.tool_calls || 0)}</td>
      <td class="${call.success ? "" : "call-fail"}">${call.success ? "ok" : esc(call.error || "failed")}</td>
    </tr>`).join("");
  const routes = (log.routing_plan || []).map((row) => `
    <tr>
      <td>${esc(row.sub_claim_id)}</td>
      <td>${esc(row.sub_claim_type)}</td>
      <td>${esc((row.collectors || []).join("+") || "—")}</td>
      <td>${esc(row.verdict_cap || "—")}</td>
      <td>${esc((row.tier_requirement || []).join(", ") || "—")}</td>
      <td>${esc(row.note || "")}</td>
    </tr>`).join("");
  const audits = (log.audit_results || []).map((row) => `
    <tr>
      <td>${esc(row.sub_claim_id)}</td>
      <td>${row.passed ? "pass" : "fail"}</td>
      <td>${esc(row.verdict)}</td>
      <td>${esc((row.failure_reasons || []).join("; "))}</td>
      <td>${esc(row.feedback_for_retry || "")}</td>
    </tr>`).join("");
  const retries = (log.retries || []).map((row) => `
    <tr>
      <td>${esc(row.sub_claim_id)}</td>
      <td>${esc(row.attempt)}</td>
      <td>${row.exhausted ? "gave up" : "retry"}</td>
      <td>${esc(row.feedback || "")}</td>
    </tr>`).join("");
  const finals = Object.entries(log.final_verdicts || {}).map(([id, verdict]) => `
    <span class="tag">${esc(id)} ${esc(verdict)}</span>`).join("");
  $("trace").innerHTML = `
    <section class="card" id="traceCard">
      <div class="section-head">
        <h2>Trace</h2>
        <span class="rid">${esc(log.request_id || "")}</span>
        <button type="button" class="text-btn" data-copy-trace="1">Copy plain trace</button>
      </div>
      ${warning ? `<p class="violation">${esc(warning)}</p>` : ""}
      <p class="card__hint">This is the decision log: gates, the routing plan, every agent call, audits, retries, and events. ${esc(log.input || "")}</p>
      <div class="stat-row">${statHtml(stats)}</div>
      <h3>Routing plan</h3>
      ${routes ? `<table><thead><tr><th>Id</th><th>Type</th><th>Collectors</th><th>Cap</th><th>Tiers</th><th>Note</th></tr></thead><tbody>${routes}</tbody></table>` : `<p class="ops__note">No routing plan. The run stopped before it.</p>`}
      <h3>Agent calls</h3>
      ${callRows ? `<table><thead><tr><th>Time</th><th>Agent</th><th>Sub-claim</th><th>Latency</th><th>Tokens</th><th>Tools</th><th>Result</th></tr></thead><tbody>${callRows}</tbody></table>` : `<p class="ops__note">No agent calls.</p>`}
      <h3>Audits</h3>
      ${audits ? `<table><thead><tr><th>Id</th><th></th><th>Verdict</th><th>Why it failed</th><th>Retry feedback</th></tr></thead><tbody>${audits}</tbody></table>` : `<p class="ops__note">No audit.</p>`}
      <h3>Retries</h3>
      ${retries ? `<table><thead><tr><th>Id</th><th>Attempt</th><th></th><th>Feedback</th></tr></thead><tbody>${retries}</tbody></table>` : `<p class="ops__note">No retries.</p>`}
      <h3>Final verdicts</h3>
      <div class="gate-row">${finals || `<span class="ops__note">None recorded.</span>`}</div>
      <h3>Events</h3>
      <ol class="timeline">
        ${(log.events || []).map((event, index) => `
          <li class="step">
            <span class="step__n">${index + 1}</span>
            <div>
              <div class="step__head"><span class="step__node">${esc(event.node)}</span><span class="step__ms">${esc((event.at || "").replace("T", " ").slice(0, 19))}</span></div>
              <p>${esc(event.message)}</p>
            </div>
          </li>`).join("") || `<li class="ops__note">No events.</li>`}
      </ol>
      <details class="payload-wrap">
        <summary>Plain trace</summary>
        <pre class="payload" id="plainTrace">${esc(traceText || "")}</pre>
      </details>
    </section>`;
}

function onPreviewClick(event) {
  const copy = event.target.closest("[data-copy]");
  if (copy) {
    navigator.clipboard.writeText(copy.dataset.copy);
    copy.textContent = "Copied";
    return;
  }
  if (event.target.closest("[data-copy-trace]")) {
    const text = $("plainTrace") ? $("plainTrace").textContent : (state.view && state.view.trace_text) || "";
    navigator.clipboard.writeText(text);
    return;
  }
  const download = event.target.closest("[data-download]");
  if (download && state.view) {
    const name = download.dataset.download === "log" ? "decision-log" : "assessment";
    const payload = download.dataset.download === "log" ? state.view.decision_log : state.view.assessment;
    downloadJson(`${state.view.assessment.request_id}-${name}.json`, payload);
    return;
  }
  const post = event.target.closest("[data-fill-post]");
  if (post) {
    state.post = post.dataset.fillPost;
    if ($("answer")) $("answer").value = post.dataset.fillPost;
    cliPreview();
    return;
  }
  const company = event.target.closest("[data-fill-company]");
  if (company) {
    state.company = company.dataset.fillCompany;
    if ($("answer")) $("answer").value = company.dataset.fillCompany;
    cliPreview();
    return;
  }
  const influencer = event.target.closest("[data-fill-influencer]");
  if (influencer) {
    state.influencer = influencer.dataset.fillInfluencer;
    if ($("answer")) $("answer").value = influencer.dataset.fillInfluencer;
    cliPreview();
    return;
  }
  if (event.target.closest("#sendAnswer")) sendAnswer();
}

function sendAnswer() {
  if (!state.view) return;
  const value = ($("answer").value || "").trim();
  if (!value) {
    $("formStatus").textContent = "Type an answer first.";
    return;
  }
  const slot = state.view.answer_slot || "claim";
  if (slot === "post") state.post = value;
  else if (slot === "company") state.company = value;
  else if (slot === "influencer") state.influencer = value;
  else {
    state.company = "";
    state.influencer = "";
    state.post = "";
    $("claim").value = value;
  }
  cliPreview();
  run();
}

function downloadJson(name, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

init();
