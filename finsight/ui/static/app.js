const state = {
  meta: null, view: null, log: null, stages: [], elapsed: 0,
  company: "", influencer: "", post: "",
};

const $ = (id) => document.getElementById(id);

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// Plain-language name for each pipeline node, shown while a run is in progress.
const STAGE_NAMES = {
  intake: "Reading the request",
  advice_gate_input: "Checking it is not a request for advice",
  scope_check: "Checking the company is covered",
  clarification: "Looking for missing details",
  collect_posts: "Finding the influencer's post",
  extract_claims: "Pulling out the factual claims",
  plan: "Planning the checks",
  gather_evidence: "Gathering evidence from filings and reports",
  audit: "Auditing the evidence",
  retry_or_accept: "Deciding whether to look again",
  apply_tier_policy: "Weighing source reliability",
  assemble: "Writing the answer",
  advice_gate_output: "Final advice check",
  finalize: "Finishing up",
  record: "Comparing the influencer's posts",
};

// What each verdict means for a reader who has never seen the pipeline.
const VERDICTS = {
  SUPPORTED: { tone: "good", meaning: "The sources back this claim up." },
  PARTIALLY_SUPPORTED: { tone: "mixed", meaning: "Part of the claim holds up; part does not or could not be confirmed." },
  CONTRADICTED: { tone: "bad", meaning: "The sources say something different." },
  INSUFFICIENT_EVIDENCE: { tone: "unknown", meaning: "FinSight could not find enough evidence to decide either way." },
};

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

function niceDate(iso) {
  if (!iso) return "";
  const [year, month, day] = String(iso).slice(0, 10).split("-");
  if (!day) return iso;
  return `${Number(day)} ${MONTHS[Number(month) - 1]} ${year}`;
}

function stageName(node) {
  return STAGE_NAMES[node] || String(node || "").replaceAll("_", " ");
}

function headline(view) {
  const assessment = view.assessment;
  if (assessment.overall_verdict) {
    const info = VERDICTS[assessment.overall_verdict] || { tone: "unknown", meaning: "" };
    return { text: view.overall_label || assessment.verdict_label, tone: info.tone, meaning: info.meaning };
  }
  const labels = {
    REFUSED: { text: "Not checked", tone: "neutral", meaning: "" },
    NEEDS_CLARIFICATION: { text: "One more detail needed", tone: "ask", meaning: "Answer the question below and FinSight will run the check again." },
    FAILED: { text: "The check failed", tone: "bad", meaning: "Something went wrong while checking. Open Details for the trace." },
    COMPLETED: { text: "Completed", tone: "neutral", meaning: "" },
  };
  return labels[assessment.status] || { text: assessment.status, tone: "neutral", meaning: "" };
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
  renderExamples();
  const asOf = state.meta.mode.corpus_as_of;
  if (asOf) {
    $("cutoffHint").textContent = `Only information published on or before this date is used. Leave blank to use the post's own date (or ${niceDate(asOf)}, the latest data, if the post is undated).`;
  }
  $("influencer").innerHTML += (state.meta.influencers || [])
    .map((name) => `<option value="${esc(name)}">${esc(name)}</option>`).join("");
  setInfluencer(influencerIn($("claim").value));
  cliPreview();
  $("cutoff").addEventListener("input", cliPreview);
  $("influencer").addEventListener("change", () => {
    state.influencer = $("influencer").value;
    cliPreview();
  });
  $("claim").addEventListener("input", () => {
    state.company = "";
    state.post = "";
    cliPreview();
  });
  $("clearHistory").addEventListener("click", clearHistory);
  $("history").addEventListener("click", onHistoryClick);
  renderHistory();
  $("runBtn").addEventListener("click", run);
  $("clearBtn").addEventListener("click", clearForm);
  $("claim").addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      run();
    }
  });
  $("examples").addEventListener("click", onExampleClick);
  $("preview").addEventListener("click", onPreviewClick);
}

function renderExamples() {
  const examples = state.meta.examples || [];
  if (!examples.length) return;
  $("examples").innerHTML = `
    <p class="examples__label">Try an example</p>
    <div class="examples__row">
      ${examples.map((item, index) => `<button type="button" class="chip" data-example="${index}" title="${esc(item.note || "")}">${esc(item.label)}</button>`).join("")}
    </div>`;
}

function onExampleClick(event) {
  const button = event.target.closest("[data-example]");
  if (!button) return;
  const item = state.meta.examples[Number(button.dataset.example)];
  $("claim").value = item.claim || "";
  state.company = item.company || "";
  setInfluencer(item.influencer || "");
  state.post = item.post || "";
  cliPreview();
  $("formStatus").textContent = item.note || "Example loaded.";
  $("claim").focus();
}

// The dropdown and state.influencer always agree; a typed name that is not listed is added.
function setInfluencer(name) {
  const select = $("influencer");
  if (name && ![...select.options].some((option) => option.value === name)) {
    select.add(new Option(name, name));
  }
  select.value = name || "";
  state.influencer = select.value;
}

// Pre-select the influencer when the claim already names one.
function influencerIn(text) {
  const lower = String(text || "").toLowerCase();
  return (state.meta.influencers || []).find((name) => lower.includes(name.toLowerCase())) || "";
}

function clearForm() {
  $("claim").value = "";
  $("cutoff").value = "";
  state.company = "";
  setInfluencer("");
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
  };
  if (!body.claim.trim()) {
    $("formStatus").textContent = "Paste a claim first.";
    $("claim").focus();
    return;
  }
  $("runBtn").disabled = true;
  $("runBtn").textContent = "Checking…";
  $("empty").hidden = true;
  $("result").innerHTML = "";
  $("details").innerHTML = "";
  state.view = null;
  state.stages = [];
  $("formStatus").textContent = "Running…";
  renderProgress([], 0);
  let response;
  try {
    response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    stopRun("The page could not reach the FinSight server.");
    return;
  }
  const data = await response.json();
  if (!response.ok) {
    stopRun(data.error || "The run was rejected.");
    return;
  }
  poll(data.job_id);
}

function stopRun(message) {
  $("formStatus").textContent = message;
  $("runBtn").disabled = false;
  $("runBtn").textContent = "Run check";
  $("progress").innerHTML = "";
  if (!state.view) $("empty").hidden = false;
}

async function poll(jobId) {
  let data;
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    data = await response.json();
    if (!response.ok) throw new Error(data.error || "Lost the run.");
  } catch (error) {
    stopRun(error.message);
    return;
  }
  state.stages = data.stages || [];
  state.elapsed = data.elapsed_ms;
  if (data.status === "running") {
    renderProgress(state.stages, data.elapsed_ms);
    setTimeout(() => poll(jobId), 400);
    return;
  }
  $("progress").innerHTML = "";
  $("runBtn").disabled = false;
  $("runBtn").textContent = "Run check";
  if (data.status === "error") {
    stopRun(data.error || "The run failed.");
    return;
  }
  state.view = data.result;
  state.log = data.result.decision_log;
  renderResult(data.result);
  renderDetails(data.result);
  saveHistory(data.result);
  $("formStatus").textContent = `Done in ${(data.elapsed_ms / 1000).toFixed(1)} s.`;
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderProgress(stages, elapsed) {
  const last = stages[stages.length - 1];
  const current = last ? stageName(last.node) : "Starting";
  const done = stages.map((stage) => `<li>${esc(stageName(stage.node))}</li>`).join("");
  $("progress").innerHTML = `
    <section class="card progress" aria-live="polite">
      <div class="progress__head">
        <span class="spinner" aria-hidden="true"></span>
        <div>
          <p class="progress__title">Checking your claim…</p>
          <p class="progress__now">${esc(current)} · ${(elapsed / 1000).toFixed(1)} s</p>
        </div>
      </div>
      <div class="progress__bar"><span style="width:${Math.min(95, 8 + stages.length * 7)}%"></span></div>
      ${done ? `<ol class="progress__list">${done}</ol>` : ""}
    </section>`;
}

const POST_NOTES = [
  "A figure that appears only on a chart, a slide, or an image is not in the stored text, so it was not checked.",
  "The transcript does not mark who is speaking, so the speaker was not separated.",
  "This transcript has no timestamps, so the report cannot point to a moment in the video.",
  "Checked the lines in this post that match the claim in the prompt.",
];
const ADVICE_NOTE = "A sentence that told you to buy or sell was removed. The claim and the labels are unchanged.";

function splitCheckedPost(text) {
  const source = String(text || "");
  const cuts = [];
  for (const note of POST_NOTES) {
    const at = source.indexOf(note);
    if (at >= 0) cuts.push({ at, text: note, video: "" });
  }
  const videoAt = source.indexOf("Video: ");
  if (videoAt >= 0) {
    const rest = source.slice(videoAt + "Video: ".length);
    const url = (rest.split(/\s/)[0] || "").replace(/[.)]+$/, "");
    cuts.push({ at: videoAt, text: "", video: url });
  }
  cuts.sort((left, right) => left.at - right.at);
  const leadEnd = cuts.length ? cuts[0].at : source.length;
  return {
    lead: source.slice(0, leadEnd).trim(),
    notes: cuts.filter((cut) => cut.text).map((cut) => cut.text),
    video: (cuts.find((cut) => cut.video) || {}).video || "",
  };
}

function splitSummary(text) {
  const source = String(text || "");
  if (source.startsWith(ADVICE_NOTE)) {
    return { advice: ADVICE_NOTE, text: source.slice(ADVICE_NOTE.length).trim() };
  }
  return { advice: "", text: source };
}

function unitsOf(view) {
  if (view.units && view.units.length) return view.units;
  return (view.assessment.sub_claim_results || []).map((row) => ({
    heading: row.sub_claim.text,
    verdict: row.verdict,
    label: row.verdict_label,
    partial_reason: null,
    is_parent: false,
    ids: [row.sub_claim.sub_claim_id],
  }));
}

function rowsById(view) {
  return Object.fromEntries((view.assessment.sub_claim_results || []).map((row) => [row.sub_claim.sub_claim_id, row]));
}

function sourceUrl(item) {
  if (typeof item.url === "string" && /^https?:\/\//.test(item.url)) return item.url;
  if (typeof item.publisher_page === "string" && /^https?:\/\//.test(item.publisher_page)) return item.publisher_page;
  return null;
}

// One line per checked part of the claim: verdict, why, and the main source behind it.
function findingHtml(unit, rows) {
  const children = unit.ids.map((id) => rows[id]).filter(Boolean);
  const why = children.map((row) => row.explanation || row.rationale).find(Boolean) || unit.partial_reason || "";
  const evidence = children.flatMap((row) => [...(row.supporting || []), ...(row.challenging || [])])[0];
  const url = evidence ? sourceUrl(evidence) : null;
  // The tier note stays in Details; here the document name and date are enough.
  const label = evidence ? String(evidence.source_label || evidence.doc_type || "").replace(/\s*\(Tier[^)]*\)\s*$/, "") : "";
  const source = evidence
    ? `<p class="finding__source"><span>Source</span> ${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)}</a>` : esc(label)}${evidence.citation_span ? ` · ${esc(evidence.citation_span)}` : ""}</p>`
    : "";
  const tone = (VERDICTS[unit.verdict] || {}).tone || "unknown";
  return `
    <li class="finding tone-${tone}">
      <span class="dot" aria-hidden="true"></span>
      <div>
        <div class="finding__head">
          <p class="finding__text">${esc(unit.heading)}</p>
          <span class="verdict ${verdictClass(unit.verdict)}">${esc(unit.label)}</span>
        </div>
        ${why ? `<p class="finding__why">${esc(why)}</p>` : ""}
        ${source}
      </div>
    </li>`;
}

function renderResult(view) {
  const assessment = view.assessment;
  const head = headline(view);
  const summary = splitSummary(assessment.summary || assessment.refusal_reason || assessment.clarification_question || "");
  const units = unitsOf(view);
  const rows = rowsById(view);
  const post = splitCheckedPost(assessment.checked_post);
  const violations = (view.decision_log.temporal_violations || []).map((item) => `
    <p class="violation">Cut-off leak: ${esc(item.evidence_id)} for ${esc(item.sub_claim_id)} was published ${esc(item.publication_date)}, after ${esc(item.cutoff_date)}. ${esc(item.produced_by)} produced it and it was dropped.</p>
  `).join("");
  const showSummary = summary.text && assessment.status !== "NEEDS_CLARIFICATION";
  const meta = [];
  if (assessment.as_of_date) meta.push(`Information up to ${esc(niceDate(assessment.as_of_date))}`);
  if (post.video) meta.push(`<a href="${esc(post.video)}" target="_blank" rel="noopener">Watch the video</a>`);
  $("result").innerHTML = `
    <article class="card answer tone-${head.tone}">
      <div class="answer__head">
        <span class="answer__icon" aria-hidden="true"></span>
        <div>
          <p class="eyebrow">${assessment.overall_verdict ? "Verdict" : "Status"}</p>
          <h2 class="answer__verdict">${esc(head.text)}</h2>
          ${head.meaning ? `<p class="answer__meaning">${esc(head.meaning)}</p>` : ""}
        </div>
      </div>
      <blockquote class="answer__claim">${esc(assessment.claim || $("claim").value)}</blockquote>
      ${showSummary ? `<p class="answer__summary">${esc(summary.text)}</p>` : ""}
      ${units.length ? `<ul class="findings">${units.map((unit) => findingHtml(unit, rows)).join("")}</ul>` : ""}
      ${violations}
      ${meta.length ? `<p class="answer__meta">${meta.join(" · ")}</p>` : ""}
      ${clarificationHtml(view)}
      ${suggestionsHtml(view)}
      <p class="disclaimer">${esc(assessment.boundary_disclaimer)}</p>
    </article>`;
  $("empty").hidden = true;
}

// Related claims from stored posts, picked by the model, that the user can check with one click.
function suggestionsHtml(view) {
  const items = view.suggestions || [];
  if (!items.length || view.assessment.status === "NEEDS_CLARIFICATION") return "";
  return `
    <div class="suggest">
      <p class="suggest__label">Check next</p>
      <div class="suggest__list">
        ${items.map((text) => `
          <button type="button" class="suggest__item" data-suggest="${esc(text)}">
            <span>${esc(text)}</span><span class="suggest__go" aria-hidden="true">→</span>
          </button>`).join("")}
      </div>
    </div>`;
}

function clarificationHtml(view) {
  const question = view.assessment.clarification_question;
  if (view.assessment.status !== "NEEDS_CLARIFICATION" || !question) return "";
  const choices = (view.post_choices || []).map((choice) => `
    <button type="button" class="choice" data-fill-post="${esc(choice.video_id)}">
      ${esc(choice.title || choice.video_id)}
      <small>${esc(choice.date)} · ${esc(choice.video_id)}</small>
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
  const ask = question.split("\n")[0];
  return `
    <div class="clarify">
      <p class="clarify__q">${esc(ask)}</p>
      ${choices || companies || influencers ? `<div class="choices">${choices}${companies}${influencers}</div>` : ""}
      <div class="clarify__row">
        <label class="field"><span>${esc(slotLabel)}</span><input id="answer" placeholder="Pick above or type here" /></label>
        <button type="button" class="run-btn" id="sendAnswer">Run again</button>
      </div>
    </div>`;
}

// Everything a curious reader or a tester may want, folded away by default.
function renderDetails(view) {
  const assessment = view.assessment;
  const post = splitCheckedPost(assessment.checked_post);
  const summary = splitSummary(assessment.summary || "");
  const sections = [];
  if (unitsOf(view).length) {
    sections.push(fold("Evidence and citations", "Every passage and filing figure behind each verdict", `
      <div id="checks">${checksHtml(view)}</div>`));
  }
  if (view.record && view.record.posts && view.record.posts.length) {
    sections.push(fold("Posts in this record", `${view.record.posts.length} post${view.record.posts.length === 1 ? "" : "s"}`, recordHtml(view.record)));
  }
  if (post.lead || post.notes.length || summary.advice) {
    sections.push(fold("About the post", "What was checked and what could not be", `
      ${post.lead ? `<p class="body-text">${esc(post.lead)}</p>` : ""}
      ${post.notes.length ? `<ul class="notes">${post.notes.map((note) => `<li>${esc(note)}</li>`).join("")}</ul>` : ""}
      ${summary.advice ? `<p class="body-text">${esc(ADVICE_NOTE)}</p>` : ""}`));
  }
  sections.push(fold("Steps", `${state.stages.length} stage${state.stages.length === 1 ? "" : "s"} · ${(state.elapsed / 1000).toFixed(1)} s`, `
    <ol class="timeline" id="timeline">${state.stages.map(stageHtml).join("") || `<li class="muted">No stages recorded.</li>`}</ol>`));
  sections.push(fold("Run info", assessment.request_id, runInfoHtml(view)));
  sections.push(fold("Decision log", "Gates, routing, agent calls, audits, retries", traceHtml(view.decision_log, view.stats, view.trace_text, view.warning)));
  $("details").innerHTML = `
    <section class="card details">
      <h2 class="details__title">Details</h2>
      ${sections.join("")}
    </section>`;
}

function fold(title, hint, body) {
  return `
    <details class="fold">
      <summary><span class="fold__title">${esc(title)}</span>${hint ? `<span class="fold__hint">${esc(hint)}</span>` : ""}</summary>
      <div class="fold__body">${body}</div>
    </details>`;
}

function runInfoHtml(view) {
  const assessment = view.assessment;
  const gates = (view.decision_log.gate_decisions || []).map((gate) => `
    <span class="tag ${gate.flagged ? "against" : "for"}">${esc(gate.gate)} ${gate.flagged ? "flagged" : "pass"}</span>
  `).join("");
  return `
    <div class="info-grid">
      <span>Status</span><span>${esc(assessment.status)}</span>
      <span>Request id</span><span class="mono">${esc(assessment.request_id)}</span>
      ${view.cutoff_note ? `<span>Cut-off</span><span>${esc(niceDate(assessment.as_of_date))} (${esc(view.cutoff_note)})</span>` : ""}
    </div>
    ${gates ? `<div class="tag-row">${gates}</div>` : ""}
    <div class="stat-row">${statHtml(view.stats)}</div>
    ${assessment.refusal_reason && !(assessment.summary || "").includes(assessment.refusal_reason) ? `<p class="body-text">${esc(assessment.refusal_reason)}</p>` : ""}
    ${assessment.status === "COMPLETED" && view.limits ? `<p class="body-text muted">${esc(view.limits)}</p>` : ""}
    <div class="button-row">
      <button type="button" class="text-btn" data-copy="${esc(assessment.request_id)}">Copy id</button>
      <button type="button" class="text-btn" data-download="assessment">Assessment JSON</button>
      <button type="button" class="text-btn" data-download="log">Decision log JSON</button>
    </div>
    ${view.cli ? `<p class="cli-label">Repeat this run</p><pre class="cli">${esc(view.cli)}</pre>` : ""}`;
}

function stageHtml(stage) {
  const lines = stage.lines.map((line) => `<p class="${lineClass(line)}">${esc(line)}</p>`).join("");
  const timing = stage.ms == null ? "" : `<span class="step__ms">${stage.ms} ms</span>`;
  return `
    <li class="step ${stepTone(stage.lines)}">
      <span class="step__n">${esc(stage.step)}</span>
      <div>
        <div class="step__head">
          <span class="step__node">${esc(stageName(stage.node))}</span>
          <span class="step__role">${esc(stage.node)} · ${esc(stage.role)}</span>
          ${timing}
        </div>
        ${lines}
      </div>
    </li>`;
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

function recordHtml(record) {
  const unfinished = record.unfinished && record.unfinished.length
    ? `<p class="line-stop">Unfinished posts: ${esc(record.unfinished.join(", "))}</p>` : "";
  const posts = record.posts.map((post) => `
    <details class="fold fold--inner">
      <summary><span class="fold__title">${esc(post.title || post.video_id)}</span><span class="fold__hint">${esc(post.label)} · ${esc(niceDate(post.upload_date) || "undated")}</span></summary>
      <div class="fold__body">
        <p class="body-text muted">Video ${esc(post.video_id)} · checked as of ${esc(niceDate(post.cutoff_date) || "unknown")}.</p>
        ${post.view ? checksHtml(post.view) : `<p class="muted">This post has no decision log.</p>`}
      </div>
    </details>`).join("");
  return `
    ${record.consistency ? `<p class="body-text">${esc(record.consistency)}</p>` : ""}
    ${record.accuracy ? `<p class="body-text">${esc(record.accuracy)}</p>` : ""}
    ${unfinished}
    ${posts}`;
}

function checksHtml(view) {
  const rows = rowsById(view);
  const units = unitsOf(view);
  if (!units.length) {
    return `<p class="muted">No sub-claim was checked. The status above is the whole result.</p>`;
  }
  const gaps = (view.assessment.evidence_gaps || []).map((gap) => `<p class="gap">Gap: ${esc(gap)}</p>`).join("");
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
    ${row.explanation ? `<p class="why">${esc(row.explanation)}</p>` : ""}
    ${evidenceGroup(row.supporting, "Evidence for", "for")}
    ${evidenceGroup(row.challenging, "Evidence against", "against")}
    ${!row.supporting.length && !row.challenging.length ? `<p class="where missing">No evidence was attached to ${esc(claim.sub_claim_id)}.</p>` : ""}
    <details class="mini">
      <summary>Audit notes</summary>
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
    </details>`;
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
  const documentUrl = typeof item.url === "string" && /^https?:\/\//.test(item.url) ? item.url : null;
  const publisherUrl = typeof item.publisher_page === "string" && /^https?:\/\//.test(item.publisher_page)
    ? item.publisher_page : null;
  const sourceUrl = documentUrl || publisherUrl;
  const linkLabel = documentUrl ? "Link" : "Publisher/platform";
  const link = sourceUrl
    ? `<p class="link"><span>${linkLabel}</span> <a href="${esc(sourceUrl)}" target="_blank" rel="noopener">${esc(sourceUrl)}</a></p>`
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

function traceHtml(log, stats, traceText, warning) {
  if (!log) return `<p class="muted">No decision log.</p>`;
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
  const table = (head, body, empty) => body
    ? `<div class="table-wrap"><table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>`
    : `<p class="muted">${empty}</p>`;
  return `
    ${warning ? `<p class="violation">${esc(warning)}</p>` : ""}
    <p class="body-text muted">${esc(log.input || "")}</p>
    <h3>Routing plan</h3>
    ${table(["Id", "Type", "Collectors", "Cap", "Tiers", "Note"], routes, "No routing plan. The run stopped before it.")}
    <h3>Agent calls</h3>
    ${table(["Time", "Agent", "Sub-claim", "Latency", "Tokens", "Tools", "Result"], callRows, "No agent calls.")}
    <h3>Audits</h3>
    ${table(["Id", "", "Verdict", "Why it failed", "Retry feedback"], audits, "No audit.")}
    <h3>Retries</h3>
    ${table(["Id", "Attempt", "", "Feedback"], retries, "No retries.")}
    <h3>Final verdicts</h3>
    <div class="tag-row">${finals || `<span class="muted">None recorded.</span>`}</div>
    <h3>Events</h3>
    <ol class="timeline">
      ${(log.events || []).map((event, index) => `
        <li class="step">
          <span class="step__n">${index + 1}</span>
          <div>
            <div class="step__head"><span class="step__node">${esc(event.node)}</span><span class="step__ms">${esc((event.at || "").replace("T", " ").slice(0, 19))}</span></div>
            <p>${esc(event.message)}</p>
          </div>
        </li>`).join("") || `<li class="muted">No events.</li>`}
    </ol>
    <details class="mini">
      <summary>Plain trace</summary>
      <button type="button" class="text-btn" data-copy-trace="1">Copy plain trace</button>
      <pre class="payload" id="plainTrace">${esc(traceText || "")}</pre>
    </details>`;
}

function onPreviewClick(event) {
  const copy = event.target.closest("[data-copy]");
  if (copy) {
    navigator.clipboard.writeText(copy.dataset.copy);
    copy.textContent = "Copied";
    return;
  }
  const copyTrace = event.target.closest("[data-copy-trace]");
  if (copyTrace) {
    const text = $("plainTrace") ? $("plainTrace").textContent : (state.view && state.view.trace_text) || "";
    navigator.clipboard.writeText(text);
    copyTrace.textContent = "Copied";
    return;
  }
  const download = event.target.closest("[data-download]");
  if (download && state.view) {
    const name = download.dataset.download === "log" ? "decision-log" : "assessment";
    const payload = download.dataset.download === "log" ? state.view.decision_log : state.view.assessment;
    downloadJson(`${state.view.assessment.request_id}-${name}.json`, payload);
    return;
  }
  const fills = [["fillPost", "post"], ["fillCompany", "company"], ["fillInfluencer", "influencer"]];
  for (const [key, slot] of fills) {
    const button = event.target.closest(`[data-${key.replace(/[A-Z]/g, (ch) => `-${ch.toLowerCase()}`)}]`);
    if (!button) continue;
    if (slot === "influencer") setInfluencer(button.dataset[key]);
    else state[slot] = button.dataset[key];
    if ($("answer")) $("answer").value = button.dataset[key];
    document.querySelectorAll(".choice.is-active").forEach((node) => node.classList.remove("is-active"));
    button.classList.add("is-active");
    cliPreview();
    return;
  }
  const example = event.target.closest("[data-try]");
  if (example) {
    $("claim").value = example.dataset.try;
    $("cutoff").value = example.dataset.cutoff || "";
    state.company = "";
    state.post = "";
    setInfluencer(influencerIn(example.dataset.try));
    cliPreview();
    $("formStatus").textContent = "Example loaded. Press Run check.";
    $("claim").focus();
    return;
  }
  const suggestion = event.target.closest("[data-suggest]");
  if (suggestion) {
    $("claim").value = suggestion.dataset.suggest;
    $("cutoff").value = "";
    state.company = "";
    state.post = "";
    setInfluencer(influencerIn(suggestion.dataset.suggest));
    cliPreview();
    run();
    return;
  }
  if (event.target.closest("#sendAnswer")) sendAnswer();
}

function sendAnswer() {
  if (!state.view) return;
  const value = ($("answer").value || "").trim();
  if (!value) {
    $("formStatus").textContent = "Pick or type an answer first.";
    return;
  }
  const slot = state.view.answer_slot || "claim";
  if (slot === "post") state.post = value;
  else if (slot === "company") state.company = value;
  else if (slot === "influencer") setInfluencer(value);
  else {
    state.company = "";
    setInfluencer("");
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

// Past checks live in this browser only, so each visitor sees just their own runs.
const HISTORY_KEY = "finsight-history";
const HISTORY_MAX = 20;

function loadHistory() {
  try {
    const items = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(items) ? items : [];
  } catch (error) {
    return [];
  }
}

function storeHistory(items) {
  // Drop the oldest entries until the list fits the browser's storage quota.
  for (let keep = items.length; keep >= 0; keep -= 1) {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, keep)));
      return;
    } catch (error) {
      if (keep === 0) return;
    }
  }
}

function saveHistory(view) {
  const head = headline(view);
  const entry = {
    id: view.assessment.request_id || String(Date.now()),
    at: new Date().toISOString(),
    claim: view.assessment.claim || $("claim").value,
    form: { claim: $("claim").value, cutoff: $("cutoff").value, influencer: state.influencer },
    label: head.text,
    tone: head.tone,
    view,
    stages: state.stages,
    elapsed: state.elapsed,
  };
  const items = loadHistory().filter((item) => item.id !== entry.id);
  items.unshift(entry);
  storeHistory(items.slice(0, HISTORY_MAX));
  renderHistory(entry.id);
}

function timeAgo(iso) {
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return niceDate(iso);
}

function renderHistory(activeId) {
  const items = loadHistory();
  $("clearHistory").hidden = !items.length;
  $("history").innerHTML = items.length
    ? items.map((item) => `
      <li>
        <button type="button" class="history__item tone-${esc(item.tone)}${item.id === activeId ? " is-active" : ""}" data-history="${esc(item.id)}">
          <span class="dot" aria-hidden="true"></span>
          <span class="history__text">
            <span class="history__claim">${esc(item.claim)}</span>
            <span class="history__meta">${esc(item.label)} · ${esc(timeAgo(item.at))}</span>
          </span>
        </button>
      </li>`).join("")
    : `<li class="history__empty">Checks you run will be listed here.</li>`;
}

function onHistoryClick(event) {
  const button = event.target.closest("[data-history]");
  if (!button) return;
  const item = loadHistory().find((entry) => entry.id === button.dataset.history);
  if (!item) return;
  $("claim").value = item.form.claim || item.claim;
  $("cutoff").value = item.form.cutoff || "";
  setInfluencer(item.form.influencer || "");
  state.company = "";
  state.post = "";
  cliPreview();
  state.view = item.view;
  state.log = item.view.decision_log;
  state.stages = item.stages || [];
  state.elapsed = item.elapsed || 0;
  $("progress").innerHTML = "";
  renderResult(item.view);
  renderDetails(item.view);
  renderHistory(item.id);
  $("formStatus").textContent = `Showing a saved check from ${timeAgo(item.at)}.`;
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function clearHistory() {
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch (error) {
    // Storage may be blocked; the list below is cleared either way.
  }
  renderHistory();
}

init();
