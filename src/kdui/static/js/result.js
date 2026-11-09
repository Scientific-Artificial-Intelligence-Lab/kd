// Result page: one run. The typeset equation with its terms, the
// native score under its own name, metrics, the three verification checks
// (each its own result: value, or kd's reason), the forward-solve figure and
// the term-ablation table, convergence, candidates, the sketch outcome, the
// embedded KD report, the full record, and the resume and edit-and-rerun
// actions. Everything shown is what the API returns; nothing is computed here.

import { get, post, artifactUrl, followJob, TERMINAL_STATES } from "./api.js";
import {
  h, frag, href, go, replaceRoute, fmtNumber, fmtBytes, fmtDuration, fmtTime, button, linkButton, tag, pageHead,
  panel, kv, table, warningBlock, loadingState, errorState, toast, statusBadge, stopReasonText, smoothingText, equation, schemaForm,
} from "./ui.js";
import { linePlot, scatterPlot } from "./plots.js";
import { iterationUnit } from "./run.js";

let S = null; // page state; null after unmount

const enc = encodeURIComponent;
const mono = (text) => h("span", { class: "mono", style: "font-size:.8rem" }, text);
const tex = (latex) => equation(latex, { size: "inline" });

/** Settle a request: { value } or { error }; a part that fails does not block the page. */
const settle = (promise) => promise.then((value) => ({ value }), (error) => ({ error }));

const CHECKS = [
  { id: "forward_error", title: "Forward simulation", fn: "kd.forward_error" },
  { id: "ablation", title: "Term ablation", fn: "kd.ablate_terms" },
  { id: "block_dispersion", title: "Block dispersion", fn: "kd.block_dispersion" },
];

const SECTIONS = [
  ["verify", "Verify"],
  ["fit", "Fit"],
  ["candidates", "Candidates"],
  ["report", "Report"],
  ["record", "Record"],
];

// Captions of KD's report figures, by file stem; other figures show their file name.
const FIGURE_TEXT = {
  parity: ["Parity", "Predicted against observed target values."],
  pde_residual_field: ["PDE residual field", "Fitted target, the equation's prediction, and the residual."],
  field_comparison: ["Field comparison", "The measured field, the field integrated from the discovered equation, and their difference."],
  forward_error: ["Forward simulation", "The measured field, the field integrated from the discovered equation, and their difference."],
  error_heatmap: ["Error heat map", "Signed error of the integrated field at each grid point."],
  time_slices: ["Time slices", "Measured and integrated field at the labelled times."],
};

const RUN_STATUS = {
  completed: ["completed", "Completed"],
  raised: ["failed", "Raised an error"],
  unsealed: ["interrupted", "Not sealed"],
  no_record: ["interrupted", "No record"],
};

// ---------------------------------------------------------------------------
// Page contract

export async function mount(container, params) {
  S = { container, params, followers: new Map(), jobs: {}, latest: {}, hosts: {} };
  if (!params.id) {
    container.append(
      pageHead({ title: "Result", sub: "The equation of one run, its verification and its report." }),
      h("div", { class: "state empty" }, h("p", null, "No run is chosen. Open a finished run from the run list."), h("div", { class: "actions" }, linkButton("All runs", href("compare")))),
    );
    return;
  }
  await load();
}

export function unmount() {
  if (!S) return;
  for (const handle of S.followers.values()) handle.stop();
  S = null;
}

/** The router asks before it reloads this page: another section of the run
 *  already shown (the header's Verify and Report steps) is a scroll. */
export function revisit(params) {
  if (!S || !S.run || params.id !== S.run.run_id) return false;
  scrollToSection(params.section || "verify");
  return true;
}

// ---------------------------------------------------------------------------
// Loading

async function load() {
  const { container, params } = S;
  container.replaceChildren(loadingState("Loading the run"));
  let run;
  try {
    run = await get(`/runs/${enc(params.id)}`);
  } catch (err) {
    if (!S || S.container !== container) return;
    container.replaceChildren(
      pageHead({ title: "Result", crumbs: [["Runs", href("compare")], [params.id, null]] }),
      errorState(err, { title: "This run could not be loaded.", retry: load }),
    );
    return;
  }
  const side = await loadSide(run);
  if (!S || S.container !== container) return;
  S.run = run;
  S.side = side;
  S.unit = iterationUnit(side.algo); // sga counts generations, pysr rounds, ...
  S.verifications = side.verifications.value ? side.verifications.value.verifications : [];
  render();
  followOpenVerifications(side.jobs.value ? side.jobs.value.jobs : []);
  if (params.section) requestAnimationFrame(() => scrollToSection(params.section));
}

async function loadSide(run) {
  const id = enc(run.run_id);
  const [verifications, checkpoints, runFiles, jobFiles, job, dataset, jobs, caps] = await Promise.all([
    settle(get(`/runs/${id}/verifications`)),
    settle(get(`/runs/${id}/checkpoints`)),
    settle(get(`/runs/${id}/artifacts`)),
    run.job_id ? settle(get(`/jobs/${enc(run.job_id)}/artifacts`)) : { value: { artifacts: [] } },
    run.job_id ? settle(get(`/jobs/${enc(run.job_id)}`)) : { value: null },
    run.dataset_id ? settle(get(`/datasets/${enc(run.dataset_id)}`)) : { value: null },
    settle(get("/jobs", { kind: "verify", limit: 100 })),
    settle(get("/capabilities")),
  ]);
  const files = [];
  const seen = new Set();
  for (const part of [runFiles, jobFiles]) {
    for (const a of part.value ? part.value.artifacts : []) {
      if (!seen.has(a.name)) files.push(a);
      seen.add(a.name);
    }
  }
  const failed = [runFiles.error, jobFiles.error].filter(Boolean);
  const filesError = failed.length ? failed[0] : null;
  const algo = caps.value && caps.value.snapshot ? caps.value.snapshot.algorithms.find((a) => a.algorithm === run.algorithm) ?? null : null;
  return { verifications, checkpoints, files, filesError, job, dataset, jobs, caps, algo };
}

// ---------------------------------------------------------------------------
// Rendering

function render() {
  const { run, container } = S;
  S.hosts.resume = h("div", { class: "r-resume", hidden: true });
  container.replaceChildren(frag(
    head(run),
    S.hosts.resume,
    notices(run),
    lawCard(run),
    jumpBar(),
    verifySection(run),
    fitSection(run),
    candidatesSection(run),
    reportSection(run),
    recordSection(run),
  ));
  renderVerification();
}

/** Why the run ended, in the words the run list uses: the job's reason when the
 *  UI ran it (kd's "callback" is the stop control), else the record's. */
function endReason(run) {
  const job = S.side.job.value;
  return (job && job.stop_reason) || run.stop_reason;
}

function head(run) {
  const title = run.algorithm ? `${run.algorithm} on ${run.dataset_name || "unknown data"}${run.seed !== null ? `, seed ${run.seed}` : ""}` : `Run ${run.run_id}`;
  const [cls, label] = RUN_STATUS[run.status] || ["queued", run.status];
  const cost = run.cost;
  const facts = [
    endReason(run) && stopReasonText(endReason(run)),
    cost && cost.wallclock_seconds !== null && fmtDuration(cost.wallclock_seconds),
    cost && cost.peak_rss_bytes !== null && `peak memory ${fmtBytes(cost.peak_rss_bytes)}`,
  ].filter(Boolean);
  const sub = h(
    "span",
    null,
    h("span", { class: `st ${cls}` }, label),
    facts.length ? `  ${facts.join(", ")}. ` : " ",
    run.origin === "imported" ? tag("imported run") : null,
    run.prior_knowledge && run.prior_knowledge.sketch ? [" ", tag("with sketch")] : null,
    lineageText(run.lineage),
  );
  return pageHead({ title, sub, crumbs: [["Runs", href("compare")], [run.run_id, null]], actions: headActions(run) });
}

function lineageText(lineage) {
  if (!lineage) return null;
  if (lineage.resumed_from_run_id) {
    const at = lineage.resume_iteration !== null ? ` at ${S.unit.singular} ${lineage.resume_iteration}` : "";
    return [" Resumed from ", h("a", { href: href("result", lineage.resumed_from_run_id) }, lineage.resumed_from_run_id), at, lineage.reseed ? " with a new seed." : "."];
  }
  if (lineage.cloned_from_run_id) return [" Edited copy of ", h("a", { href: href("result", lineage.cloned_from_run_id) }, lineage.cloned_from_run_id), "."];
  return null;
}

function headActions(run) {
  const ck = S.side.checkpoints;
  const resumable = Boolean(ck.value && ck.value.resumable);
  const resumeWhy = ck.error ? `Checkpoints could not be read: ${ck.error.message}` : ck.value && ck.value.reason;
  const canRerun = Boolean(run.request || run.config);
  return [
    run.equation ? linkButton("Add to comparison", href("compare", null, { runs: run.run_id })) : null,
    button("Edit and rerun", { disabled: !canRerun, title: canRerun ? "Open the search form filled with this run's settings" : "This run has neither a request nor a recorded configuration.", onClick: () => go("search", null, { clone: run.run_id }) }),
    button("Resume", { disabled: !resumable, title: resumable ? "Continue this run from a checkpoint" : resumeWhy, onClick: toggleResume }),
  ];
}

function notices(run) {
  const job = S.side.job.value;
  const parts = [];
  if (run.status === "raised") {
    const text = job && job.error ? `${job.error.error_type}: ${job.error.message}` : "The run raised an error before its seal; kd wrote no equation.";
    parts.push(h("div", { class: "warning", role: "note" }, h("b", null, "kd raised an error. "), text));
  }
  if (run.status === "unsealed") {
    parts.push(
      h(
        "div",
        { class: "warning r-row", role: "note" },
        h("span", null, "This run has not sealed: it is still running, or it ended before kd wrote its record."),
        run.job_id ? linkButton("Follow the run", href("run", run.job_id), { small: true }) : null,
      ),
    );
  }
  if (run.status !== "completed" && job && job.warnings && job.warnings.length) parts.push(warningBlock(job.warnings));
  return parts.length ? h("div", { class: "stack r-notices" }, parts) : null;
}

// ---------------------------------------------------------------------------
// Equation card

function lawCard(run) {
  if (!run.equation) {
    const why = run.status === "completed" ? "The record holds no equation." : "There is no sealed equation for this run.";
    return panel({}, h("p", { class: "note" }, why), referenceRow(run));
  }
  const card = h("section", { class: "panel lawcard", "aria-label": "Discovered equation" }, equation(run.equation.latex, { size: "large" }), reads(run));
  S.hosts.ledger = h("div");
  card.append(frag(S.hosts.ledger, referenceRow(run)));
  return card;
}

function reads(run) {
  const score = run.native_score;
  const m = run.metrics || {};
  const read = (label, value, w = null) => value !== null && value !== undefined && h("div", { class: "r" }, h("span", { class: "l" }, label), h("span", { class: "v" }, value), w && h("span", { class: "w" }, w));
  const better = (s) => (s.direction === "min" ? "lower is better" : "higher is better");
  return h(
    "div",
    { class: "reads" },
    score && read(`${score.kind}, ${run.algorithm}'s native score`, fmtNumber(score.value), `${better(score)}; compare only with other ${run.algorithm} runs`),
    read("nmse", m.nmse === null || m.nmse === undefined ? null : fmtNumber(m.nmse), "of the reported equation, from the run's record"),
    read(["R² on ", tex(run.equation.lhs_latex)], m.r2 === null || m.r2 === undefined ? null : fmtNumber(m.r2, 6)),
    read("MSE", m.mse === null || m.mse === undefined ? null : fmtNumber(m.mse)),
    read("Terms", run.equation.terms.length),
    read("Complexity", m.complexity ?? null),
  );
}

/** The term ledger: one column per term; the ablation row when the check exists. */
function renderLedger() {
  const host = S.hosts.ledger;
  if (!host) return;
  const terms = S.run.equation.terms;
  const ablation = S.latest.ablation && S.latest.ablation.result;
  const row = (label, cells, head = false) => h("div", { class: head ? "row head" : "row" }, h("span", null, label), cells);
  const rows = [
    row("Term", terms.map((t) => h("span", null, tex(t.latex))), true),
    row("Coefficient", terms.map((t) => h("span", { class: "num" }, fmtNumber(t.coefficient)))),
  ];
  if (ablation) {
    const byTerm = new Map(ablation.rows.map((r) => [r.dropped_latex, r]));
    rows.push(
      row(
        "nmse without this term",
        terms.map((t) => {
          const r = byTerm.get(t.latex);
          return h("span", { class: "num" }, r ? fmtNumber(r.nmse) : "n/a", r && r.relative_rise !== null ? h("span", { class: "src" }, `MSE rise ${fmtNumber(r.relative_rise)} times the full equation's MSE`) : null);
        }),
      ),
    );
  }
  host.replaceChildren(h("div", { class: "ledger", style: `--terms:${terms.length}` }, rows));
}

function referenceRow(run) {
  const detail = S.side.dataset.value;
  if (!detail || !detail.reference_available) return null;
  const host = h("div", { class: "r-ref" });
  const draw = (d) => {
    const shown = d.reference_shown && d.reference_equation;
    const toggle = button(shown ? "Hide reference equation" : "Show reference equation", {
      small: true,
      onClick: async () => draw(await post(`/datasets/${enc(d.dataset_id)}/reference`, { shown: !d.reference_shown })),
    });
    host.replaceChildren(frag(
      shown ? h("span", null, "Reference equation of ", h("b", null, d.label), ": ", tex(d.reference_equation.latex)) : h("span", null, "Reference equation hidden."),
      toggle,
      h("span", { class: "note" }, `The toggle belongs to the dataset ${d.label}. Reports of runs started while it is on include the reference equation.`),
    ));
  };
  draw(detail);
  return host;
}

// ---------------------------------------------------------------------------
// Section navigation

function jumpBar() {
  const buttons = SECTIONS.map(([id, label]) => h("button", { type: "button", "data-section": id, onClick: () => scrollToSection(id) }, label));
  S.hosts.jump = h("nav", { class: "r-jump", "aria-label": "Sections" }, buttons);
  return S.hosts.jump;
}

function scrollToSection(id) {
  if (!S) return;
  const target = S.container.querySelector(`#r-${id}`);
  if (!target) return;
  target.scrollIntoView({ block: "start" });
  for (const b of S.hosts.jump.querySelectorAll("button")) b.classList.toggle("on", b.dataset.section === id);
  replaceRoute("result", S.run.run_id, id === "verify" ? null : { section: id });
}

const sectionTitle = (id, title, meta = null, actions = null) =>
  h("div", { class: "section-title", id: id ? `r-${id}` : null }, h(id ? "h2" : "h3", null, title), meta && h("span", { class: "meta" }, meta), actions && h("span", { class: "actions" }, actions));

// ---------------------------------------------------------------------------
// Verification

function verifySection(run) {
  const hasEquation = Boolean(run.equation);
  const all = button("Run all three checks", { small: true, disabled: !hasEquation, title: hasEquation ? null : "There is no sealed equation to check.", onClick: () => startVerify(CHECKS.map((c) => c.id)) });
  for (const c of CHECKS) S.hosts[c.id] = h("section", { class: "panel measure", "aria-label": c.title });
  S.hosts.forwardFigure = h("div");
  S.hosts.ablationTable = h("div");
  const listError = S.side.verifications.error;
  return h(
    "div",
    null,
    sectionTitle("verify", "Verification", "Three measurements on the same data. Each is reported under its own name; there is no combined score and no pass threshold.", all),
    listError ? h("p", { class: "note", style: "margin-bottom:.6rem" }, `The recorded checks could not be read: ${listError.message}`) : null,
    h("div", { class: "grid-3" }, CHECKS.map((c) => S.hosts[c.id])),
    h("p", { class: "note", style: "margin-top:.6rem" }, "Term ablation and block dispersion are computed when a fit seals. The forward solve runs on demand as a background job, under the memory cap and queue of this machine; it can take minutes on large grids."),
    sectionTitle(null, "Forward simulation", "figure written by the check"),
    S.hosts.forwardFigure,
    sectionTitle(null, "Term ablation", "each term dropped once; the rest refitted"),
    S.hosts.ablationTable,
  );
}

/** The newest recorded result (or kd's refusal) of each check. */
function computeLatest() {
  S.latest = {};
  for (const c of CHECKS) {
    for (const v of S.verifications) {
      if (v[c.id] !== null && v[c.id] !== undefined) {
        S.latest[c.id] = { result: v[c.id] };
        break;
      }
      if (v.unavailable && v.unavailable[c.id]) {
        S.latest[c.id] = { unavailable: v.unavailable[c.id] };
        break;
      }
    }
  }
}

function renderVerification() {
  if (!S) return;
  computeLatest();
  for (const c of CHECKS) S.hosts[c.id].replaceChildren(frag(measureCard(c)));
  renderLedger();
  renderForwardFigure();
  renderAblationTable();
}

function measureCard(check) {
  const latest = S.latest[check.id];
  const job = S.jobs[check.id];
  const busy = job && !TERMINAL_STATES.has(job.state);
  const label = latest ? "Recompute" : "Compute";
  const action = button(label, { small: true, disabled: busy || !S.run.equation, onClick: () => startVerify([check.id]) });
  return [
    h("h3", null, check.title),
    ...measureBody(check.id, latest),
    job ? jobLine(job) : null,
    h("span", { class: "how" }, h("code", null, check.fn), action),
  ];
}

function measureBody(id, latest) {
  if (!latest) {
    const text = id === "forward_error" ? "Not computed yet. The forward solve runs on demand." : "Not recorded for this run.";
    return [h("span", { class: "v none" }, "not computed"), h("span", { class: "what" }, text)];
  }
  if (latest.unavailable) return [h("span", { class: "v none" }, "not available"), h("span", { class: "what" }, `kd cannot run this check here: ${latest.unavailable}`)];
  const r = latest.result;
  if (id === "forward_error") {
    if (r.value === null) return [h("span", { class: "v none" }, "did not finish"), h("span", { class: "what" }, r.reason || ""), h("span", { class: "what" }, r.definition)];
    return [h("span", { class: "v" }, fmtNumber(r.value)), h("span", { class: "what" }, r.definition.replace(/\.$/, ""), `. Took ${fmtDuration(r.seconds)}.`), warningBlock(r.warning ? [r.warning] : [])];
  }
  if (id === "ablation") {
    const nmse = r.rows.map((row) => row.nmse);
    const value = nmse.length ? (nmse.length === 1 ? fmtNumber(nmse[0]) : `${fmtNumber(Math.min(...nmse))} to ${fmtNumber(Math.max(...nmse))}`) : "no terms";
    return [h("span", { class: "v" }, value), h("span", { class: "what" }, `nmse after dropping one term and refitting the rest, against ${fmtNumber(r.baseline_nmse)} for the full equation. One row per term below.`)];
  }
  const along = r.axis ? ` along ${r.axis}` : "";
  return [h("span", { class: "v" }, fmtNumber(r.value)), h("span", { class: "what" }, `Spread of the coefficients refitted on ${r.n_blocks} separate blocks${along}.`)];
}

function jobLine(job) {
  const parts = [statusBadge(job.state, job.phase)];
  if (job.state === "queued" && job.waiting) parts.push(h("span", null, job.waiting.message));
  else if (job.state === "running" && job.started_at) parts.push(h("span", null, `started ${fmtTime(job.started_at)}`));
  else if (job.error) parts.push(h("span", null, `${job.error.error_type}: ${job.error.message}`));
  parts.push(h("a", { href: href("run", job.job_id) }, "job"));
  return h("div", { class: "r-job" }, parts, warningBlock(job.warnings));
}

function renderForwardFigure() {
  const latest = S.latest.forward_error;
  const id = latest && latest.result && latest.result.figure_artifact_id;
  S.hosts.forwardFigure.replaceChildren(
    id
      ? figure(artifactUrl(id), "forward_error", "Measured field, field integrated from the discovered equation, and their difference")
      : h("p", { class: "note" }, latest ? "The forward solve wrote no figure." : "The figure appears when the forward solve has run."),
  );
}

function renderAblationTable() {
  const latest = S.latest.ablation;
  if (!latest || !latest.result) {
    S.hosts.ablationTable.replaceChildren(h("p", { class: "note" }, latest ? `kd cannot run this check here: ${latest.unavailable}` : "No term ablation recorded for this run."));
    return;
  }
  const r = latest.result;
  const columns = [
    { label: "Dropped term", render: (row) => tex(row.dropped_latex) },
    { label: "nmse", num: true, render: (row) => fmtNumber(row.nmse) },
    { label: "MSE", num: true, render: (row) => fmtNumber(row.mse) },
    { label: "MSE rise", num: true, render: (row) => fmtNumber(row.delta_mse) },
    { label: "Relative rise", num: true, render: (row) => fmtNumber(row.relative_rise) },
  ];
  S.hosts.ablationTable.replaceChildren(frag(
    table(columns, r.rows, { empty: "The equation has no terms to drop." }),
    h("p", { class: "note", style: "margin-top:.5rem" }, `Full equation: nmse ${fmtNumber(r.baseline_nmse)}, MSE ${fmtNumber(r.baseline_mse)}. MSE rise is the MSE without the term minus the full equation's MSE; relative rise divides it by the full equation's MSE. kd sets no threshold on either.`),
  ));
}

async function startVerify(checks) {
  const run = S.run;
  const accepted = await post("/jobs", { kind: "verify", run_id: run.run_id, checks });
  if (!S || S.run !== run) return;
  for (const job of accepted.jobs) {
    for (const check of checks) S.jobs[check] = job;
    trackVerify(job.job_id, checks);
    for (const w of job.warnings) toast(w, "warn");
  }
  renderVerification();
  toast(checks.length === 1 ? "Check queued." : "Checks queued.");
}

/** Verify jobs on this run that were still open when the page loaded. */
function followOpenVerifications(jobs) {
  for (const job of jobs) {
    if (job.kind === "verify" && job.run_id === S.run.run_id && !TERMINAL_STATES.has(job.state)) trackVerify(job.job_id, null);
  }
}

function trackVerify(jobId, submitted) {
  if (S.followers.has(jobId)) return;
  const run = S.run;
  const handle = followJob(jobId, {
    interval: 2000,
    onJob: (job) => {
      if (!S || S.run !== run) return;
      const checks = (job.request && job.request.checks) || submitted || [];
      for (const check of checks) S.jobs[check] = job;
      if (TERMINAL_STATES.has(job.state)) verifyEnded(job);
      else renderVerification();
    },
    onError: (err) => toast(`Following the check failed: ${err.message}`, "warn"),
  });
  S.followers.set(jobId, handle);
}

async function verifyEnded(job) {
  const run = S.run;
  S.followers.delete(job.job_id);
  if (job.state === "completed" && job.result_available) {
    try {
      const result = await get(`/jobs/${enc(job.job_id)}/result`);
      if (!S || S.run !== run) return;
      S.verifications = [result, ...S.verifications];
      toast("Check finished.");
    } catch (err) {
      toast(`The check's result could not be read: ${err.message}`, "error");
    }
  }
  if (S && S.run === run) renderVerification();
}

// ---------------------------------------------------------------------------
// Fit: convergence and KD's figures

function figure(src, stem, alt) {
  const plain = stem.replace(/_/g, " ").replace(/^plugin /, "");
  const [title, text] = stem === "convergence" ? ["Convergence", `Best score per ${S.unit.singular}.`] : FIGURE_TEXT[stem] || [plain.charAt(0).toUpperCase() + plain.slice(1), ""];
  return h("figure", { class: "fig" }, h("img", { src, alt: alt || title, loading: "lazy" }), h("figcaption", null, h("b", null, text ? `${title}.` : title), text ? " " : null, text));
}

/** "report/coefficient_bar.svg" -> "coefficient_bar": kd's report writes its figures in report/. */
const stemOf = (name) => name.replace(/^.*\//, "").replace(/\.[^.]+$/, "");

function fitSection(run) {
  const verifyFigures = new Set(S.verifications.map((v) => v.forward_error && v.forward_error.figure_artifact_id).filter(Boolean));
  const figures = S.side.files.filter((a) => a.kind === "figure" && !verifyFigures.has(a.artifact_id));
  return h(
    "div",
    null,
    sectionTitle("fit", "Fit", "convergence and the figures of kd's report"),
    convergencePanel(run),
    figures.length
      ? h("div", { class: "figrow" }, figures.map((a) => figure(artifactUrl(a.artifact_id), stemOf(a.name))))
      : h("p", { class: "note", style: "margin-top:.8rem" }, S.side.filesError ? `The run's files could not be listed: ${S.side.filesError.message}` : "No figures were written for this run."),
  );
}

function convergencePanel(run) {
  const c = run.convergence;
  if (!c) return panel({ title: "Convergence" }, h("p", { class: "note" }, "The run has no recorded convergence."));
  const better = c.score_direction === "min" ? "lower is better" : "higher is better";
  const unit = S.unit;
  const meta = `best ${c.score_kind} per ${unit.singular} (${better})`;
  if (c.iterations.length < 2) {
    return panel({ title: "Convergence", meta }, h("p", { class: "note" }, `One ${unit.singular}: best ${c.score_kind} ${fmtNumber(c.best_score[0])}.`));
  }
  const el = h("div", { class: "plot" });
  const finite = c.best_score.filter((v) => v !== null);
  const logY = finite.length > 0 && finite.every((v) => v > 0) && Math.max(...finite) / Math.min(...finite) > 100;
  requestAnimationFrame(() => {
    if (el.isConnected) linePlot(el, { series: [{ name: c.score_kind, x: c.iterations, y: c.best_score, step: true }], xLabel: unit.singular, yLabel: `best ${c.score_kind}`, logY });
  });
  return panel({ title: "Convergence", meta }, el);
}

// ---------------------------------------------------------------------------
// Candidates and sketch

function candidatesSection(run) {
  S.hosts.candidates = h("div");
  if (run.candidates) drawCandidates(run.candidates);
  else S.hosts.candidates.append(h("p", { class: "note" }, `${run.algorithm || "This run"} kept no candidate list. A one-shot fit (pysindy) has none; searches list the best equations of each generation or iteration, and pysr its Pareto front.`));
  return h("div", null, sectionTitle("candidates", "Candidates", "what the algorithm considered"), panel({}, S.hosts.candidates), sketchPanel(run));
}

function drawCandidates(list) {
  const better = list.score_direction === "min" ? "lower is better" : "higher is better";
  const columns = [
    { label: "Rank", num: true, key: "rank" },
    { label: "Equation", render: (c) => tex(c.equation_latex) },
    { label: `${list.score_kind} (${better})`, num: true, render: (c) => fmtNumber(c.score) },
    { label: "Complexity", num: true, render: (c) => c.complexity ?? "n/a" },
  ];
  const parts = [];
  if (list.form === "top_k") {
    const unit = S.unit.singular;
    const select = h("select", { "aria-label": unit[0].toUpperCase() + unit.slice(1), onChange: (e) => loadCandidates(Number(e.target.value)) }, list.iterations.map((i) => h("option", { value: i, selected: i === list.iteration }, `${unit} ${i}`)));
    parts.push(h("div", { class: "r-row", style: "margin-bottom:.6rem" }, h("span", { class: "note" }, `The best equations of one ${unit} (${list.iterations.length} recorded).`), select));
  } else {
    const el = h("div", { class: "plot short" });
    requestAnimationFrame(() => {
      if (el.isConnected) scatterPlot(el, { series: [{ name: "Pareto front", x: list.rows.map((c) => c.complexity), y: list.rows.map((c) => c.score), labels: list.rows.map((c) => c.equation_latex) }], xLabel: "complexity", yLabel: list.score_kind });
    });
    parts.push(h("p", { class: "note" }, "The Pareto front of the search: the best score at each complexity."), el);
  }
  parts.push(table(columns, list.rows, { empty: `No candidates in this ${S.unit.singular}.` }));
  S.hosts.candidates.replaceChildren(frag(parts));
}

async function loadCandidates(iteration) {
  const run = S.run;
  try {
    const list = await get(`/runs/${enc(run.run_id)}/candidates`, { iteration });
    if (S && S.run === run) drawCandidates(list);
  } catch (err) {
    toast(`Candidates of ${S.unit.singular} ${iteration} could not be read: ${err.message}`, "error");
  }
}

function sketchPanel(run) {
  const o = run.sketch_outcome;
  if (!o) {
    const meta = run.prior_knowledge ? (run.prior_knowledge.sketch ? "a sketch was set; the record holds no outcome" : "none set for this run") : "not known for an imported run";
    return h("section", { class: "panel", style: "margin-top:1rem" }, h("header", null, h("h3", null, "Sketch"), h("span", { class: "meta" }, meta)));
  }
  const columns = [
    { label: "Clause", render: (c) => c.clause },
    { label: "Enforced as", render: (c) => c.level },
    { label: "Satisfied", render: (c) => (c.satisfied === null ? "not decided" : c.satisfied ? tag("yes", "ok") : tag("no", "no")) },
    { label: "Detail", render: (c) => c.detail || "" },
  ];
  return h(
    "section",
    { class: "panel", style: "margin-top:1rem" },
    h("header", null, h("h3", null, "Sketch"), h("span", { class: "meta" }, "kd's verdict on each clause")),
    kv([
      ["Best solution that satisfies the sketch", o.solution ? tex(o.solution.latex) : "none"],
      ["Best candidate overall", o.best_candidate ? tex(o.best_candidate.latex) : "none"],
    ]),
    o.failure ? h("div", { class: "warning", style: "margin-top:.6rem" }, o.failure) : null,
    h("div", { style: "margin-top:.7rem" }, table(columns, o.clauses, { empty: "The sketch has no clauses." })),
  );
}

// ---------------------------------------------------------------------------
// Report

function reportSection(run) {
  const report = S.side.files.find((a) => a.kind === "report");
  if (!report) {
    return h("div", null, sectionTitle("report", "Report"), h("p", { class: "note" }, noReportText(run)));
  }
  const url = artifactUrl(report.artifact_id);
  const actions = [
    h("a", { class: "btn small", href: url, target: "_blank", rel: "noopener" }, "Open in a new tab"),
    h("a", { class: "btn small", href: url, download: report.name }, `Download HTML (${fmtBytes(report.size_bytes)})`),
  ];
  return h(
    "div",
    null,
    sectionTitle("report", "Report", null, actions),
    h("iframe", { class: "r-report", src: url, sandbox: "allow-scripts", title: "KD report for this run" }),
    h("p", { class: "note", style: "margin-top:.5rem" }, "Written by Model.report when the run sealed, in the run's own process, and shown as kd wrote it."),
  );
}

function noReportText(run) {
  if (S.side.filesError) return `The run's files could not be listed: ${S.side.filesError.message}`;
  if (run.status === "raised") return "This run has no report: it raised an error before its seal, and kd writes the report after the seal.";
  if (run.status === "unsealed") return "This run has no report yet: kd writes the report after the seal.";
  if (run.origin === "imported") return "This run directory holds no report. Runs started outside the UI have one only when Model.report was called.";
  return "This run has no report file.";
}

// ---------------------------------------------------------------------------
// Record

function recordSection(run) {
  const rerun = button("Edit and rerun", { small: true, disabled: !(run.request || run.config), onClick: () => go("search", null, { clone: run.run_id }) });
  return h(
    "div",
    null,
    sectionTitle("record", "Record", "read from the run folder"),
    h(
      "div",
      { class: "grid-2" },
      h("section", { class: "panel" }, h("header", null, h("h3", null, "Run and cost")), kv(costRows(run))),
      h("section", { class: "panel" }, h("header", null, h("h3", null, "Configuration"), rerun), configList(run.config)),
    ),
    h("section", { class: "panel", style: "margin-top:1rem" }, h("header", null, h("h3", null, "Provenance")), kv(provenanceRows(run))),
    filesPanel(),
  );
}

function costRows(run) {
  const c = run.cost || {};
  const sec = (v) => (v === null || v === undefined ? null : fmtDuration(v));
  const split = [c.search_seconds != null && `search ${sec(c.search_seconds)}`, c.preprocessing_seconds != null && `preprocessing ${sec(c.preprocessing_seconds)}`, c.surrogate_train_seconds != null && `surrogate training ${sec(c.surrogate_train_seconds)}`].filter(Boolean);
  return [
    ["Run id", mono(run.run_id)],
    run.kd_run_id && ["kd run id", mono(run.kd_run_id)],
    ["Status", `${run.status}${endReason(run) ? `, ${stopReasonText(endReason(run))}` : ""}${run.stop_reason ? ` (kd: ${run.stop_reason})` : ""}`],
    run.created_at && ["Started", fmtTime(run.created_at)],
    c.wallclock_seconds != null && ["Wall clock", `${sec(c.wallclock_seconds)}${split.length ? ` (${split.join(", ")})` : ""}`],
    c.cpu_seconds != null && ["CPU time", sec(c.cpu_seconds)],
    c.peak_rss_bytes != null && ["Peak memory", fmtBytes(c.peak_rss_bytes)],
    c.device && ["Device", c.device],
    (c.iterations != null || c.max_iterations != null) && ["Iterations run / budget", `${c.iterations ?? "n/a"} / ${c.max_iterations ?? "n/a"}`],
    (c.tokens_in != null || c.tokens_out != null) && ["LLM tokens", `${c.tokens_in ?? "n/a"} in, ${c.tokens_out ?? "n/a"} out`],
    ["Run folder", mono(run.provenance.run_dir)],
    !run.cost && ["Cost", "not in the record"],
  ];
}

const showValue = (v) => (v === null ? "none" : Array.isArray(v) ? v.map((x) => (typeof x === "object" ? JSON.stringify(x) : String(x))).join(", ") : typeof v === "object" ? JSON.stringify(v) : String(v));

function configList(config) {
  if (!config) return h("p", { class: "note" }, "The record holds no resolved configuration.");
  return h(
    "dl",
    { class: "kv" },
    Object.entries(config).map(([k, v]) => [h("dt", { class: "mono", style: "font-size:.82rem" }, k), h("dd", null, showValue(v))]),
  );
}

function provenanceRows(run) {
  const p = run.provenance;
  const env = p.environment ? Object.entries(p.environment).map(([k, v]) => `${k} ${v}`).join(", ") : null;
  const smoothing = run.preprocessing ? smoothingText(run.preprocessing.smoothing, run.savgol_window) ?? "none" : null;
  return [
    ["KD version", p.kd_version || "n/a"],
    run.dataset_name && ["Dataset", run.dataset_id ? h("a", { href: href("data", run.dataset_id) }, run.dataset_name) : run.dataset_name],
    run.dataset_fingerprint && ["Data fingerprint", mono(run.dataset_fingerprint)],
    run.loaded_fingerprint && run.loaded_fingerprint !== run.dataset_fingerprint && ["Fingerprint as loaded", mono(run.loaded_fingerprint)],
    smoothing && ["Smoothing", smoothing],
    ["Configuration hash", p.config_hash ? mono(p.config_hash) : "n/a"],
    ["Record hash", p.record_hash ? mono(p.record_hash) : "n/a"],
    ["Machine", p.hostname || "n/a"],
    env && ["Environment", env],
    ["Origin", run.origin === "ui" ? "started in kd-ui" : `imported from history source ${run.source_id}`],
    ["Prior", run.prior_knowledge ? (run.prior_knowledge.sketch ? "a sketch was supplied" : "none") : "not known for an imported run"],
    run.lineage && ["Lineage", lineageText(run.lineage)],
  ];
}

function filesPanel() {
  const columns = [
    { label: "File", render: (a) => h("a", { href: artifactUrl(a.artifact_id), target: "_blank", rel: "noopener" }, a.name) },
    { label: "Kind", key: "kind" },
    { label: "Size", num: true, render: (a) => fmtBytes(a.size_bytes) },
  ];
  return h("section", { class: "panel", style: "margin-top:1rem" }, h("header", null, h("h3", null, "Files")), S.side.filesError ? h("p", { class: "note", style: "margin-bottom:.5rem" }, `A file listing could not be read: ${S.side.filesError.message}`) : null, table(columns, S.side.files, { empty: "No files listed." }));
}

// ---------------------------------------------------------------------------
// Resume

function toggleResume() {
  const host = S.hosts.resume;
  if (!host.hidden) {
    host.hidden = true;
    return;
  }
  host.hidden = false;
  host.replaceChildren(resumePanel(S.side.checkpoints.value, S.side.algo, S.side.caps.error));
}

function resumePanel(ck, algo, capsError) {
  const run = S.run;
  const unit = ck.unit || "iterations";
  const pick = h("select", { "aria-label": "Checkpoint" }, ck.checkpoints.map((c, i) => h("option", { value: c.iteration, selected: i === ck.checkpoints.length - 1 }, `after ${c.iteration} ${unit}${c.size_bytes !== null ? ` (${fmtBytes(c.size_bytes)})` : ""}`)));
  const budget = algo && algo.facade_params.find((f) => f.effect === "max_iterations");
  const budgetInput = budget && h("input", { type: "text", class: "num", "aria-label": budget.name, value: (run.request && run.request.params[budget.name]) ?? budget.default });
  const reseed = h("input", { type: "checkbox" });
  const seed = h("input", { type: "text", class: "num", "aria-label": "New seed", placeholder: "seed" });
  const seconds = h("input", { type: "text", class: "num", "aria-label": "Time budget in seconds", placeholder: "none" });
  const hardSeconds = h("input", { type: "text", class: "num", "aria-label": "Hard time limit in seconds", placeholder: "auto" });
  const fields = algo ? algo.fields : [];
  const form = algo ? schemaForm(fields, { values: run.config || {}, lockNotResumeSafe: true, resumeColumn: false }) : null;
  const submit = button("Resume", {
    kind: "primary",
    onClick: () => submitResume({ pick, budget, budgetInput, reseed, seed, seconds, hardSeconds, form, fields }),
  });
  return panel(
    { title: "Resume from a checkpoint", meta: `a new run continues this one; ${unit} count from zero in it`, actions: [button("Close", { small: true, onClick: () => { S.hosts.resume.hidden = true; } })] },
    capsError ? warningBlock([`The parameter schema could not be read (${capsError.message}); only the checkpoint and the seed can be set.`]) : null,
    algo ? null : !capsError && warningBlock([`No parameter schema for ${run.algorithm}; only the checkpoint and the seed can be set.`]),
    kv([
      ["Checkpoint", pick],
      budget && [`${unit} in the new run (${budget.name})`, budgetInput],
      (!algo || algo.segmentation.reseed) && ["New seed", h("span", { class: "r-inline" }, h("label", { class: "check" }, reseed, "branch with a new seed"), seed)],
      ["Time budget (s)", seconds],
      ["Hard time limit (s)", h("span", { class: "r-inline" }, hardSeconds, h("span", { class: "note" }, "the run is ended when this has passed; empty: the time budget plus 2 min, or none"))],
    ]),
    form ? h("details", { style: "margin-top:.8rem" }, h("summary", null, `Parameters (only those safe to change on resume are editable: ${ck.resume_safe_params.join(", ") || "none"})`), form.element) : null,
    h("div", { class: "actions", style: "margin-top:.8rem" }, submit),
  );
}

/** Resume-safe parameters whose value differs from the run's configuration. */
function resumeChanges(form, fields) {
  if (!form) return {};
  const values = form.values();
  const config = S.run.config || {};
  const changes = {};
  for (const f of fields) {
    const key = f.facade_param || f.name;
    if (f.resume_tier !== "resume_safe" || !(key in values)) continue;
    if (JSON.stringify(values[key]) !== JSON.stringify(config[f.name])) changes[key] = values[key];
  }
  return changes;
}

async function submitResume({ pick, budget, budgetInput, reseed, seed, seconds, hardSeconds, form, fields }) {
  const errors = form ? form.errors() : [];
  if (errors.length) throw new Error(`${errors[0].key}: ${errors[0].message}`);
  const params = resumeChanges(form, fields);
  if (budget && budgetInput.value.trim() !== "") params[budget.name] = Number(budgetInput.value);
  const body = { iteration: Number(pick.value), params, reseed: reseed.checked };
  if (reseed.checked && seed.value.trim() !== "") body.seed = Number(seed.value);
  if (seconds.value.trim() !== "") body.max_seconds = Number(seconds.value);
  if (hardSeconds.value.trim() !== "") body.hard_seconds = Number(hardSeconds.value);
  const accepted = await post(`/runs/${enc(S.run.run_id)}/resume`, body);
  const job = accepted.jobs[0];
  for (const w of job.warnings) toast(w, "warn");
  toast("Resume queued.");
  go("run", job.job_id);
}
