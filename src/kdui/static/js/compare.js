// Compare page: the history of runs (the UI's own and those read
// from registered run folders), the selection of several runs on one dataset,
// and the comparison view of a `compare` job: each run's native score beside
// the nmse KD re-measures in one shared frame, the equations, each run's own
// checks, KD's comparison figure.
//
// Addresses: #/compare (history), #/compare?runs=a,b (those runs selected,
// not yet re-measured), #/compare?job=<job_id> (the result of a compare job).
// The browser computes no scientific quantity: it shows what the API returns.

import { get, post, artifactUrl, followJob, TERMINAL_STATES } from "./api.js";
import {
  h, clear, href, go, replaceRoute, fmtNumber, fmtDuration, fmtTime, button, linkButton, tag, pageHead,
  panel, kv, warningBlock, reserved, loadingState, emptyState, errorState, toast, statusBadge, stopReasonText, smoothingText, equation,
} from "./ui.js";
import { dotPlot, linePlot } from "./plots.js";

const RUN_STATUS = {
  completed: ["completed", "Completed"],
  raised: ["failed", "Raised an error"],
  unsealed: ["queued", "No seal yet"],
  no_record: ["cancelled", "No record"],
};

let poller = null;
let page = null; // per-mount state

export async function mount(container, params) {
  page = {
    container,
    runs: [],
    listWarnings: [],
    nextCursor: null,
    sources: [],
    selected: new Set((params.runs || "").split(",").filter(Boolean)),
    filters: { status: "all", algorithm: "all", dataset: "all", origin: "all" },
    sortKey: "nmse",
  };
  const compareEl = h("div");
  const historyEl = h("div", { id: "cmp-history" });
  page.compareEl = compareEl;
  page.historyEl = historyEl;
  container.append(compareEl, historyEl);
  // Reading the runs is one short kd call (a few seconds); say so meanwhile.
  historyEl.append(loadingState("Reading the runs"));
  await loadHistory();
  if (!page || page.container !== container) return;
  defaultDatasetFilter();
  renderHistory();
  if (params.job) followCompareJob(params.job);
  else renderPending();
}

export function unmount() {
  if (poller) poller.stop();
  poller = null;
  page = null;
}

// ---------------------------------------------------------------------------
// Data

async function loadHistory() {
  const [list, sources] = await Promise.all([get("/runs", { limit: 200 }), get("/history/sources")]);
  page.runs = list.runs;
  page.nextCursor = list.next_cursor;
  page.listWarnings = list.warnings;
  page.sources = sources.sources;
}

async function loadMore() {
  const list = await get("/runs", { limit: 200, cursor: page.nextCursor });
  page.runs = page.runs.concat(list.runs);
  page.nextCursor = list.next_cursor;
  renderHistory();
}

const runById = (id) => page.runs.find((r) => r.run_id === id) || null;
const groupKey = (r) => `${r.dataset_name}\u0000${r.loaded_fingerprint}`;
const tail = (fp) => (fp ? `…${fp.slice(-8)}` : "n/a");

/** The group (dataset name and fingerprint as loaded) of the current selection. */
function selectionGroup() {
  for (const id of page.selected) {
    const r = runById(id);
    if (r && r.loaded_fingerprint) return r;
  }
  return null;
}

/** Why run `r` cannot join the selection, or null when it can. */
function blockReason(r) {
  if (!r.loaded_fingerprint) return r.status === "unsealed" ? "No record of the data yet: the run has not sealed." : "No record of the data as loaded: the run ended without writing one.";
  const g = selectionGroup();
  if (!g || page.selected.has(r.run_id) || groupKey(g) === groupKey(r)) return null;
  if (r.dataset_name !== g.dataset_name) {
    return `On ${r.dataset_name ?? "unnamed data"}; the selection is on ${g.dataset_name}. Runs on different datasets cannot be compared: KD re-measures every equation on one dataset's data.`;
  }
  return `Also ${r.dataset_name}, but the data as loaded differs (fingerprint ${tail(r.loaded_fingerprint)}, the selection has ${tail(g.loaded_fingerprint)}).`;
}

function preprocessingLabel(pre, savgolWindow) {
  if (pre === null || pre === undefined) return "preprocessing not recorded (run started outside the UI)";
  const s = smoothingText(pre.smoothing, savgolWindow);
  return s ? `smoothed copy (${s})` : "data as loaded";
}

function originText(r) {
  if (r.origin === "ui") return "from this page";
  const src = page.sources.find((s) => s.source_id === r.source_id);
  return src ? `read from ${src.path}` : "started outside the UI";
}

// ---------------------------------------------------------------------------
// Compare job: submit, follow, show

async function submitCompare(runIds) {
  const accepted = await post("/jobs", { kind: "compare", run_ids: runIds });
  const job = accepted.jobs[0];
  toast(`Comparison of ${runIds.length} runs queued.`);
  go("compare", null, { job: job.job_id });
}

function renderPending() {
  const ids = [...page.selected];
  const el = clear(page.compareEl);
  if (ids.length < 2) {
    el.append(pageHead({ title: "Runs and comparison", sub: "Every run of this workspace and of the run folders you added. Select two or more runs on one dataset to compare them." }));
    return;
  }
  const g = selectionGroup();
  el.append(
    pageHead({ title: `Compare ${ids.length} runs${g ? ` on ${g.dataset_name}` : ""}`, sub: "Not yet re-measured in the shared frame." }),
    panel(
      { title: "Shared frame" },
      h("p", { class: "note" }, "KD re-measures every run's equation on the same data as loaded and reports one nmse per run. The runs' own coefficients and native scores are not changed."),
      h("div", { class: "actions", style: "margin-top:.6rem" }, button("Re-measure in the shared frame", { kind: "primary", onClick: () => submitCompare(ids) })),
    ),
    chips(ids),
  );
}

function followCompareJob(jobId) {
  const el = clear(page.compareEl);
  const status = h("div", null, loadingState("Reading the comparison job"));
  el.append(pageHead({ title: "Compare runs", sub: h("span", { class: "mono" }, jobId) }), status);
  let first = null;
  poller = followJob(jobId, {
    onJob: (job) => {
      if (!page) return;
      if (!first) {
        first = job;
        page.selected = new Set(job.request.run_ids);
        renderHistory();
      }
      if (job.state === "completed" && job.result_available) {
        showResult(job).catch((err) => page && clear(status).append(errorState(err, { title: "The comparison result could not be read." })));
      }
      else if (TERMINAL_STATES.has(job.state)) showJobEnded(job, status);
      else showJobWaiting(job, status);
    },
    onError: (err) => {
      if (!page) return;
      poller.stop();
      clear(status).append(errorState(err, { title: "The comparison could not be read.", retry: () => go("compare", null, { job: jobId }) }));
    },
  });
}

function showJobWaiting(job, status) {
  const what = job.state === "queued" ? (job.waiting ? job.waiting.message : "Waiting for the scheduler.") : `KD is re-measuring ${job.request.run_ids.length} runs in one frame.`;
  clear(status).append(
    panel({ title: "Re-measuring", meta: h("span", null, statusBadge(job.state, job.phase)) }, h("p", null, what), warningBlock(job.warnings)),
    chips(job.request.run_ids),
  );
}

function showJobEnded(job, status) {
  const message = job.error ? `${job.error.error_type}: ${job.error.message}` : `The comparison ended: ${stopReasonText(job.stop_reason)}.`;
  clear(status).append(
    errorState({ message }, { title: "The comparison did not finish.", retry: () => submitCompare(job.request.run_ids) }),
    chips(job.request.run_ids),
  );
  renderHistory();
}

// A run directory that is gone, or a run with no verification list (imported), answers 404.
const or404 = (err) => {
  if (err.status === 404) return null;
  throw err;
};

async function showResult(job) {
  const result = await get(`/jobs/${encodeURIComponent(job.job_id)}/result`);
  const ids = result.rows.map((r) => r.run_id);
  const [details, verifications] = await Promise.all([
    Promise.all(ids.map((id) => get(`/runs/${encodeURIComponent(id)}`).catch(or404))),
    Promise.all(ids.map((id) => get(`/runs/${encodeURIComponent(id)}/verifications`).catch(or404))),
  ]);
  if (!page) return;
  page.result = result;
  page.details = Object.fromEntries(ids.map((id, i) => [id, details[i]]));
  page.verifications = Object.fromEntries(ids.map((id, i) => [id, verifications[i] ? verifications[i].verifications : null]));
  page.jobId = job.job_id;
  replaceRoute("compare", null, { job: job.job_id });
  renderResult();
  renderHistory();
}

// ---------------------------------------------------------------------------
// Comparison view

function rowLabels(rows) {
  const base = (r) => `${r.algorithm}, seed ${r.seed ?? "n/a"}`;
  const counts = {};
  rows.forEach((r) => (counts[base(r)] = (counts[base(r)] || 0) + 1));
  const labels = {};
  for (const r of rows) {
    const d = page.details[r.run_id];
    const id = (d && d.kd_run_id) || r.run_id;
    labels[r.run_id] = counts[base(r)] > 1 ? `${base(r)} (${id.slice(-6)})` : base(r);
  }
  return labels;
}

function renderResult() {
  const { result } = page;
  const first = page.details[result.rows[0].run_id] || runById(result.rows[0].run_id);
  const dataset = first ? first.dataset_name : "one dataset";
  page.labels = rowLabels(result.rows);
  const el = clear(page.compareEl);
  el.append(
    pageHead({
      title: `Compare runs on ${dataset}`,
      sub: h("span", null, `${result.rows.length} runs on ${dataset} as loaded, fingerprint `, h("span", { class: "mono", title: result.dataset_fingerprint }, tail(result.dataset_fingerprint)), "."),
      actions: [
        button("Add runs", { onClick: () => page.historyEl.scrollIntoView({ behavior: "smooth" }) }),
        button("Download table (CSV)", { onClick: downloadCsv }),
        result.figure_artifact_id ? h("a", { class: "btn", href: artifactUrl(result.figure_artifact_id), download: "comparison" }, "Export figure") : null,
      ],
    }),
    framePanel(result, dataset),
    chips(result.rows.map((r) => r.run_id)),
    h("div", { class: "section-title" }, h("h2", null, "Runs"), h("span", { class: "meta" }, "sorted by same-frame nmse; click a column name to sort by it")),
    h("div", { id: "cmp-table" }),
    h("p", { class: "note", style: "margin-top:.5rem" }, "Native scores are shown under each algorithm's own name and are not ranked against each other. The nmse column is computed the same way for every run. The last two columns are each run's own checks, not part of the shared frame. KD reports no combined score."),
    plotsBlock(),
    h("div", null, diffTables()),
    h("div", null, figureBlock(result)),
    h("div", { class: "section-title" }, h("h2", null, "Consensus")),
    reserved("Consensus across runs: reserved for version 2", "Terms that several runs agree on, from kd.build_consensus over the runs' evidence stores."),
  );
  renderTable();
  drawPlots();
}

function framePanel(result, dataset) {
  const smoothed = result.rows.filter((r) => r.preprocessing && smoothingText(r.preprocessing.smoothing, r.savgol_window));
  return panel(
    { title: "Shared frame" },
    kv([
      ["Data", `${dataset} as loaded`],
      ["Fingerprint", h("span", { class: "mono", style: "font-size:.82rem" }, result.dataset_fingerprint)],
      ["Frame", result.frame],
      smoothed.length ? ["Smoothed copies", `${smoothed.length} run${smoothed.length > 1 ? "s" : ""} ran on a smoothed copy; their equations are refitted on the data as loaded`] : null,
    ]),
    h("p", { class: "note", style: "margin-top:.6rem" }, "KD refits every run's terms in this one frame. A run's own coefficients and its native score stay in their own columns and are not changed. Runs on a smoothed copy of this data can be added and are labelled with their preprocessing; runs on other datasets cannot."),
  );
}

function chips(ids) {
  return h(
    "div",
    { class: "cmp-chips", "aria-label": "Runs in this comparison" },
    ids.map((id) => {
      const r = runById(id);
      const label = page.labels && page.labels[id] ? page.labels[id] : r ? `${r.algorithm}, seed ${r.seed ?? "n/a"}` : id;
      const rest = ids.filter((x) => x !== id);
      const remove = () => (rest.length >= 2 ? submitCompare(rest) : go("compare", null, { runs: rest.join(",") }));
      return h("span", { class: "cmp-chip", title: id }, label, h("button", { type: "button", "aria-label": `Remove ${label} from the comparison`, onClick: remove }, "×"));
    }),
  );
}

const SORTS = {
  nmse: (r) => r.nmse,
  complexity: (r) => r.complexity,
  time: (r) => wallclock(r.run_id),
  native: (r) => (r.native_score ? r.native_score.value : null),
};

function sortedRows() {
  const key = SORTS[page.sortKey];
  const rows = [...page.result.rows];
  rows.sort((a, b) => {
    const [x, y] = [key(a), key(b)];
    if (x === null || x === undefined) return y === null || y === undefined ? 0 : 1;
    if (y === null || y === undefined) return -1;
    return x - y;
  });
  return rows;
}

const wallclock = (id) => {
  const d = page.details[id];
  return d && d.cost ? d.cost.wallclock_seconds : null;
};

/** Native scores rank runs only within one algorithm: two algorithms may give
 *  the same name to scores measured differently. Across algorithms the shared
 *  re-measured nmse ranks; the native column stays as information. */
function nativeComparable() {
  const rows = page.result.rows;
  const kinds = new Set(rows.map((r) => (r.native_score ? `${r.algorithm}|${r.native_score.kind}|${r.native_score.direction}` : null)));
  return kinds.size === 1 && !kinds.has(null) && rows[0].native_score.direction === "min";
}

function sortHeader(label, key, cls = "num") {
  const allowed = key !== "native" || nativeComparable();
  const title = allowed ? `Sort by ${label}` : "Native scores rank runs of one algorithm only; runs of different algorithms are ranked by the shared nmse.";
  const onClick = allowed ? () => { page.sortKey = key; renderTable(); } : null;
  const mark = page.sortKey === key ? " ↑" : "";
  return h("th", { class: `${cls}${allowed ? " sortable" : ""}`, title, "aria-sort": page.sortKey === key ? "ascending" : null, onClick }, label + mark);
}

function renderTable() {
  const rows = sortedRows();
  const firstTerms = rows[0] && rows[0].equation ? termKey(rows[0].equation) : null;
  const thead = h(
    "thead",
    null,
    h("tr", null, h("th", { class: "grp", colspan: 3 }, "As the run reported it"), h("th", { class: "grp shared", colspan: 3 }, "Re-measured by KD in the shared frame"), h("th", { class: "grp own", colspan: 2 }, "Checks of each run")),
    h(
      "tr",
      null,
      h("th", null, "Run and equation"),
      sortHeader("Native score", "native", ""),
      sortHeader("Time", "time"),
      sortHeader("nmse", "nmse", "num sh-first"),
      sortHeader("Terms", "complexity", ""),
      h("th", null, "Error"),
      h("th", { class: "num own-first" }, "Forward error"),
      h("th", { class: "num" }, "Block dispersion"),
    ),
  );
  const tbody = h("tbody", null, rows.map((r, i) => compareRow(r, i, firstTerms)));
  clear(page.compareEl.querySelector("#cmp-table")).append(h("div", { class: "tablebox" }, h("table", { class: "t cmp" }, thead, tbody)));
}

const termKey = (eq) => eq.terms.map((t) => t.latex).sort().join("\u0000");

function compareRow(r, i, firstTerms) {
  const d = page.details[r.run_id];
  const summary = d || runById(r.run_id);
  const sub = [
    (d && d.kd_run_id) || r.run_id,
    summary ? originText(summary) : null,
    d && d.provenance.kd_version ? `kd ${d.provenance.kd_version}` : null,
    preprocessingLabel(r.preprocessing, r.savgol_window),
  ].filter(Boolean);
  const terms = !r.equation
    ? h("span", { class: "nm" }, "n/a")
    : i > 0 && termKey(r.equation) === firstTerms
      ? `same as row 1 (${r.equation.terms.length})`
      : h("span", null, `${r.complexity ?? r.equation.terms.length}: `, r.equation.terms.map((t, k) => [k ? ", " : null, equation(t.latex, { size: "inline" })]));
  return h(
    "tr",
    null,
    h(
      "td",
      null,
      h("a", { class: "who", href: href("result", r.run_id) }, page.labels[r.run_id]),
      r.prior_knowledge && r.prior_knowledge.sketch ? [" ", tag("sketch", "prior")] : null,
      h("span", { class: "sub2" }, h("span", { class: "mono" }, sub[0]), sub.length > 1 ? `, ${sub.slice(1).join(", ")}` : ""),
      r.equation ? h("div", { class: "eq" }, equation(r.equation.latex, { size: "inline" })) : h("div", { class: "nm" }, "no equation"),
    ),
    h("td", null, r.native_score ? [h("span", null, fmtNumber(r.native_score.value)), h("span", { class: "sub2" }, `${r.native_score.kind}, ${r.native_score.direction === "min" ? "lower" : "higher"} is better`)] : h("span", { class: "nm" }, "no score")),
    h("td", { class: "num" }, wallclock(r.run_id) === null ? h("span", { class: "nm" }, "n/a") : fmtDuration(wallclock(r.run_id))),
    h("td", { class: "num sh-first" }, r.nmse === null ? h("span", { class: "nm" }, "not measured") : fmtNumber(r.nmse)),
    h("td", null, terms),
    h("td", null, r.error ? h("span", { class: "rowerr" }, r.error) : ""),
    h("td", { class: "num own-first" }, forwardCell(r.run_id)),
    h("td", { class: "num" }, dispersionCell(r.run_id)),
  );
}

function latestCheck(id, check) {
  const list = page.verifications[id];
  if (!list) return undefined;
  const found = list.find((v) => v[check] !== null);
  return found ? found[check] : null;
}

function forwardCell(id) {
  const fe = latestCheck(id, "forward_error");
  if (fe === undefined) return h("span", { class: "nm" }, "n/a");
  if (fe && fe.value !== null) return h("span", { title: fe.definition }, fmtNumber(fe.value));
  const cell = h("div", null, h("span", { class: "nm" }, fe ? `not finished: ${fe.reason}` : "not measured"));
  cell.append(h("br"), button("Measure", { small: true, onClick: () => measure(id, cell) }));
  return cell;
}

async function measure(runId, cell) {
  const accepted = await post("/jobs", { kind: "verify", run_id: runId, checks: ["forward_error"] });
  const job = accepted.jobs[0];
  toast("Forward solve queued.");
  clear(cell).append(h("a", { href: href("run", job.job_id) }, "queued"));
  const warn = warningBlock(job.warnings);
  if (warn) cell.append(warn);
}

function dispersionCell(id) {
  const bd = latestCheck(id, "block_dispersion");
  if (bd === undefined || bd === null) return h("span", { class: "nm" }, "n/a");
  return h("span", { title: `${bd.n_blocks} blocks${bd.axis ? ` along ${bd.axis}` : ""}` }, fmtNumber(bd.value));
}

// ---------------------------------------------------------------------------
// Plots: same-frame nmse, and best native score by iteration per score kind

function convergenceGroups() {
  const groups = new Map();
  for (const r of page.result.rows) {
    const d = page.details[r.run_id];
    const c = d && d.convergence;
    if (!c || c.iterations.length < 2) continue;
    const key = `${c.score_kind}|${c.score_direction}`;
    if (!groups.has(key)) groups.set(key, { kind: c.score_kind, direction: c.score_direction, runs: [] });
    groups.get(key).runs.push({ id: r.run_id, c });
  }
  return [...groups.values()];
}

function plotsBlock() {
  const groups = convergenceGroups();
  const without = page.result.rows.filter((r) => !groups.some((g) => g.runs.some((x) => x.id === r.run_id)));
  const withoutText = without.length
    ? `No curve for ${without.map((r) => `${page.labels[r.run_id]} (${curveMissing(r.run_id)})`).join(", ")}.`
    : null;
  const convPanels = groups.map((g, i) =>
    panel(
      { title: `Best ${g.kind} by iteration, ${[...new Set(g.runs.map((x) => page.result.rows.find((r) => r.run_id === x.id).algorithm))].join(", ")} runs`, meta: g.direction === "min" ? "lower is better" : "higher is better" },
      h("div", { class: "plot", "data-conv": i }),
      h("p", { class: "note", style: "margin-top:.4rem" }, "Only runs with the same native score share a chart.", withoutText && i === groups.length - 1 ? ` ${withoutText}` : ""),
    ),
  );
  if (!groups.length) convPanels.push(panel({ title: "Best native score by iteration" }, emptyState(withoutText || "No run in this comparison recorded a curve.")));
  return h(
    "div",
    { class: "cmp-pair" },
    panel({ title: "Same-frame nmse", meta: "log scale; lower is a closer fit" }, h("div", { id: "cmp-dots" })),
    convPanels,
  );
}

function curveMissing(id) {
  const d = page.details[id];
  if (!d) return "run not readable";
  if (d.convergence) return "one iteration only";
  return "no convergence recorded";
}

function drawPlots() {
  const rows = sortedRows();
  dotPlot(page.compareEl.querySelector("#cmp-dots"), {
    rows: rows.map((r) => ({ label: page.labels[r.run_id], value: r.nmse, note: r.run_id })),
    xLabel: "same-frame nmse",
  });
  convergenceGroups().forEach((g, i) => {
    linePlot(page.compareEl.querySelector(`[data-conv="${i}"]`), {
      series: g.runs.map((x) => ({ name: page.labels[x.id], x: x.c.iterations, y: x.c.best_score, step: true })),
      xLabel: "iteration",
      yLabel: `best ${g.kind}`,
    });
  });
}

// ---------------------------------------------------------------------------
// Differences between runs of the same algorithm

function diffTables() {
  const byAlgo = new Map();
  for (const r of page.result.rows) {
    const d = page.details[r.run_id];
    if (!d) continue;
    if (!byAlgo.has(r.algorithm)) byAlgo.set(r.algorithm, []);
    byAlgo.get(r.algorithm).push(d);
  }
  return [...byAlgo.entries()].filter(([, ds]) => ds.length >= 2).map(([algo, ds]) => diffTable(algo, ds));
}

function diffRows(ds) {
  const cost = (d) => (d.cost ? `${fmtDuration(d.cost.cpu_seconds)} over ${fmtDuration(d.cost.wallclock_seconds)}` : "n/a");
  const rows = [
    ["Run", ds.map((d) => d.kd_run_id || d.run_id)],
    ["Configuration hash", ds.map((d) => (d.provenance.config_hash ? `${d.provenance.config_hash.slice(0, 12)}…` : "n/a"))],
    ["Seed", ds.map((d) => String(d.seed ?? "n/a"))],
    ["KD version", ds.map((d) => d.provenance.kd_version || "n/a")],
    ["Data the fit saw", ds.map((d) => tail(d.dataset_fingerprint))],
    ["Preprocessing", ds.map((d) => preprocessingLabel(d.preprocessing, d.savgol_window))],
  ];
  const envKeys = [...new Set(ds.flatMap((d) => Object.keys(d.provenance.environment || {})))];
  for (const k of envKeys) rows.push([`Environment: ${k}`, ds.map((d) => (d.provenance.environment || {})[k] ?? "n/a")]);
  rows.push(["CPU time over wall clock", ds.map(cost)], ["Stop reason", ds.map((d) => (d.stop_reason ? stopReasonText(d.stop_reason) : "n/a"))]);
  const keys = [...new Set(ds.flatMap((d) => Object.keys(d.config || {})))].filter((k) => k !== "seed").sort();
  const differing = keys.filter((k) => new Set(ds.map((d) => JSON.stringify((d.config || {})[k]))).size > 1);
  for (const k of differing) rows.push([`Parameter ${k}`, ds.map((d) => (d.config && k in d.config ? JSON.stringify(d.config[k]) : "n/a"))]);
  return { rows, sameParams: keys.length - differing.length };
}

function diffTable(algo, ds) {
  const { rows, sameParams } = diffRows(ds);
  const eqs = ds.map((d) => d.equation_latex);
  const body = rows.map(([label, values]) =>
    h("tr", { class: new Set(values).size > 1 ? "differs" : null }, h("td", null, label), values.map((v) => h("td", { class: label === "Run" ? "mono" : null }, v))),
  );
  body.push(h("tr", { class: new Set(eqs).size > 1 ? "differs" : null }, h("td", null, "Equation"), eqs.map((e) => h("td", null, e ? equation(e, { size: "inline" }) : "n/a"))));
  return [
    h("div", { class: "section-title" }, h("h2", null, `Differences between the ${algo} runs`), h("span", { class: "meta" }, `${sameParams} parameters equal`)),
    h("div", { class: "tablebox" }, h("table", { class: "t diff" }, h("thead", null, h("tr", null, h("th", { style: "width:22%" }), ds.map((d) => h("th", null, page.labels[d.run_id])))), h("tbody", null, body))),
    h("p", { class: "note", style: "margin-top:.5rem" }, "Rows where the runs differ are shaded. Parameters are listed only where they differ. The configuration hash covers every algorithm parameter and the seed; it does not cover the environment."),
  ];
}

/** KD's comparison figure, only when the comparison has one drawn from the
 *  re-measured numbers; the server sends null otherwise (the charts above are
 *  drawn from the table). */
function figureBlock(result) {
  if (!result.figure_artifact_id) return null;
  const body = h("figure", { class: "fig" }, h("img", { src: artifactUrl(result.figure_artifact_id), alt: "KD's comparison figure" }), h("figcaption", null, "Written by KD for this comparison."));
  return [h("div", { class: "section-title" }, h("h2", null, "KD's comparison figure")), body];
}

function downloadCsv() {
  const cols = ["run_id", "algorithm", "seed", "native_score_kind", "native_score_direction", "native_score", "same_frame_nmse", "complexity", "wallclock_seconds", "equation_latex", "error"];
  const rows = sortedRows().map((r) => [
    r.run_id, r.algorithm, r.seed, r.native_score && r.native_score.kind, r.native_score && r.native_score.direction, r.native_score && r.native_score.value,
    r.nmse, r.complexity, wallclock(r.run_id), r.equation && r.equation.latex, r.error,
  ]);
  const cell = (v) => (v === null || v === undefined ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  const text = [cols, ...rows].map((row) => row.map(cell).join(",")).join("\n") + "\n";
  const a = h("a", { href: URL.createObjectURL(new Blob([text], { type: "text/csv" })), download: `comparison-${page.jobId}.csv` });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------------------------------------------------------------------------
// History

function defaultDatasetFilter() {
  const g = selectionGroup();
  const current = g ? g.dataset_name : null;
  if (current && page.runs.some((r) => r.dataset_name === current)) page.filters.dataset = current;
}

function visibleRuns() {
  const f = page.filters;
  return page.runs.filter(
    (r) =>
      (f.status === "all" || r.status === f.status) &&
      (f.algorithm === "all" || r.algorithm === f.algorithm) &&
      (f.dataset === "all" || r.dataset_name === f.dataset) &&
      (f.origin === "all" || r.origin === f.origin),
  );
}

function filterSelect(label, key, values) {
  const select = h("select", { onChange: (e) => { page.filters[key] = e.target.value; renderHistory(); } }, h("option", { value: "all" }, "all"), values.map(([v, text]) => h("option", { value: v }, text)));
  select.value = page.filters[key];
  return h("label", null, label, select);
}

function filters() {
  const distinct = (key) => [...new Set(page.runs.map((r) => r[key]).filter((v) => v !== null))].sort().map((v) => [v, v]);
  return h(
    "div",
    { class: "cmp-filters" },
    filterSelect("Status", "status", Object.entries(RUN_STATUS).map(([k, [, text]]) => [k, text])),
    filterSelect("Algorithm", "algorithm", distinct("algorithm")),
    filterSelect("Dataset", "dataset", distinct("dataset_name")),
    filterSelect("Started from", "origin", [["ui", "this page"], ["imported", "run folders"]]),
  );
}

function renderHistory() {
  if (!page) return;
  const el = clear(page.historyEl);
  const n = page.selected.size;
  const g = selectionGroup();
  el.append(
    h("div", { class: "section-title" }, h("h2", null, "History"), h("span", { class: "meta" }, "select two or more runs on one dataset to compare them")),
    foldersPanel(),
    warningBlock(page.listWarnings) || "",
    h("div", { style: "margin-top:1rem" }, filters()),
    h(
      "div",
      { class: "cmp-selbar" },
      button(`Compare ${n} selected`, { kind: "primary", disabled: n < 2, onClick: () => submitCompare([...page.selected]) }),
      n ? button("Clear selection", { small: true, onClick: () => { page.selected.clear(); renderHistory(); } }) : null,
      h("span", { class: "note" }, g ? `Selection on ${g.dataset_name}, data as loaded ${tail(g.loaded_fingerprint)}. Runs on other datasets cannot join.` : "Runs on different datasets cannot be selected together."),
    ),
    historyTable(visibleRuns()),
    h("div", { style: "margin-top:.6rem" }, page.nextCursor ? button("Load more runs", { onClick: loadMore }) : null),
  );
}

function historyTable(runs) {
  const head = h("thead", null, h("tr", null, ["", "Status", "Run", "Algorithm, data, seed", "Started", "Native score", "Own nmse", "Actions"].map((t, i) => h("th", { class: i === 6 ? "num" : null, title: i === 6 ? "The run's own nmse from its record, measured as that algorithm measures it; not comparable across algorithms. Compare runs for the shared-frame value." : null }, t))));
  const body = runs.length ? runs.map(historyRow) : [h("tr", null, h("td", { colspan: 8, class: "empty" }, page.runs.length ? "No run matches these filters." : "No runs yet. Start one on the Search page, or add a folder of runs above."))];
  return h("div", { class: "tablebox" }, h("table", { class: "t" }, head, h("tbody", null, body)));
}

function historyRow(r) {
  const reason = blockReason(r);
  const checked = page.selected.has(r.run_id);
  const box = h("input", { type: "checkbox", checked, disabled: Boolean(reason) && !checked, "aria-label": `Select ${r.run_id}`, title: reason, onChange: (e) => toggle(r, e.target.checked) });
  const [cls, text] = RUN_STATUS[r.status];
  const sealed = r.status === "completed" || r.status === "raised";
  return h(
    "tr",
    { class: checked ? "sel" : reason ? "off" : null },
    h("td", null, box),
    h("td", null, h("span", { class: `st ${cls}` }, text), r.stop_reason ? h("span", { class: "sub" }, stopReasonText(r.stop_reason)) : null),
    h("td", null, sealed ? h("a", { class: "mono", href: href("result", r.run_id) }, r.kd_run_id || r.run_id) : h("span", { class: "mono" }, r.kd_run_id || r.run_id), lineageText(r), reason && !checked ? h("span", { class: "sub reason" }, reason) : null),
    h("td", null, `${r.algorithm ?? "unknown"} on ${r.dataset_name ?? "unknown data"}${r.seed !== null ? `, seed ${r.seed}` : ""}`, h("span", { class: "sub" }, preprocessingLabel(r.preprocessing, r.savgol_window)), r.prior_knowledge && r.prior_knowledge.sketch ? tag("sketch", "prior") : null),
    h("td", null, r.created_at ? fmtTime(r.created_at) : "n/a", h("span", { class: "sub" }, originText(r))),
    h("td", null, r.native_score ? [fmtNumber(r.native_score.value), h("span", { class: "sub" }, `${r.native_score.kind}, ${r.native_score.direction === "min" ? "lower" : "higher"} is better`)] : h("span", { class: "sub" }, "no score")),
    h("td", { class: "num" }, fmtNumber(r.nmse)),
    h("td", null, h("div", { class: "actions" }, sealed ? linkButton("Open", href("result", r.run_id), { small: true }) : null, r.origin === "ui" ? linkButton("Edit and rerun", href("search", null, { clone: r.run_id }), { small: true }) : null)),
  );
}

function lineageText(r) {
  const l = r.lineage;
  if (!l) return null;
  if (l.resumed_from_run_id) return h("span", { class: "sub" }, `resumed from ${l.resumed_from_run_id}${l.resume_iteration !== null ? ` at iteration ${l.resume_iteration}` : ""}${l.reseed ? ", new seed" : ""}`);
  if (l.cloned_from_run_id) return h("span", { class: "sub" }, `edited from ${l.cloned_from_run_id}`);
  return null;
}

function toggle(r, on) {
  if (on) page.selected.add(r.run_id);
  else page.selected.delete(r.run_id);
  renderHistory();
}

function foldersPanel() {
  const input = h("input", { type: "text", placeholder: "/path/to/a/folder/of/runs", "aria-label": "Folder of runs" });
  const add = async () => {
    const path = input.value.trim();
    if (!path) throw new Error("Give the path of a folder on this machine.");
    const src = await post("/history/sources", { path });
    toast(`Added ${src.path}: ${src.n_runs} runs${src.n_unsealed ? `, ${src.n_unsealed} without a record` : ""}.`);
    await loadHistory();
    renderHistory();
  };
  const own = page.runs.filter((r) => r.origin === "ui").length;
  return panel(
    { title: "Run folders", meta: "runs started from notebooks, scripts, kd-agent or kd-bench are read in place" },
    h(
      "div",
      { class: "cmp-folders" },
      h("div", { class: "f" }, h("span", null, "This workspace", h("span", { class: "sub" }, "runs started from this page")), h("span", null, `${own} runs`)),
      page.sources.map((s) =>
        h("div", { class: "f" }, h("span", null, h("span", { class: "p" }, s.path), h("span", { class: "sub" }, `added ${fmtTime(s.imported_at)}`)), h("span", null, `${s.n_runs} runs${s.n_unsealed ? `, ${s.n_unsealed} without a record` : ""}`)),
      ),
    ),
    h("form", { class: "cmp-add", onSubmit: (e) => e.preventDefault() }, input, button("Add folder", { type: "submit", onClick: add })),
  );
}
