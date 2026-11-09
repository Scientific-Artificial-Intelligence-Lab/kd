// Page "run" (#/run/<job_id>): one job, live. Phase strip, the best equation
// so far, the best native score by iteration, the equation changes, candidate
// counts, stop and force stop, resources, the request and the log tail. Every
// number is the API's; the page only lays them out.
//
// The helpers exported at the bottom (styles, stop and kill, iteration unit
// and budget) are shared with jobs.js.

import { JOBS_CHANGED, TERMINAL_STATES, followJob, get, post } from "./api.js";
import {
  button,
  elapsedSeconds,
  emptyState,
  equation,
  errorState,
  fmtBytes,
  fmtDuration,
  fmtNumber,
  fmtTime,
  go,
  h,
  href,
  jobTitle,
  kv,
  linkButton,
  loadingState,
  memoryMeter,
  nativeScore,
  pageHead,
  panel,
  smoothingText,
  statusBadge,
  stopReasonText,
  table,
  toast,
  warningBlock,
} from "./ui.js";
import { linePlot } from "./plots.js";

const MAX_LOG_LINES = 2000;

let page = null; // state of the mounted page

export async function mount(container, params) {
  const jobId = params.id;
  if (!jobId) {
    container.append(pageHead({ title: "Run", crumbs: [["Runs", href("jobs")]] }), emptyState("No job is named in the address.", linkButton("All runs", href("jobs"))));
    return;
  }
  container.append(loadingState("Reading the job"));
  const [job, snapshot, labels] = await Promise.all([
    get(`/jobs/${encodeURIComponent(jobId)}`).catch((err) => err),
    loadSnapshot(),
    loadDatasetLabels(),
  ]);
  if (!container.isConnected) return;
  container.replaceChildren();
  if (job instanceof Error) {
    container.append(
      pageHead({ title: "Run", crumbs: [["Runs", href("jobs")], [jobId, null]] }),
      errorState(job, { title: `Job ${jobId} could not be read.`, retry: () => go("run", jobId) }),
    );
    return;
  }
  // The handlers write to this page's own state: an answer for a page that was
  // left must never reach the page shown now.
  const self = newPage(container, job, snapshot, labels);
  page = self;
  render();
  self.handle = followJob(jobId, {
    onJob: (next) => {
      self.job = next;
      self.pollError = null;
    },
    onEvents: (eventsPage) => takeEvents(self, eventsPage),
    onLog: (logPage) => takeLog(self, logPage),
    onError: (err) => {
      self.pollError = err;
      if (page === self) render();
    },
  });
  // followJob calls the three handlers in turn; draw once per round.
  self.timer = window.setInterval(() => page === self && self.dirty && render(), 250);
  self.onChanged = () => self.handle.now();
  window.addEventListener(JOBS_CHANGED, self.onChanged);
}

export function unmount() {
  if (!page) return;
  page.handle && page.handle.stop();
  window.clearInterval(page.timer);
  window.removeEventListener(JOBS_CHANGED, page.onChanged);
  page = null;
}

// ---------------------------------------------------------------------------
// State

function newPage(container, job, snapshot, labels) {
  const slots = {};
  for (const name of ["head", "phases", "notice", "outcome", "left", "right"]) slots[name] = h("div");
  slots.left.className = "stack";
  slots.right.className = "stack";
  container.append(slots.head, slots.phases, slots.notice, slots.outcome, h("div", { class: "split-wide" }, slots.left, slots.right));
  const algo = snapshot && job.algorithm ? snapshot.algorithms.find((a) => a.algorithm === job.algorithm) || null : null;
  return {
    container,
    slots,
    job,
    algo,
    labels,
    phases: [], // PhaseChange
    iters: [], // IterationUpdate
    changes: [], // { iteration, score, latex }
    skipped: 0,
    logLines: [],
    logTail: "", // an unterminated last line
    dirty: true,
    built: null, // panels built once: { kind, ... }
    pollError: null,
    handle: null,
  };
}

function takeEvents(page, eventsPage) {
  if (eventsPage.reset) {
    page.phases = [];
    page.iters = [];
    page.changes = [];
    page.skipped = 0;
  }
  for (const event of eventsPage.events) {
    if (event.event === "phase") page.phases.push(event);
    else {
      page.iters.push(event);
      if (event.best_equation_latex !== null) page.changes.push({ iteration: event.iteration, score: event.best_score, latex: event.best_equation_latex });
    }
  }
  page.skipped += eventsPage.skipped;
  page.dirty = true;
}

function takeLog(page, logPage) {
  if (logPage.reset) {
    page.logLines = [];
    page.logTail = "";
  }
  if (!logPage.text) {
    page.dirty = true;
    return;
  }
  const parts = (page.logTail + logPage.text).split("\n");
  page.logTail = parts.pop();
  page.logLines.push(...parts);
  if (page.logLines.length > MAX_LOG_LINES) page.logLines.splice(0, page.logLines.length - MAX_LOG_LINES);
  page.dirty = true;
}

// ---------------------------------------------------------------------------
// Rendering

function render() {
  if (!page || !page.container.isConnected) return;
  page.dirty = false;
  const { job } = page;
  renderHead();
  page.slots.phases.replaceChildren(phaseStrip());
  page.slots.notice.replaceChildren(...[pollNotice(), warningBlock(job.warnings)].filter(Boolean));
  page.slots.outcome.replaceChildren(...[outcomePanel()].filter(Boolean));
  if (job.kind === "fit") renderFit();
  else renderLight();
}

function renderHead() {
  const { job } = page;
  const title = jobTitle(job, page.labels.get(job.dataset_id));
  page.container.ownerDocument.title = `${title} | KD`;
  const sub = h("p", { class: "sub" }, statusBadge(job.state, job.phase), " ", statusLine(), " ", hostTag(job));
  const head = pageHead({ title, crumbs: [["Runs", href("jobs")], [job.job_id, null]], actions: headActions() });
  head.firstChild.append(sub);
  page.slots.head.replaceChildren(head);
}

function statusLine() {
  const { job } = page;
  const unit = iterationUnit(page.algo);
  if (job.state === "queued") return job.waiting ? `waiting: ${job.waiting.message}` : "starts at the next pass of the scheduler.";
  if (job.state === "running") {
    const runFor = `running for ${fmtDuration(Math.max(0, elapsedSeconds(job.started_at)))}`;
    if (job.phase === "starting") return `starting: the worker is loading the data, ${runFor}.`;
    if (job.phase === "preparing") return `preparing: kd has not started the search yet, ${runFor}.`;
    if (job.phase === "reporting") return `the result is sealed; the report is being written, ${runFor}.`;
    if (job.phase === "working") return `working, ${runFor}.`;
    const it = lastIteration();
    if (it === null) return `searching, ${runFor}.`;
    const budget = iterationBudget(requestOf(job), page.algo);
    const done = `${it + 1}${budget !== null ? ` of ${budget}` : ""} ${unit.plural} done`;
    const into = searchElapsed() !== null ? `${fmtDuration(searchElapsed())} into the search` : runFor;
    return `searching, ${done}, ${into}.`;
  }
  const took = job.started_at ? `, took ${fmtDuration(elapsedSeconds(job.started_at, job.ended_at))}` : "";
  const reason = job.stop_reason && job.stop_reason !== "finished" ? `${stopReasonText(job.stop_reason)}, ` : "";
  return `${reason}ended ${job.ended_at ? fmtTime(job.ended_at) : "at an unknown time"}${took}.`;
}

function headActions() {
  const { job } = page;
  const actions = [];
  const runId = job.kind === "fit" ? job.run_id : null;
  if (TERMINAL_STATES.has(job.state)) {
    if (job.kind === "fit" && runId) actions.push(linkButton("Open result", href("result", runId), { kind: "primary" }));
    if (job.kind === "verify" && job.state === "completed") actions.push(linkButton("Open the checked run", href("result", job.request.run_id), { kind: "primary" }));
    if (job.kind === "compare" && job.state === "completed") actions.push(linkButton("Open the comparison", href("compare", null, { runs: job.request.run_ids.join(","), job: job.job_id }), { kind: "primary" }));
    if (job.kind === "fit" && runId) actions.push(linkButton("Resume", href("jobs", null, { resume: runId })));
  }
  if (runId) actions.push(linkButton("Edit and rerun", href("search", null, { clone: runId })));
  if (job.state === "queued") actions.push(button("Remove from queue", { onClick: () => stopJob(job).then(applyJob) }));
  if (job.state === "running") {
    actions.push(stopButton(job), killButton(job));
  }
  return actions;
}

function stopButton(job, opts = {}) {
  const requested = Boolean(job.stop_requested_at);
  return button(requested ? "Stop requested" : "Stop and keep result", { ...opts, disabled: requested, onClick: () => stopJob(job).then(applyJob) });
}

function killButton(job, opts = {}) {
  return button("Force stop", { ...opts, kind: "danger", disabled: Boolean(job.kill_requested_at), onClick: () => killJob(job, page.labels.get(job.dataset_id)).then(applyJob) });
}

function applyJob(next) {
  if (!page || !next) return;
  page.job = next;
  render();
  page.handle.now();
}

function hostTag(job) {
  return job.hostname ? h("span", { class: "tag host", title: job.hostname }, "this machine") : null;
}

function pollNotice() {
  if (!page.pollError) return null;
  return h("div", { class: "warning", role: "status" }, `Updates paused: ${page.pollError.message} The page tries again every few seconds.`);
}

// Phase strip --------------------------------------------------------------

function phaseStrip() {
  const { job } = page;
  const at = new Map();
  for (const p of page.phases) if (!at.has(p.phase)) at.set(p.phase, p.at);
  const unit = iterationUnit(page.algo);
  const ended = TERMINAL_STATES.has(job.state);
  const endCell = {
    p: ended ? endLabel(job) : job.kind === "fit" ? "Sealed" : "Result",
    w: ended ? `${job.ended_at ? fmtTime(job.ended_at) : ""}${job.stop_reason ? `, ${stopReasonText(job.stop_reason)}` : ""}` : job.kind === "fit" ? "the result is sealed, then the report is written" : "the result is written",
  };
  let cells;
  if (job.kind === "fit") {
    const searchStart = at.get("searching");
    const prepStart = at.get("preparing");
    const it = lastIteration();
    cells = [
      { id: "queued", p: "Queued", w: `submitted ${fmtTime(job.created_at)}` },
      { id: "starting", p: "Started", w: job.started_at ? fmtTime(job.started_at) : "not yet" },
      { id: "preparing", p: "Preparing", w: prepStart ? `${fmtTime(prepStart)}${searchStart ? `, took ${fmtDuration(elapsedSeconds(prepStart, searchStart))}` : ""}` : "not yet" },
      { id: "searching", p: "Searching", w: it !== null ? `${unit.singular} ${it}${searchElapsed() !== null ? `, ${fmtDuration(searchElapsed())}` : ""}` : searchStart ? `from ${fmtTime(searchStart)}` : "not yet" },
      { id: "end", ...endCell, p: job.phase === "reporting" ? "Reporting" : endCell.p },
    ];
  } else {
    cells = [
      { id: "queued", p: "Queued", w: `submitted ${fmtTime(job.created_at)}` },
      { id: "starting", p: "Started", w: job.started_at ? fmtTime(job.started_at) : "not yet" },
      { id: "working", p: "Working", w: at.get("working") ? `from ${fmtTime(at.get("working"))}` : "not yet" },
      { id: "end", ...endCell },
    ];
  }
  const current = ended ? cells.length : job.state === "queued" ? 0 : cells.findIndex((c) => c.id === (job.phase === "reporting" ? "end" : job.phase));
  return h(
    "div",
    { class: "phases", style: `grid-template-columns:repeat(${cells.length},minmax(0,1fr))` },
    cells.map((c, i) => h("div", { class: i < current ? "done" : i === current ? "now" : "later" }, h("span", { class: "p" }, c.p), h("span", { class: "w" }, c.w))),
  );
}

function endLabel(job) {
  return { completed: "Completed", budget_stopped: "Budget reached", cancelled: "Cancelled", failed: "Failed", interrupted: "Interrupted" }[job.state] || job.state;
}

// Outcome of an ended job ----------------------------------------------------

function outcomePanel() {
  const { job } = page;
  if (!TERMINAL_STATES.has(job.state) || (job.state === "completed" && !job.error)) return null;
  const items = [];
  const cap = job.limits.memory_cap_bytes;
  const peak = job.process ? job.process.peak_memory_bytes : null;
  const measure = job.process ? job.process.memory_measure : null;
  const reason = job.stop_reason;
  if (reason === "memory_limit") {
    items.push(h("p", null, `Memory limit reached: the measured peak was ${peak !== null ? fmtBytes(peak) : "not recorded"}${measure ? ` (${measure})` : ""} against a cap of ${fmtBytes(cap)} (${capKindText(job.limits.memory_cap_kind)}).`));
  } else if (reason === "system_killed") {
    items.push(h("p", null, `Ended by the system, probably out of memory: the operating system ended the worker; kd-ui did not ask for it. Peak ${peak !== null ? fmtBytes(peak) : "not recorded"}, cap ${fmtBytes(cap)}.`));
  } else if (reason === "killed") {
    items.push(h("p", null, job.kind === "fit" ? `Force-stopped. No equation was sealed; the structure below is the last best one in the progress journal${lastIteration() !== null ? ` (${iterationUnit(page.algo).singular} ${lastIteration()})` : ""}.` : "Force-stopped. Nothing the job computed was kept."));
  } else if (reason === "user_stop") {
    items.push(h("p", null, "Stopped by you at the end of an iteration; the result was sealed."));
  } else if (reason === "hard_timeout") {
    items.push(h("p", null, `The hard time limit${job.limits.hard_seconds ? ` of ${fmtDuration(job.limits.hard_seconds)}` : ""} ended the worker.`));
  } else if (reason) {
    items.push(h("p", null, `Stop reason: ${stopReasonText(reason)}.`));
  }
  if (job.error) {
    items.push(h("p", { class: "note" }, `${job.error.error_type}, kd's message:`), h("pre", { class: "log err" }, job.error.message));
  }
  if (job.process) {
    items.push(kv([["Return code", job.process.returncode === null ? "none" : String(job.process.returncode)], ["Wall clock", fmtDuration(job.process.wallclock_seconds)], ["Peak memory", `${fmtBytes(job.process.peak_memory_bytes)} (${job.process.memory_measure})`]]));
  }
  const resumable = job.kind === "fit" && job.run_id && ["memory_limit", "system_killed", "killed", "user_stop", "hard_timeout", "machine_restarted", "process_lost", "terminated"].includes(reason);
  const actions = resumable ? [linkButton("Resume from a checkpoint", href("jobs", null, { resume: job.run_id }), { small: true })] : null;
  const title = job.state === "failed" ? "Why the job failed" : job.state === "interrupted" ? "Why the job was interrupted" : "How the job ended";
  return panel({ title, actions }, ...items, resumable ? h("p", { class: "note" }, "A fit keeps its checkpoints; Resume shows them and says whether this run can continue.") : null);
}

// Fit layout ---------------------------------------------------------------

function renderFit() {
  const { job, slots } = page;
  const oneShot = page.algo ? page.algo.one_shot : false;
  if (!page.built) {
    page.built = {
      best: h("section", { class: "panel" }),
      score: oneShot ? null : plotPanel("score"),
      changes: h("section", { class: "panel" }),
      cands: oneShot ? null : plotPanel("cands"),
      oneShot: oneShot ? h("section", { class: "panel" }) : null,
      stop: h("section", { class: "panel" }),
      resources: h("section", { class: "panel" }),
      run: h("section", { class: "panel" }),
      params: h("section", { class: "panel" }),
      log: logPanel(),
      nChanges: -1,
      scorePlot: null,
      candPlot: null,
    };
    const b = page.built;
    slots.left.replaceChildren(...[b.best, b.oneShot, b.score && b.score.el, b.changes, b.cands && b.cands.el].filter(Boolean));
    slots.right.replaceChildren(b.stop, b.resources, b.run, b.params, b.log.el);
  }
  const b = page.built;
  b.best.replaceChildren(...bestPanelContent());
  if (b.oneShot) b.oneShot.replaceChildren(...oneShotContent());
  // Until a fit reports an iteration, the first panel says why there is none.
  const noProgress = !page.iters.length;
  for (const el of [b.score && b.score.el, b.changes, b.cands && b.cands.el]) if (el) el.hidden = noProgress;
  if (b.score) drawScore(b.score);
  if (b.changes && b.nChanges !== page.changes.length) {
    b.nChanges = page.changes.length;
    b.changes.replaceChildren(...changesContent());
  }
  if (b.cands) drawCandidates(b.cands);
  b.stop.replaceChildren(...stopContent());
  b.stop.hidden = !["queued", "running"].includes(job.state);
  b.resources.replaceChildren(...resourcesContent());
  b.run.replaceChildren(...runContent());
  b.params.replaceChildren(...paramsContent());
  drawLog(b.log);
}

function bestPanelContent() {
  const { job } = page;
  const unit = iterationUnit(page.algo);
  const last = page.changes[page.changes.length - 1] || null;
  const latex = last ? last.latex : job.progress ? job.progress.best_equation_latex : null;
  const score = page.iters.length ? page.iters[page.iters.length - 1].best_score : job.progress ? job.progress.best_score : null;
  const it = lastIteration();
  const head = h("header", null, h("h2", null, TERMINAL_STATES.has(job.state) ? "Best structure in the progress journal" : "Best structure so far"), h("span", { class: "meta" }, it !== null ? `progress journal, ${unit.singular} ${it}` : "progress journal"));
  if (!latex) {
    return [head, h("p", { class: "note" }, waitingText())];
  }
  const reads = h(
    "div",
    { class: "reads best-meta" },
    h("div", { class: "r" }, h("span", { class: "l" }, score ? `${score.kind}, ${job.algorithm}'s native score (${score.direction === "min" ? "lower" : "higher"} is better)` : "Native score"), h("span", { class: "v" }, score ? fmtNumber(score.value) : "n/a")),
    last ? h("div", { class: "r" }, h("span", { class: "l" }, `Found at ${unit.singular}`), h("span", { class: "v" }, String(last.iteration))) : null,
    last && it !== null ? h("div", { class: "r" }, h("span", { class: "l" }, "Unchanged for"), h("span", { class: "v" }, `${it - last.iteration} ${it - last.iteration === 1 ? unit.singular : unit.plural}`)) : null,
  );
  const foot = TERMINAL_STATES.has(job.state)
    ? h("p", { class: "note" }, "The best structure in the progress journal. Coefficients shown here, if any, are the search's working values as kd wrote them (SGA writes them); the fitted equation is on the result page.", job.run_id ? [" ", h("a", { href: href("result", job.run_id) }, "Open the result")] : null)
    : h("p", { class: "note" }, "The best structure found so far, from the run's progress journal. Coefficients shown here, if any, are the search's working values as kd wrote them (SGA writes them). The fitted equation and the run's nmse are read from the sealed record when the run ends.");
  return [head, equation(latex, { size: "large" }), reads, foot];
}

function waitingText() {
  const { job } = page;
  if (job.state === "queued") return `Not started. ${job.waiting ? job.waiting.message : "It starts at the next pass of the scheduler."}`;
  if (job.phase === "starting") return "The worker is starting and loading the data. No iteration has been reported yet.";
  if (job.phase === "preparing") return "kd is preparing the search (for example, a surrogate network is training). No iterations are reported in this phase, so no progress fraction is shown.";
  if (job.state === "running") return "No iteration has been reported yet.";
  return "The progress journal of this job holds no iteration.";
}

function oneShotContent() {
  return [
    h("header", null, h("h2", null, "Progress")),
    h("p", { class: "note" }, `${page.job.algorithm} fits once, with no iterations: there is no live score curve. The equation appears when the fit returns.`),
  ];
}

function plotPanel(which) {
  const plot = h("div", { class: which === "score" ? "plot" : "plot short" });
  const empty = h("p", { class: "note" });
  const header = h("header", null, h("h2"), h("span", { class: "meta" }));
  const caption = h("p", { class: "note" });
  const el = h("section", { class: "panel" }, header, plot, empty, caption);
  return { el, plot, empty, header, caption };
}

function drawScore(p) {
  const { job } = page;
  const unit = iterationUnit(page.algo);
  const first = page.iters[0];
  const kind = first ? first.best_score.kind : page.algo ? page.algo.score_kind : "score";
  const direction = first ? first.best_score.direction : page.algo ? page.algo.score_direction : "min";
  p.header.firstChild.textContent = `Best ${kind} by ${unit.singular}`;
  p.header.lastChild.textContent = `${job.algorithm}'s native score; ${direction === "min" ? "lower" : "higher"} is better`;
  const has = page.iters.length > 0;
  p.plot.hidden = !has;
  p.empty.hidden = has;
  p.empty.textContent = has ? "" : waitingText();
  p.caption.textContent = has ? `Dots mark the ${unit.plural} where the best structure changed.${remaining() ? ` The shaded part is the remaining ${unit.singular} budget.` : ""}` : "";
  if (!has) return;
  const data = {
    series: [{ name: `best ${kind}`, x: page.iters.map((e) => e.iteration), y: page.iters.map((e) => e.best_score.value), step: true }],
    xLabel: unit.singular,
    yLabel: `best ${kind}`,
    marks: page.changes.map((c) => ({ x: c.iteration, y: c.score.value })),
    ...remaining(),
  };
  if (p.chart) p.chart.update(data);
  else p.chart = linePlot(p.plot, data);
}

function remaining() {
  const budget = iterationBudget(requestOf(page.job), page.algo);
  const it = lastIteration();
  if (budget === null || it === null || TERMINAL_STATES.has(page.job.state) || it + 1 >= budget) return null;
  return { remainingFrom: it, xMax: budget };
}

function changesContent() {
  const unit = iterationUnit(page.algo);
  const kind = page.changes.length ? page.changes[0].score.kind : "score";
  const head = h("header", null, h("h2", null, "Equation changes"), h("span", { class: "meta" }, "each time the best structure changed, as the progress journal has it"));
  if (!page.changes.length) return [head, h("p", { class: "note" }, waitingText())];
  const rows = page.changes;
  return [
    head,
    table(
      [
        { label: capitalize(unit.singular), num: true, render: (c) => String(c.iteration) },
        { label: kind, num: true, render: (c) => fmtNumber(c.score.value) },
        { label: "Best structure", className: "tex", render: (c) => equation(c.latex, { size: "inline" }) },
      ],
      rows,
      { className: "hist" },
    ),
  ];
}

function drawCandidates(p) {
  const unit = iterationUnit(page.algo);
  p.header.firstChild.textContent = `Candidates per ${unit.singular}`;
  p.header.lastChild.textContent = "candidate equations evaluated, and how many were invalid";
  const has = page.iters.length > 0;
  p.plot.hidden = !has;
  p.empty.hidden = has;
  p.empty.textContent = has ? "" : waitingText();
  if (!has) {
    p.caption.textContent = "";
    return;
  }
  const x = page.iters.map((e) => e.iteration);
  const data = {
    series: [
      { name: "evaluated", x, y: page.iters.map((e) => e.n_candidates) },
      { name: "invalid", x, y: page.iters.map((e) => e.n_invalid) },
    ],
    xLabel: unit.singular,
    yLabel: "candidates",
    xMax: remaining() ? remaining().xMax : undefined,
  };
  if (p.chart) p.chart.update(data);
  else p.chart = linePlot(p.plot, data);
  const last = page.iters[page.iters.length - 1];
  const parts = [`Last ${unit.singular} (${last.iteration}): ${last.n_candidates} evaluated, ${last.n_invalid} invalid`];
  if (last.n_unique_candidates !== null) parts.push(`${last.n_unique_candidates} distinct`);
  if (last.mean_complexity !== null) parts.push(`mean complexity ${fmtNumber(last.mean_complexity, 3)}`);
  p.caption.textContent = `${parts.join(", ")}.${page.skipped ? ` ${page.skipped} unreadable line${page.skipped === 1 ? "" : "s"} of the journal skipped.` : ""}`;
}

function stopContent() {
  const { job } = page;
  const head = h("header", null, h("h2", null, job.state === "queued" ? "Leaving the queue" : "Stopping this run"));
  if (job.state === "queued") {
    return [head, h("div", { class: "stopbox" }, button("Remove from queue", { onClick: () => stopJob(job).then(applyJob) }), h("p", null, "The job has not started; removing it ends it as cancelled. Nothing else is affected."))];
  }
  if (job.state !== "running") return [head];
  const unit = iterationUnit(page.algo);
  const it = lastIteration();
  const per = it !== null && searchElapsed() ? searchElapsed() / (it + 1) : null;
  const oneShot = page.algo ? page.algo.one_shot : false;
  let soft;
  if (oneShot) soft = `${job.algorithm} fits once: the stop takes effect only when that fit returns, and the result is then sealed. Status becomes Completed, stopped by you.`;
  else if (job.phase === "starting" || job.phase === "preparing") soft = `Stops at the end of the current ${unit.singular} and seals the result. During ${job.phase === "starting" ? "start-up" : "preparation"} it has no effect until the search begins. Status becomes Completed, stopped by you.`;
  else soft = `Stops at the end of the current ${unit.singular}${per !== null ? ` (this run takes about ${fmtDuration(per)} per ${unit.singular})` : ""} and seals the result as for a completed run. Status becomes Completed, stopped by you, with the result kept; it can be resumed later from its last checkpoint.`;
  const best = page.changes.length ? page.changes[page.changes.length - 1].iteration : null;
  const hard = `Ends the run's process group at once. No equation is sealed: the run keeps the best structure from its progress journal${best !== null ? ` (${unit.singular} ${best})` : ""} and the checkpoints written so far, from which it can be resumed. Status becomes Cancelled, force-stopped. Use this when the soft stop has no effect, for example during preparation.`;
  return [
    head,
    h(
      "div",
      { class: "stopbox" },
      stopButton(job),
      h("p", null, soft),
      job.stop_requested_at ? h("p", { class: "note" }, `Stop requested at ${fmtTime(job.stop_requested_at)}; the run ends at the next ${unit.singular} boundary.`) : null,
      killButton(job),
      h("p", null, hard),
      job.kill_requested_at ? h("p", { class: "note" }, `Force stop requested at ${fmtTime(job.kill_requested_at)}.`) : null,
    ),
  ];
}

function resourcesContent() {
  const { job } = page;
  const lim = job.limits;
  const pairs = [];
  if (job.state === "running") pairs.push(["Memory now", h("div", null, memoryMeter(job.memory_bytes, lim.memory_cap_bytes))]);
  else pairs.push(["Memory cap", fmtBytes(lim.memory_cap_bytes)]);
  if (job.process) pairs.push(["Peak", `${fmtBytes(job.process.peak_memory_bytes)} (${job.process.memory_measure})`]);
  pairs.push(["Cap enforced by", lim.memory_cap_kind ? capKindText(lim.memory_cap_kind) : "known once the job starts"]);
  pairs.push(["Time budget", lim.max_seconds ? `${fmtDuration(lim.max_seconds)}, kd's soft budget` : "none"]);
  if (lim.hard_seconds) {
    const left = job.state === "running" && job.started_at ? `, ${fmtDuration(Math.max(0, lim.hard_seconds - elapsedSeconds(job.started_at)))} left` : "";
    pairs.push(["Hard time limit", `${fmtDuration(lim.hard_seconds)}${left}`]);
  } else pairs.push(["Hard time limit", "none"]);
  const req = requestOf(job);
  if (req && req.checkpoint_every) pairs.push(["Checkpoints", `every ${req.checkpoint_every} ${iterationUnit(page.algo).plural}`]);
  pairs.push(["Machine", job.hostname ? `this machine (${job.hostname})` : "this machine, once started"]);
  return [h("header", null, h("h2", null, "Resources"), h("span", { class: "meta" }, "sampled by the run supervisor")), kv(pairs)];
}

function runContent() {
  const { job } = page;
  const req = job.request;
  const label = page.labels.get(job.dataset_id);
  const pairs = [
    ["Job id", h("span", { class: "mono" }, job.job_id)],
    ["Kind", { fit: "fit (search)", verify: "verification", compare: "comparison" }[job.kind]],
  ];
  if (job.kind === "fit") {
    pairs.push(["Run id", job.run_id ? h("span", { class: "mono" }, job.run_id) : "assigned at start"]);
    pairs.push(["Algorithm", [job.algorithm, ", mode ", h("span", { class: "mono" }, req.mode || "chosen by kd")]]);
    pairs.push(["Dataset", [label ? h("span", { class: "mono" }, label) : null, label ? " " : null, h("span", { class: "mono note" }, job.dataset_id)]]);
    pairs.push(["Seed", String(job.seed)]);
    pairs.push(["Device", req.device]);
    pairs.push(["Smoothing", smoothingText(req.preprocessing.smoothing, req.params.savgol_window) ?? "none"]);
    if (job.prior_knowledge) pairs.push(["Sketch", job.prior_knowledge.sketch ? "yes" : "no"]);
    if (req.note) pairs.push(["Note", req.note]);
  }
  pairs.push(["Submitted", fmtTime(job.created_at)]);
  if (job.started_at) pairs.push(["Started", fmtTime(job.started_at)]);
  if (job.ended_at) pairs.push(["Ended", fmtTime(job.ended_at)]);
  const lin = job.lineage;
  if (lin && lin.resumed_from_run_id) pairs.push(["Resumed from", [h("a", { href: href("result", lin.resumed_from_run_id), class: "mono" }, lin.resumed_from_run_id), lin.resume_iteration !== null ? ` at ${iterationUnit(page.algo).singular} ${lin.resume_iteration}` : "", lin.reseed ? ", new seed (branch)" : ""]]);
  if (lin && lin.cloned_from_run_id) pairs.push(["Edited from", h("a", { href: href("result", lin.cloned_from_run_id), class: "mono" }, lin.cloned_from_run_id)]);
  return [h("header", null, h("h2", null, "Run")), kv(pairs)];
}

function paramsContent() {
  const { job } = page;
  const params = job.request.params || {};
  const keys = Object.keys(params).sort();
  return [
    h("header", null, h("h2", null, "Parameters"), job.run_id ? linkButton("Edit and rerun", href("search", null, { clone: job.run_id }), { small: true }) : null),
    keys.length ? kv(keys.map((k) => [h("span", { class: "mono" }, k), shownValue(params[k])])) : h("p", { class: "note" }, "kd's defaults: no parameter was changed in the request."),
  ];
}

// Light jobs -------------------------------------------------------------

function renderLight() {
  const { job, slots } = page;
  if (!page.built) {
    page.built = { what: h("section", { class: "panel" }), stop: h("section", { class: "panel" }), resources: h("section", { class: "panel" }), run: h("section", { class: "panel" }), log: logPanel() };
    const b = page.built;
    slots.left.replaceChildren(b.what, b.log.el);
    slots.right.replaceChildren(b.stop, b.resources, b.run);
  }
  const b = page.built;
  const req = job.request;
  const what =
    job.kind === "verify"
      ? [h("p", null, "Checks of run ", h("a", { href: href("result", req.run_id), class: "mono" }, req.run_id), ": ", req.checks.map((c) => c.replace(/_/g, " ")).join(", "), ".")]
      : [h("p", null, "Comparison of ", req.run_ids.length, " runs in one shared frame: "), h("ul", null, req.run_ids.map((id) => h("li", null, h("a", { href: href("result", id), class: "mono" }, id))))];
  const done = job.state === "completed" ? h("p", { class: "note" }, job.kind === "verify" ? "The results are shown on the result page of the checked run." : "The comparison table is on the Compare page.") : null;
  b.what.replaceChildren(h("header", null, h("h2", null, job.kind === "verify" ? "Verification" : "Comparison")), ...what, ...[done].filter(Boolean));
  b.stop.replaceChildren(...lightStopContent());
  b.stop.hidden = !["queued", "running"].includes(job.state);
  b.resources.replaceChildren(...resourcesContent());
  b.run.replaceChildren(...runContent());
  drawLog(b.log);
}

function lightStopContent() {
  const { job } = page;
  if (job.state === "queued") return [h("header", null, h("h2", null, "Leaving the queue")), h("div", { class: "stopbox" }, button("Remove from queue", { onClick: () => stopJob(job).then(applyJob) }))];
  return [h("header", null, h("h2", null, "Stopping this job")), h("div", { class: "stopbox" }, killButton(job), h("p", null, "Ends the job's process at once. Nothing it computed so far is kept."))];
}

// Log --------------------------------------------------------------------

function logPanel() {
  const pre = h("pre", { class: "log" });
  const el = panel({ title: "Log", meta: "the worker's log, newest last" }, pre);
  return { el, pre, shown: -1 };
}

function drawLog(log) {
  const lines = page.logTail ? [...page.logLines, page.logTail] : page.logLines;
  const key = `${lines.length}:${page.logTail.length}`;
  if (log.shown === key) return;
  log.shown = key;
  const atBottom = log.pre.scrollTop + log.pre.clientHeight >= log.pre.scrollHeight - 8;
  if (!lines.length) log.pre.replaceChildren(h("div", { class: "note" }, page.job.state === "queued" ? "Nothing yet: the job has not started." : "Nothing logged yet."));
  else log.pre.replaceChildren(...lines.map((line) => h("div", { class: /\b(ERROR|Traceback)\b/.test(line) ? "err" : null }, line || " ")));
  if (atBottom) log.pre.scrollTop = log.pre.scrollHeight;
}

// ---------------------------------------------------------------------------
// Small readers

function lastIteration() {
  if (page.iters.length) return page.iters[page.iters.length - 1].iteration;
  return page.job.progress && page.job.progress.iteration !== null ? page.job.progress.iteration : null;
}

/** Seconds since the search loop started, at the last iteration. */
function searchElapsed() {
  if (page.iters.length) return page.iters[page.iters.length - 1].elapsed_seconds;
  return page.job.progress ? page.job.progress.elapsed_seconds : null;
}

function requestOf(job) {
  return job.request || null;
}

function shownValue(v) {
  if (v === null) return "None";
  if (typeof v === "object") return h("span", { class: "mono" }, JSON.stringify(v));
  return String(v);
}

const capitalize = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

function capKindText(kind) {
  return { cgroup: "the kernel (systemd scope)", job_object: "a Windows job object (committed memory)", sampled: "sampling by the supervisor" }[kind] || kind;
}

// ---------------------------------------------------------------------------
// Shared with jobs.js

/** The capability snapshot, or null when it is not available (a warning is shown). */
export async function loadSnapshot() {
  try {
    const caps = await get("/capabilities");
    return caps.snapshot;
  } catch (err) {
    toast(`The algorithm list could not be read (${err.message}); units and budgets are not shown.`, "warn");
    return null;
  }
}

/** dataset_id -> label of the registered datasets; empty when the list cannot be read. */
export async function loadDatasetLabels() {
  const labels = new Map();
  try {
    const list = await get("/datasets");
    for (const d of list.registered) labels.set(d.dataset_id, d.label);
  } catch (err) {
    toast(`The dataset list could not be read (${err.message}).`, "warn");
  }
  return labels;
}

/** What one iteration of the algorithm is called (Segmentation.unit). */
export function iterationUnit(algo) {
  const plural = algo ? algo.segmentation.unit : "iterations";
  return { plural, singular: plural.replace(/s$/, "") };
}

/** The iteration budget of a fit request: kd's facade parameter whose effect is
 *  max_iterations, as requested or its default. null when kd declares none. */
export function iterationBudget(request, algo) {
  if (!request || request.kind !== "fit" || !algo) return null;
  const param = algo.facade_params.find((p) => p.effect === "max_iterations");
  if (!param) return null;
  const value = param.name in request.params ? request.params[param.name] : param.default;
  return typeof value === "number" ? value : null;
}

/** POST stop: cancels a queued job, or asks a running fit to stop and keep its result. */
export async function stopJob(job) {
  const next = await post(`/jobs/${encodeURIComponent(job.job_id)}/stop`);
  toast(job.state === "queued" ? "Removed from the queue." : "Stop requested: the run stops at the end of the current iteration and seals its result.");
  return next;
}

/** The confirmation of a force stop: what the run and result pages show afterwards. */
export function killConfirmText(job, datasetLabel = null) {
  const kept = job.kind === "fit" ? "No equation is sealed: the run keeps the best structure from its progress journal and the checkpoints written so far, from which it can be resumed." : "Nothing it computed so far is kept.";
  return `Force stop ${jobTitle(job, datasetLabel)}?\n\nThe process ends now. ${kept}`;
}

/** POST kill after a confirmation; resolves with the JobDetail, or null when declined. */
export async function killJob(job, datasetLabel = null) {
  const ok = window.confirm(killConfirmText(job, datasetLabel));
  if (!ok) return null;
  const next = await post(`/jobs/${encodeURIComponent(job.job_id)}/kill`);
  toast("Force stop requested.");
  return next;
}

