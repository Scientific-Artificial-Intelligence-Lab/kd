// Page "jobs" (#/jobs, titled Runs): memory of the machine, run folders, the
// running and queued jobs with their controls, the history of runs (this UI's
// and those of added folders), ended verification and comparison jobs, and
// the resume panel (#/jobs?resume=<run_id>).

import { JOBS_CHANGED, TERMINAL_STATES, get, pollEvery, post } from "./api.js";
import {
  button,
  elapsedSeconds,
  emptyState,
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
  machineMemory,
  memoryMeter,
  nativeScore,
  pageHead,
  panel,
  replaceRoute,
  schemaForm,
  statusBadge,
  stopReasonText,
  table,
  toast,
  warningBlock,
} from "./ui.js";
import { iterationBudget, iterationUnit, killJob, loadDatasetLabels, loadSnapshot, stopJob } from "./run.js";

let page = null;

export async function mount(container, params) {
  const slots = {
    machine: h("section", { class: "panel" }, loadingState()),
    folders: h("section", { class: "panel" }, loadingState()),
    active: h("div", null, loadingState()),
    history: h("div", null, loadingState()),
    light: h("div"),
    resume: h("div", { id: "resume" }),
  };
  container.append(
    pageHead({
      title: "Runs",
      sub: "Jobs started here and runs found in the run folders below. Closing the browser or the kd-ui server does not stop a run; each run has its own address and is listed here again when the page is reopened.",
    }),
    h("div", { class: "grid-2" }, slots.machine, slots.folders),
    sectionTitle("Running and queued", "fits and light jobs wait in separate queues"),
    slots.active,
    sectionTitle("History", "select two or more runs on one dataset to compare them"),
    slots.history,
    slots.light,
    slots.resume,
  );
  page = {
    container,
    slots,
    system: null,
    jobs: null,
    runs: null,
    listWarnings: [],
    sources: null,
    foldersList: null,
    snapshot: null,
    labels: new Map(),
    selected: new Set(),
    filters: { status: "all", algorithm: "all", dataset: "all", origin: "all" },
    endedKey: null,
    historyKey: null,
    resumeRun: params.resume || null,
  };
  const [snapshot, labels] = await Promise.all([loadSnapshot(), loadDatasetLabels()]);
  if (!page || page.container !== container) return;
  page.snapshot = snapshot;
  page.labels = labels;
  page.poll = pollEvery(refreshJobs, {
    interval: 2000,
    onError: (err) => {
      page.slots.active.replaceChildren(errorState(err, { title: "The job list could not be read.", retry: () => page.poll.now() }));
    },
  });
  page.onChanged = () => {
    page.poll.now();
    loadRuns();
  };
  window.addEventListener(JOBS_CHANGED, page.onChanged);
  loadRuns();
  loadSources();
  if (page.resumeRun) showResume(page.resumeRun);
}

export function unmount() {
  if (!page) return;
  page.poll && page.poll.stop();
  window.removeEventListener(JOBS_CHANGED, page.onChanged);
  page = null;
}

const sectionTitle = (title, meta) => h("div", { class: "section-title" }, h("h2", null, title), h("span", { class: "meta" }, meta));

// ---------------------------------------------------------------------------
// Loading

async function refreshJobs() {
  const [system, list] = await Promise.all([get("/system"), get("/jobs", { limit: 200 })]);
  if (!page) return true;
  const first = page.system === null;
  page.system = system;
  page.jobs = list.jobs;
  if (first && page.sources) renderFolders();
  renderMachine();
  renderActive();
  renderLight();
  // Reload the history when a job ended since the last round.
  const endedKey = page.jobs.filter((j) => TERMINAL_STATES.has(j.state)).map((j) => j.job_id).join(",");
  if (page.endedKey !== null && endedKey !== page.endedKey) loadRuns();
  page.endedKey = endedKey;
  if (page.runs) renderHistory();
  return false;
}

async function loadRuns() {
  try {
    const list = await get("/runs", { limit: 200 });
    if (!page) return;
    page.runs = list.runs;
    page.listWarnings = list.warnings;
    renderHistory();
  } catch (err) {
    if (!page) return;
    page.historyKey = null;
    page.slots.history.replaceChildren(errorState(err, { title: "The run history could not be read.", retry: loadRuns }));
  }
}

async function loadSources() {
  try {
    const list = await get("/history/sources");
    if (!page) return;
    page.sources = list.sources;
    renderFolders();
  } catch (err) {
    if (!page) return;
    page.foldersList = null;
    page.slots.folders.replaceChildren(h("header", null, h("h2", null, "Run folders")), errorState(err, { title: "The run folders could not be read.", retry: loadSources }));
  }
}

// ---------------------------------------------------------------------------
// Machine and folders

function renderMachine() {
  const { system } = page;
  const fit = system.queues.find((q) => q.queue === "fit");
  const light = system.queues.find((q) => q.queue === "light");
  const mem = system.memory;
  page.slots.machine.replaceChildren(
    h("header", null, h("h2", null, "Memory on this machine"), h("span", { class: "meta" }, "sampled every 2 s")),
    machineMemory(mem, { jobsAtATime: fit ? fit.slots : null }),
    warningBlock(system.warnings) || "",
    kv([
      fit ? ["Fit queue", `${fit.slots} at a time; ${fit.running} running, ${fit.queued} queued`] : null,
      light ? ["Light queue", `${light.slots} at a time (verification, comparison); ${light.running} running, ${light.queued} queued`] : null,
      ["Default caps", `fit ${fmtBytes(mem.default_fit_cap_bytes)}, light job ${fmtBytes(mem.default_light_cap_bytes)}`],
    ]),
    h(
      "p",
      { class: "note", style: "margin-top:.6rem" },
      `A queued job waits only behind running jobs: it starts when its queue has a free slot and its cap plus the supervisor allowance (${fmtBytes(mem.supervisor_allowance_bytes)}) fits beside the running jobs. A job with nothing running ahead of it always starts, with a warning when headroom is low.`,
    ),
  );
}

function renderFolders() {
  if (!page.foldersList) buildFolders();
  const workspace = page.system ? page.system.workspace : null;
  // kd-ui resolves the workspace: a POSIX path starts with "/", a Windows path never does.
  const runsDir = workspace ? `${workspace}${workspace.startsWith("/") ? "/" : "\\"}runs` : "runs of this UI";
  page.foldersList.replaceChildren(
    h("li", null, h("span", null, h("span", { class: "path" }, runsDir), h("span", { class: "src" }, "runs started from this page")), h("span", null, uiRunCount())),
    ...page.sources.map((s) =>
      h(
        "li",
        null,
        h("span", null, h("span", { class: "path" }, s.path), h("span", { class: "src" }, `added ${fmtTime(s.imported_at)}; read in place, nothing is copied`)),
        h("span", null, `${s.n_runs} run${s.n_runs === 1 ? "" : "s"}${s.n_unsealed ? `, ${s.n_unsealed} unsealed` : ""}`),
      ),
    ),
  );
}

/** The folders panel once: its list is redrawn, the path input is kept. */
function buildFolders() {
  const input = h("input", { type: "text", placeholder: "/path/to/a/folder/of/kd/runs", "aria-label": "Folder of runs" });
  const add = button("Add folder", {
    small: true,
    onClick: async () => {
      const path = input.value.trim();
      if (!path) {
        toast("Type the path of a folder of kd runs first.", "warn");
        return;
      }
      const source = await post("/history/sources", { path });
      toast(`Added ${source.path}: ${source.n_runs} run${source.n_runs === 1 ? "" : "s"} found.`);
      input.value = "";
      loadSources();
      loadRuns();
    },
  });
  page.foldersList = h("ul", { class: "folders" });
  page.slots.folders.replaceChildren(
    h("header", null, h("h2", null, "Run folders")),
    page.foldersList,
    h("div", { class: "addfolder" }, input, add),
    h("p", { class: "note", style: "margin-top:.4rem" }, "A folder of runs written by a script, a notebook, kd-agent or kd-bench is added once and remembered."),
  );
}

function uiRunCount() {
  if (!page.runs) return "";
  const n = page.runs.filter((r) => r.origin === "ui").length;
  return `${n} run${n === 1 ? "" : "s"}`;
}

// ---------------------------------------------------------------------------
// Running and queued

function renderActive() {
  const active = page.jobs.filter((j) => !TERMINAL_STATES.has(j.state));
  // The list is newest first; the queue order is oldest first.
  const ordered = [];
  for (const queue of ["fit", "light"]) {
    const inQueue = active.filter((j) => j.queue === queue);
    const running = inQueue.filter((j) => j.state === "running");
    const queued = inQueue.filter((j) => j.state === "queued").sort((a, b) => a.created_at.localeCompare(b.created_at));
    if (!running.length && !queued.length) continue;
    ordered.push({ group: queue === "fit" ? `Fit queue: ${running.length} running, ${queued.length} queued` : `Light queue (verification and comparison): ${running.length} running, ${queued.length} queued` });
    running.forEach((j) => ordered.push({ job: j, pos: "run" }));
    queued.forEach((j, i) => ordered.push({ job: j, pos: String(i + 1) }));
  }
  if (!ordered.length) {
    page.slots.active.replaceChildren(emptyState("No job is running or queued.", linkButton("Start a search", href("search"), { kind: "primary" })));
    return;
  }
  const algoOf = (job) => (page.snapshot && job.algorithm ? page.snapshot.algorithms.find((a) => a.algorithm === job.algorithm) : null);
  const tbody = h(
    "tbody",
    null,
    ordered.map((row) => {
      if (row.group) return h("tr", { class: "group" }, h("td", { colspan: 8 }, row.group));
      const job = row.job;
      const unit = iterationUnit(algoOf(job));
      return h(
        "tr",
        null,
        h("td", null, row.pos),
        h("td", null, statusBadge(job.state, job.phase), job.state === "running" && job.phase ? h("span", { class: "why" }, job.phase) : null, job.waiting ? h("span", { class: "why" }, job.waiting.message) : null),
        h(
          "td",
          null,
          h("a", { href: href("run", job.job_id) }, jobTitle(job, page.labels.get(job.dataset_id))),
          h("span", { class: "why mono" }, job.job_id),
          job.warnings.map((w) => h("span", { class: "why warn" }, w)),
        ),
        h("td", null, progressText(job, unit)),
        h("td", null, job.progress ? nativeScore(job.progress.best_score) : h("span", { class: "note" }, "none yet")),
        h("td", null, job.state === "running" ? memoryMeter(job.memory_bytes, job.limits.memory_cap_bytes) : `cap ${fmtBytes(job.limits.memory_cap_bytes)}`),
        h("td", null, h("span", { class: "tag host", title: job.hostname || "" }, "this machine")),
        h("td", null, h("div", { class: "runs-actions" }, activeActions(job))),
      );
    }),
  );
  const head = h("thead", null, h("tr", null, ["#", "Status", "Job", "Progress", "Best native score", "Memory", "Machine", "Actions"].map((t) => h("th", null, t))));
  page.slots.active.replaceChildren(h("div", { class: "tablebox" }, h("table", { class: "t" }, head, tbody)));
}

function progressText(job, unit) {
  if (job.state === "queued") return h("span", { class: "note" }, "not started");
  const elapsed = job.started_at ? fmtDuration(Math.max(0, elapsedSeconds(job.started_at))) : "";
  if (job.progress && job.progress.iteration !== null) {
    const searched = job.progress.elapsed_seconds !== null ? `${fmtDuration(job.progress.elapsed_seconds)} into the search` : `${elapsed} elapsed`;
    return [`${capitalize(unit.singular)} ${job.progress.iteration}`, h("span", { class: "why" }, searched)];
  }
  if (job.phase === "preparing") return ["preparing", h("span", { class: "why" }, `${elapsed} elapsed, no iterations yet`)];
  return [job.phase || "started", h("span", { class: "why" }, `${elapsed} elapsed`)];
}

function activeActions(job) {
  const open = linkButton("Open", href("run", job.job_id), { small: true });
  if (job.state === "queued") return [open, button("Remove", { small: true, title: "Remove this job from the queue", onClick: () => stopJob(job).then(() => page && page.poll.now()) })];
  return [
    open,
    button(job.stop_requested_at ? "Stop requested" : "Stop and keep result", { small: true, disabled: Boolean(job.stop_requested_at), onClick: () => stopJob(job).then(() => page && page.poll.now()) }),
    button("Force stop", { small: true, kind: "danger", disabled: Boolean(job.kill_requested_at), onClick: () => killJob(job, page.labels.get(job.dataset_id)).then(() => page && page.poll.now()) }),
  ];
}

// ---------------------------------------------------------------------------
// History of runs

const RUN_STATUS = {
  completed: ["completed", "Completed"],
  raised: ["failed", "Failed"],
  no_record: ["interrupted", "No record"],
  unsealed: ["queued", "Unsealed"],
};

function jobOfRun(run) {
  return run.job_id && page.jobs ? page.jobs.find((j) => j.job_id === run.job_id) || null : null;
}

/** [filter key, badge, note under it] of one run. A UI run takes its job's state. */
function runStatus(run) {
  const job = jobOfRun(run);
  if (job) return [job.state, statusBadge(job.state, job.phase), stopReasonText(job.stop_reason)];
  const [cls, label] = RUN_STATUS[run.status] || ["queued", run.status];
  const note = run.status === "unsealed" ? "no manifest: still running, or ended before the seal" : run.stop_reason ? stopReasonText(run.stop_reason) : run.status === "raised" ? "kd raised an error" : "";
  return [run.status === "raised" ? "failed" : run.status, h("span", { class: `st ${cls}` }, label), note];
}

function renderHistory() {
  const runs = page.runs;
  const key = JSON.stringify([runs, page.listWarnings, page.filters, [...page.selected], runs.map((r) => (jobOfRun(r) || {}).state)]);
  if (key === page.historyKey) return;
  page.historyKey = key;
  if (page.sources) renderFolders();
  const f = page.filters;
  const shown = runs.filter((r) => {
    const [status] = runStatus(r);
    return (f.status === "all" || status === f.status) && (f.algorithm === "all" || r.algorithm === f.algorithm) && (f.dataset === "all" || r.dataset_name === f.dataset) && (f.origin === "all" || r.origin === f.origin);
  });
  const options = (values) => ["all", ...[...new Set(values.filter((v) => v !== null && v !== undefined))].sort()];
  const select = (label, name, values, text = (v) => v) =>
    h(
      "label",
      null,
      label,
      h(
        "select",
        { onChange: (e) => { page.filters[name] = e.target.value; renderHistory(); } },
        values.map((v) => h("option", { value: v, selected: v === f[name] }, v === "all" ? "all" : text(v))),
      ),
    );
  const selected = runs.filter((r) => page.selected.has(r.run_id));
  const compare = button(`Compare ${selected.length} selected`, {
    kind: "primary",
    disabled: selected.length < 2,
    onClick: () => {
      const names = new Set(selected.map((r) => r.dataset_name));
      if (names.size > 1) toast("The selected runs are on different datasets; the comparison groups runs by dataset.", "warn");
      go("compare", null, { runs: selected.map((r) => r.run_id).join(",") });
    },
  });
  const filters = h(
    "div",
    { class: "runs-filters" },
    select("Status", "status", options(runs.map((r) => runStatus(r)[0]))),
    select("Algorithm", "algorithm", options(runs.map((r) => r.algorithm))),
    select("Dataset", "dataset", options(runs.map((r) => r.dataset_name))),
    select("Started from", "origin", ["all", "ui", "imported"], (v) => (v === "ui" ? "this UI" : "an added folder")),
    compare,
  );
  const columns = [
    {
      label: "",
      render: (r) =>
        h("input", {
          type: "checkbox",
          "aria-label": `Select ${r.run_id}`,
          checked: page.selected.has(r.run_id),
          onChange: (e) => {
            if (e.target.checked) page.selected.add(r.run_id);
            else page.selected.delete(r.run_id);
            renderHistory();
          },
        }),
    },
    { label: "Status", render: (r) => { const [, badge, note] = runStatus(r); return [badge, note ? h("span", { class: "why" }, note) : null]; } },
    { label: "Run", render: (r) => [h("a", { href: runLink(r), class: "mono" }, r.run_id), r.kd_run_id && r.kd_run_id !== r.run_id ? h("span", { class: "why mono" }, r.kd_run_id) : null, lineageNote(r)] },
    { label: "Algorithm, data, seed", render: (r) => (r.algorithm ? `${r.algorithm} on ${r.dataset_name || "unknown data"}${r.seed !== null ? `, seed ${r.seed}` : ""}` : h("span", { class: "note" }, "not recorded")) },
    { label: "Started", render: (r) => [r.created_at ? fmtTime(r.created_at) : h("span", { class: "note" }, "unknown"), h("span", { class: "why" }, originText(r))] },
    { label: "Native score", render: (r) => (r.native_score ? nativeScore(r.native_score) : h("span", { class: "note" }, "none")) },
    { label: "Own nmse", num: true, render: (r) => (r.nmse !== null ? fmtNumber(r.nmse) : h("span", { class: "note" }, "n/a")) },
    { label: "Actions", render: (r) => h("div", { class: "runs-actions" }, historyActions(r)) },
  ];
  page.slots.history.replaceChildren(
    ...[warningBlock(page.listWarnings)].filter(Boolean),
    filters,
    table(columns, shown, { empty: runs.length ? "No run matches these filters." : "No runs yet. Runs started from Search, and runs in added folders, are listed here.", rowClass: (r) => (page.selected.has(r.run_id) ? "sel" : null) }),
    h("p", { class: "note", style: "margin-top:.4rem" }, "Native score and nmse are each run's own values, measured as its algorithm measures them; native scores of different kinds are not comparable. Compare re-measures the selected runs in one shared frame."),
  );
}

function runLink(run) {
  const job = jobOfRun(run);
  if (job && !TERMINAL_STATES.has(job.state)) return href("run", job.job_id);
  return href("result", run.run_id);
}

function lineageNote(run) {
  const lin = run.lineage;
  if (!lin) return null;
  if (lin.resumed_from_run_id) return h("span", { class: "why" }, `resumed from ${lin.resumed_from_run_id}${lin.resume_iteration !== null ? ` at ${lin.resume_iteration}` : ""}${lin.reseed ? ", new seed" : ""}`);
  if (lin.cloned_from_run_id) return h("span", { class: "why" }, `edited from ${lin.cloned_from_run_id}`);
  return null;
}

function originText(run) {
  if (run.origin === "ui") return "from this page";
  const source = page.sources ? page.sources.find((s) => s.source_id === run.source_id) : null;
  return source ? `from ${source.path}` : "from an added folder";
}

function historyActions(run) {
  const job = jobOfRun(run);
  const live = job && !TERMINAL_STATES.has(job.state);
  const actions = [linkButton("Open", runLink(run), { small: true })];
  if (!live && run.status !== "unsealed") actions.push(button("Resume", { small: true, onClick: () => showResume(run.run_id) }));
  if (run.status !== "unsealed" || live) actions.push(linkButton("Edit and rerun", href("search", null, { clone: run.run_id }), { small: true }));
  return actions;
}

// ---------------------------------------------------------------------------
// Ended verification and comparison jobs

function renderLight() {
  const ended = page.jobs.filter((j) => j.kind !== "fit" && TERMINAL_STATES.has(j.state));
  if (!ended.length) {
    page.slots.light.replaceChildren();
    return;
  }
  page.slots.light.replaceChildren(
    sectionTitle("Verification and comparison jobs", "ended"),
    table(
      [
        { label: "Status", render: (j) => [statusBadge(j.state), j.stop_reason && j.stop_reason !== "finished" ? h("span", { class: "why" }, stopReasonText(j.stop_reason)) : null] },
        { label: "Job", render: (j) => [h("a", { href: href("run", j.job_id) }, jobTitle(j)), h("span", { class: "why mono" }, j.job_id)] },
        { label: "Ended", render: (j) => [j.ended_at ? fmtTime(j.ended_at) : "", j.started_at && j.ended_at ? h("span", { class: "why" }, fmtDuration(elapsedSeconds(j.started_at, j.ended_at))) : null] },
        { label: "Actions", render: (j) => h("div", { class: "runs-actions" }, linkButton("Open", href("run", j.job_id), { small: true }), j.kind === "verify" && j.run_id ? linkButton("Checked run", href("result", j.run_id), { small: true }) : null) },
      ],
      ended,
    ),
  );
}

// ---------------------------------------------------------------------------
// Resume

async function showResume(runId) {
  page.resumeRun = runId;
  replaceRoute("jobs", null, { resume: runId });
  const slot = page.slots.resume;
  slot.replaceChildren(sectionTitle(`Resume ${runId}`, "continue a run from one of its checkpoints"), loadingState("Reading the checkpoints"));
  slot.scrollIntoView({ behavior: "smooth", block: "start" });
  let checkpoints;
  let run;
  try {
    [checkpoints, run] = await Promise.all([get(`/runs/${encodeURIComponent(runId)}/checkpoints`), get(`/runs/${encodeURIComponent(runId)}`)]);
  } catch (err) {
    if (!page || page.resumeRun !== runId) return;
    slot.replaceChildren(sectionTitle(`Resume ${runId}`, ""), errorState(err, { title: "The checkpoints of this run could not be read.", retry: () => showResume(runId) }), h("div", { class: "actions" }, button("Close", { onClick: closeResume })));
    return;
  }
  if (!page || page.resumeRun !== runId) return;
  const title = run.algorithm ? `Resume ${run.algorithm} on ${run.dataset_name || "unknown data"}${run.seed !== null ? `, seed ${run.seed}` : ""}` : `Resume ${runId}`;
  if (!checkpoints.resumable || !checkpoints.checkpoints.length) {
    slot.replaceChildren(
      sectionTitle(title, run.run_id),
      panel(
        { title: "This run cannot be resumed" },
        h("p", null, checkpoints.reason || "It has no checkpoint."),
        h("p", { class: "note" }, "Edit and rerun starts a new run from the beginning with the same or changed parameters."),
        h("div", { class: "actions", style: "margin-top:.6rem" }, linkButton("Edit and rerun", href("search", null, { clone: runId }), { kind: "primary" }), button("Close", { onClick: closeResume })),
      ),
    );
    return;
  }
  slot.replaceChildren(sectionTitle(title, h("span", { class: "mono" }, run.run_id)), resumeForm(run, checkpoints));
}

function closeResume() {
  if (!page) return;
  page.resumeRun = null;
  page.slots.resume.replaceChildren();
  replaceRoute("jobs");
}

function resumeForm(run, checkpoints) {
  const algo = page.snapshot && run.algorithm ? page.snapshot.algorithms.find((a) => a.algorithm === run.algorithm) || null : null;
  const unit = iterationUnit(algo || (checkpoints.unit ? { segmentation: { unit: checkpoints.unit } } : null));
  const list = [...checkpoints.checkpoints].sort((a, b) => b.iteration - a.iteration);
  let chosen = list[0].iteration;
  const radios = table(
    [
      { label: "", render: (c) => h("input", { type: "radio", name: "ckpt", checked: c.iteration === chosen, "aria-label": `${unit.singular} ${c.iteration}`, onChange: () => { chosen = c.iteration; } }) },
      { label: capitalize(unit.singular), num: true, render: (c) => String(c.iteration) },
      { label: "Size", num: true, render: (c) => (c.size_bytes === null ? "" : fmtBytes(c.size_bytes)) },
      { label: "", render: (c, i) => (i === 0 ? "latest, the default" : "") },
    ],
    list,
  );
  // The iteration budget of the resumed segment (counted from zero).
  const budgetParam = algo ? algo.facade_params.find((p) => p.effect === "max_iterations") : null;
  const original = iterationBudget(run.request, algo);
  const budget = budgetParam ? h("input", { type: "number", class: "num", min: "1", step: "1", value: original !== null ? String(original) : String(budgetParam.default) }) : null;
  const reseed = h("input", { type: "checkbox" });
  const seed = h("input", { type: "number", class: "num", step: "1", value: run.seed !== null ? String(run.seed + 1) : "0", disabled: true });
  reseed.addEventListener("change", () => (seed.disabled = !reseed.checked));
  const maxSeconds = h("input", { type: "number", class: "num", min: "0", step: "any", placeholder: "none" });
  const hardSeconds = h("input", { type: "number", class: "num", min: "0", step: "any", placeholder: "auto", title: "The supervisor ends the run when this time has passed since it started; nothing is sealed, its checkpoints stay. Empty: the time budget plus 2 min, or no limit without a time budget." });
  const capGiB = h("input", { type: "number", class: "num", min: "0", step: "any", placeholder: page.system ? (page.system.memory.default_fit_cap_bytes / 2 ** 30).toFixed(1) : "default" });
  const canReseed = algo ? algo.segmentation.reseed : true;
  const opts = h(
    "div",
    { class: "resume-opts" },
    budget ? h("div", { class: "row" }, h("label", null, `${capitalize(unit.plural)} to run after the checkpoint `, budget), h("span", { class: "note" }, `parameter ${budgetParam.name}; the resumed segment counts from zero`)) : null,
    canReseed ? h("div", { class: "row" }, h("label", { class: "check" }, reseed, " continue with a new seed (branch)"), h("label", null, "seed ", seed)) : null,
    h("div", { class: "row" }, h("label", null, "Time budget (s) ", maxSeconds), h("label", null, "Hard time limit (s) ", hardSeconds), h("label", null, "Memory cap (GiB) ", capGiB)),
    h("p", { class: "note" }, "Resuming writes a new run whose record names this run as its parent. This run stays as it is."),
  );
  const initial = { ...(run.config || {}), ...(run.request ? run.request.params : {}) };
  const form = algo ? schemaForm(algo.fields, { values: initial, lockNotResumeSafe: true }) : null;
  const before = form ? form.values() : {};
  const submit = button("Queue resumed run", {
    kind: "primary",
    onClick: async () => {
      if (form && form.errors().length) {
        toast(form.errors().map((e) => `${e.key}: ${e.message}`).join("; "), "error");
        return;
      }
      const params = {};
      if (form) {
        for (const [key, value] of Object.entries(form.values())) if (JSON.stringify(value) !== JSON.stringify(before[key])) params[key] = value;
      }
      if (budget && budget.value !== "") params[budgetParam.name] = Number(budget.value);
      const body = {
        iteration: chosen === list[0].iteration ? null : chosen,
        params,
        reseed: reseed.checked,
        seed: reseed.checked && seed.value !== "" ? Number(seed.value) : null,
        max_seconds: maxSeconds.value !== "" ? Number(maxSeconds.value) : null,
        hard_seconds: hardSeconds.value !== "" ? Number(hardSeconds.value) : null,
        memory_cap_bytes: capGiB.value !== "" ? Math.round(Number(capGiB.value) * 2 ** 30) : null,
      };
      const accepted = await post(`/runs/${encodeURIComponent(run.run_id)}/resume`, body);
      const job = accepted.jobs[0];
      toast(`Queued: ${jobTitle(job, page.labels.get(job.dataset_id))}.`);
      for (const w of job.warnings) toast(w, "warn");
      go("run", job.job_id);
    },
  });
  return h(
    "div",
    { class: "grid-2" },
    panel({ title: "Checkpoint" }, radios, opts),
    panel(
      { title: "Parameters" },
      form ? form.element : h("p", { class: "note" }, "The parameter list of this algorithm is not available; the run resumes with its own parameters."),
      h("p", { class: "note", style: "margin-top:.5rem" }, `Only parameters kd marks resume_safe can change${checkpoints.resume_safe_params.length ? ` (${checkpoints.resume_safe_params.join(", ")})` : ""}. To change any other parameter, use Edit and rerun, which starts a new run from the beginning.`),
      h("div", { class: "actions", style: "margin-top:.8rem;justify-content:flex-end" }, button("Cancel", { onClick: closeResume }), submit),
    ),
  );
}

const capitalize = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
