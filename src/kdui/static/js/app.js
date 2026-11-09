// The shell: session, hash router, page header with KD's six steps, and the
// run tray shown on every page. Pages are modules with mount(container,
// params) and unmount(); see README.md for the contract and the routes.

import { ApiError, JOBS_CHANGED, SYSTEM_READ, TERMINAL_STATES, get, openSession, pollEvery, post, store } from "./api.js";
import {
  elapsedSeconds,
  errorState,
  fmtBytes,
  fmtDuration,
  fmtNumber,
  frag,
  h,
  href,
  jobTitle,
  linkButton,
  machineMemory,
  memoryMeter,
  statusBadge,
  stopReasonText,
  tag,
  toast,
  warningBlock,
} from "./ui.js";
import * as compare from "./compare.js";
import * as data from "./data.js";
import * as jobs from "./jobs.js";
import * as prepare from "./prepare.js";
import * as result from "./result.js";
import * as run from "./run.js";
import * as search from "./search.js";

const PAGES = { data, prepare, search, jobs, run, result, compare };
const DEFAULT_ROUTE = "#/data";

const header = document.getElementById("kd-header");
const main = document.getElementById("kd-main");
const trayEl = document.getElementById("kd-tray");

// ---------------------------------------------------------------------------
// Router

/** "#/run/j-1?x=1" -> { page: "run", params: { id: "j-1", path: ["j-1"], x: "1" } } */
export function parseHash(hash) {
  const match = /^#\/?([^?]*)(?:\?(.*))?$/.exec(hash || "");
  const segments = (match ? match[1] : "").split("/").filter(Boolean).map(decodeURIComponent);
  const params = Object.fromEntries(new URLSearchParams(match && match[2] ? match[2] : ""));
  params.path = segments.slice(1);
  if (segments.length > 1) params.id = segments[1];
  return { page: segments[0] || "data", params };
}

let current = null; // { name, module, container }
let routeSeq = 0;

async function route() {
  if (!/^#\//.test(window.location.hash)) {
    window.history.replaceState(null, "", DEFAULT_ROUTE);
  }
  const { page, params } = parseHash(window.location.hash);
  const seq = ++routeSeq;
  if (current && current.name === page && current.module.revisit && current.module.revisit(params)) {
    renderHeader(page, params);
    return;
  }
  if (current) {
    const previous = current;
    current = null;
    try {
      await previous.module.unmount();
    } catch (err) {
      console.error(`unmount of ${previous.name} failed`, err);
    }
    if (previous.container.querySelector("[_echarts_instance_]")) {
      (await import("./plots.js")).disposeWithin(previous.container);
    }
  }
  if (seq !== routeSeq) return; // a newer navigation took over
  if (page === "result" && params.id) store.set("run", params.id);
  renderHeader(page, params);
  const container = h("div", { class: "page", "data-page": page });
  main.replaceChildren(container);
  window.scrollTo(0, 0);
  const module = PAGES[page];
  if (!module) {
    container.append(errorState(new Error(`There is no page "${page}".`), { title: "Unknown address" }), linkButton("Go to the data step", DEFAULT_ROUTE));
    return;
  }
  current = { name: page, module, container };
  try {
    await module.mount(container, params);
  } catch (err) {
    console.error(err);
    if (container.isConnected) container.append(errorState(err, { title: "This page could not be shown.", retry: route }));
  }
}

// ---------------------------------------------------------------------------
// Header: KD mark, current dataset, the six steps, the runs link

const STEPS = [
  ["data", "Data", () => href("data")],
  ["prepare", "Prepare", () => href("prepare")],
  ["search", "Search", () => href("search")],
  ["verify", "Verify", () => href("result", store.get("run"))],
  ["compare", "Compare", () => href("compare")],
  ["report", "Report", () => href("result", store.get("run"), { section: "report" })],
];

function stepOf(page, params) {
  if (page === "result") return params.section === "report" ? "report" : "verify";
  return STEPS.some(([id]) => id === page) ? page : null;
}

let lastRoute = { page: "data", params: {} };
const labels = new Map(); // dataset_id -> label
const units = new Map(); // algorithm -> its iteration unit ("generations", "steps", ...)

function renderHeader(page = lastRoute.page, params = lastRoute.params) {
  lastRoute = { page, params };
  const step = stepOf(page, params);
  const dataset = store.get("dataset");
  const done = {
    data: Boolean(dataset),
    prepare: store.get("preprocessing") !== null,
    search: tray.jobs.length > 0,
    verify: store.get("run") !== null,
  };
  const counts = { running: 0, queued: 0 };
  for (const q of tray.system ? tray.system.queues : []) {
    counts.running += q.running;
    counts.queued += q.queued;
  }
  const datasetName = dataset ? dataset.label || labels.get(dataset.dataset_id) || dataset.dataset_id : null;
  header.replaceChildren(
    h("a", { class: "mark", href: href("data"), "aria-label": "KD" }, h("span", { class: "k" }, "K"), h("span", { class: "d" }, "D")),
    h(
      "div",
      { class: "ctx" },
      h("span", { class: "label" }, "Dataset"),
      dataset ? h("span", { class: "value" }, datasetName, dataset.summary ? h("small", null, dataset.summary) : null) : h("span", { class: "value" }, h("small", { style: "margin:0" }, "none chosen")),
    ),
    h(
      "nav",
      { "aria-label": "Steps" },
      h(
        "ul",
        { class: "steps" },
        STEPS.map(([id, label, link], i) =>
          h("li", null, h("a", { href: link(), class: done[id] ? "done" : null, "aria-current": id === step ? "step" : null }, h("span", { class: "n" }, String(i + 1)), label)),
        ),
      ),
    ),
    h("span", { class: "spacer" }),
    h(
      "a",
      { class: "runs-link", href: href("jobs"), "aria-current": page === "jobs" || page === "run" ? "page" : null },
      h("span", { class: counts.running ? "st running" : "st queued" }, `${counts.running} running`),
      h("span", null, h("b", null, String(counts.queued)), " queued"),
      h("span", { class: "tag host", title: tray.system ? tray.system.hostname : "" }, "this machine"),
    ),
  );
}

// ---------------------------------------------------------------------------
// Run tray: machine memory, running jobs with stop controls, queue, ended

const tray = { system: null, jobs: [], error: null };
const DAY_MS = 24 * 3600 * 1000;

function jobLink(job) {
  if (job.kind === "fit" && job.run_id && TERMINAL_STATES.has(job.state)) return href("result", job.run_id);
  return href("run", job.job_id);
}

/** "Generation" for sga, "Iteration" when the algorithm list is not read yet. */
function unitOf(job) {
  const plural = units.get(job.algorithm) ?? "iterations";
  return plural.charAt(0).toUpperCase() + plural.slice(1).replace(/s$/, "");
}

function runningCard(job) {
  const p = job.progress;
  const preparing = job.phase === "starting" || job.phase === "preparing";
  const facts = [];
  if (preparing) facts.push(job.phase === "starting" ? "Starting: loading data" : "Preparing: the search has not started");
  else if (p && p.iteration !== null) facts.push(`${unitOf(job)} ${p.iteration}`);
  else if (job.phase) facts.push(job.phase === "reporting" ? "Writing the report" : `Phase: ${job.phase}`);
  const elapsed = p && p.elapsed_seconds !== null ? p.elapsed_seconds : job.started_at ? elapsedSeconds(job.started_at) : null;
  if (elapsed !== null) facts.push(`${fmtDuration(elapsed)} elapsed`);
  const best = p && p.best_score && p.best_score.value !== null ? `Best ${p.best_score.kind} ${fmtNumber(p.best_score.value)}` : null;
  return h(
    "div",
    { class: "job" },
    h("div", { class: "what" }, h("a", { href: jobLink(job) }, jobTitle(job, labels.get(job.dataset_id))), statusBadge(job.state, job.phase)),
    h("div", { class: "facts" }, facts.join(", ")),
    best && h("div", { class: "facts" }, best),
    memoryMeter(job.memory_bytes, job.limits.memory_cap_bytes),
    warningBlock(job.warnings),
    h(
      "div",
      { class: "ctl" },
      h("button", { class: "btn small", type: "button", onClick: (e) => stopJob(e.currentTarget, job) }, "Stop and keep result"),
      h("button", { class: "btn small danger", type: "button", onClick: (e) => killJob(e.currentTarget, job) }, "Force stop"),
    ),
  );
}

async function stopJob(btn, job) {
  btn.disabled = true;
  try {
    await post(`/jobs/${encodeURIComponent(job.job_id)}/stop`);
    toast("Stop requested. The run ends after its current iteration and keeps its result.");
  } catch (err) {
    toast(err.message, "error");
    btn.disabled = false;
  }
}

async function killJob(btn, job) {
  const ok = window.confirm(run.killConfirmText(job, labels.get(job.dataset_id)));
  if (!ok) return;
  btn.disabled = true;
  try {
    await post(`/jobs/${encodeURIComponent(job.job_id)}/kill`);
    toast("Force stop sent.");
  } catch (err) {
    toast(err.message, "error");
    btn.disabled = false;
  }
}

function endedNote(job) {
  const took = job.started_at && job.ended_at ? fmtDuration(elapsedSeconds(job.started_at, job.ended_at)) : null;
  return [took, stopReasonText(job.stop_reason)].filter(Boolean).join(", ");
}

function renderTray() {
  const sys = tray.system;
  const running = tray.jobs.filter((j) => j.state === "running");
  const queued = tray.jobs.filter((j) => j.state === "queued").sort((a, b) => a.created_at.localeCompare(b.created_at));
  const now = Date.now();
  const ended = tray.jobs
    .filter((j) => TERMINAL_STATES.has(j.state) && j.ended_at && now - new Date(j.ended_at) < DAY_MS)
    .sort((a, b) => b.ended_at.localeCompare(a.ended_at))
    .slice(0, 8);
  const fitQueue = sys ? sys.queues.find((q) => q.queue === "fit") : null;
  trayEl.replaceChildren(
    frag(
    h("h3", null, "This machine", h("span", { class: "meta" }, sys ? sys.hostname : "memory")),
    sys ? machineMemory(sys.memory, { jobsAtATime: fitQueue ? fitQueue.slots : null }) : h("p", { class: "note" }, "Reading the machine state"),
    sys && sys.warnings.length ? sys.warnings.map((w) => warningBlock([w])) : null,
    tray.error && h("p", { class: "err" }, `Could not read the run state: ${tray.error}`),
    h("h3", null, "Running", h("span", { class: "meta" }, String(running.length))),
    running.length ? running.map(runningCard) : h("p", { class: "empty" }, "Nothing is running."),
    h("h3", null, "Queue", h("span", { class: "meta" }, `${queued.length} waiting`)),
    queued.length
      ? h(
          "ol",
          { class: "queue" },
          queued.map((j, i) =>
            h(
              "li",
              null,
              h("span", { class: "pos" }, String(i + 1)),
              h("span", null, h("a", { href: jobLink(j), style: "color:inherit;text-decoration:none" }, jobTitle(j, labels.get(j.dataset_id))), j.waiting && h("span", { class: "why" }, j.waiting.message)),
              h("span", { class: "cap" }, fmtBytes(j.limits.memory_cap_bytes)),
            ),
          ),
        )
      : h("p", { class: "empty" }, "The queue is empty."),
    h("h3", null, "Ended", h("span", { class: "meta" }, "last 24 h")),
    ended.length
      ? h(
          "ol",
          { class: "queue ended" },
          ended.map((j) => h("li", null, h("span", null, h("a", { href: jobLink(j) }, jobTitle(j, labels.get(j.dataset_id))), h("span", { class: "why" }, endedNote(j))), statusBadge(j.state))),
        )
      : h("p", { class: "empty" }, "No run ended in the last 24 hours."),
    h("div", { class: "foot" }, linkButton("All runs and history", href("jobs"), { small: true })),
    ),
  );
}

async function refreshLabels() {
  const list = await get("/datasets");
  for (const d of list.registered) labels.set(d.dataset_id, d.label);
}

async function refreshTray() {
  try {
    const [system, jobList] = await Promise.all([get("/system"), get("/jobs", { limit: 200 })]);
    tray.system = system;
    tray.jobs = jobList.jobs;
    tray.error = null;
    window.dispatchEvent(new CustomEvent(SYSTEM_READ, { detail: system }));
    if (tray.jobs.some((j) => j.dataset_id && !labels.has(j.dataset_id))) await refreshLabels();
  } catch (err) {
    tray.error = err.message;
    if (err instanceof ApiError && err.status === 401) throw err;
  }
  renderTray();
  renderHeader();
}

// ---------------------------------------------------------------------------
// Start

function fatal(title, message) {
  document.getElementById("app").replaceChildren(
    h("div", { class: "fatal" }, h("h1", null, title), h("p", null, message)),
  );
}

async function start() {
  try {
    await openSession();
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) {
      fatal("No session", "Open the address that kd-ui printed when it started; it ends in #token=. A copied address without the token does not open a session.");
    } else {
      fatal("KD is not reachable", err.message);
    }
    return;
  }
  renderHeader();
  renderTray();
  get("/capabilities")
    .then((caps) => {
      for (const a of caps.snapshot ? caps.snapshot.algorithms : []) units.set(a.algorithm, a.segmentation.unit);
      renderTray();
    })
    .catch((err) => console.error(err));
  const poller = pollEvery(refreshTray, { interval: 2000, hiddenInterval: 10000, onError: (err) => console.error(err) });
  window.addEventListener(JOBS_CHANGED, () => poller.now());
  for (const key of ["dataset", "preprocessing", "run"]) store.on(key, () => renderHeader());
  window.addEventListener("hashchange", route);
  await route();
}

start();
