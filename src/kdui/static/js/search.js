// Search step: choose algorithms, modes, seeds and
// parameters for the chosen dataset, run kd's pre-launch check and queue one
// fit job per algorithm, mode and seed.
//
// Data: GET /capabilities (forms from kd.instrument_schemas), GET /system
// (default memory cap, queue), GET /datasets/{id} and /datasets/{id}/preview
// (kd's applicability of every mode), POST /preflight, POST /jobs.
// Edit and rerun: #/search?clone=<run_id> reads GET /runs/{id} and submits the
// configuration of that run through POST /runs/{id}/clone.
//
// Only kd's own word blocks: a mode whose declaration refuses the data, a mode
// that declares a sketch clause unsupported, and a refusal of the pre-launch
// check. Availability findings and memory headroom are warnings.

import { ApiError, SYSTEM_READ, get, post, store } from "./api.js";
import {
  button,
  emptyState,
  errorState,
  frag,
  fmtBytes,
  h,
  href,
  linkButton,
  loadingState,
  pageHead,
  panel,
  parseSeeds,
  schemaForm,
  table,
  toast,
  warningBlock,
} from "./ui.js";

const GIB = 1024 ** 3;
/** kdui.settings.Settings.hard_slack_seconds, the default hard limit's margin over the time budget. */
const HARD_SLACK_MINUTES = 2;
const SET = "<set>"; // applicability placeholder: the mode needs a value here
const SKETCH_CLAUSES = ["fixed_terms", "anchors", "hole_count", "derivative_order", "operator_set", "field_axis_set"];
const DERIVATIVE_TEXT = { finite_diff: "finite differences", fd2: "second-order finite differences", autograd: "autograd (neural field)", both: "finite differences or autograd, as each mode takes them" };
const SMOOTHING_TEXT = { lowpass: "low-pass smoothing", network: "network smoothing" };

let S = null; // page state; null when unmounted

/** Replace the children of el; null children are skipped. */
const fill = (el, ...children) => el.replaceChildren(frag(...children));

// ---------------------------------------------------------------------------
// Mount

export async function mount(container, params) {
  fill(container, loadingState("Reading KD's algorithms and the dataset"));
  try {
    await load(container, params);
  } catch (err) {
    if (!(err instanceof ApiError)) throw err;
    if (!container.isConnected) return;
    const what = params.clone ? `Run ${params.clone} or its dataset could not be read.` : "The chosen dataset or KD's algorithm list could not be read.";
    fill(container, pageHead({ title: "Search" }), errorState(err, { title: what, retry: () => mount(container, params) }), h("p", { class: "note" }, "Choose the dataset again on the ", h("a", { href: href("data") }, "Data"), " page if it was removed."));
  }
}

async function load(container, params) {
  const clone = params.clone ? await get(`/runs/${encodeURIComponent(params.clone)}`) : null;
  const datasetId = clone?.dataset_id ?? store.get("dataset")?.dataset_id ?? null;
  if (!datasetId) {
    fill(container, 
      pageHead({ title: "Search", sub: "Choose algorithms, modes, seeds and parameters, and queue the runs." }),
      emptyState(clone ? `Run ${clone.run_id} names no registered dataset. Choose the data it was fitted on first.` : "No dataset is chosen yet. Search runs on the dataset chosen in step 1.", linkButton("Choose data", href("data"), { kind: "primary" })),
    );
    return;
  }
  const [caps, system, dataset] = await Promise.all([get("/capabilities"), get("/system"), get(`/datasets/${encodeURIComponent(datasetId)}`)]);
  if (!container.isConnected) return;
  if (!caps.snapshot) {
    fill(container, pageHead({ title: "Search" }), emptyState("KD's algorithm list has not been read yet.", button("Read it now", { kind: "primary", onClick: async () => { await post("/capabilities/refresh"); if (container.isConnected) await mount(container, params); } })));
    return;
  }
  let preview = null;
  let previewError = null;
  try {
    preview = await get(`/datasets/${encodeURIComponent(datasetId)}/preview`);
  } catch (err) {
    if (!(err instanceof ApiError)) throw err;
    previewError = err;
  }
  if (!container.isConnected) return;
  S = newState(container, { clone, caps: caps.snapshot, system, dataset, preview, previewError });
  render();
  // The queue table says when each run starts: follow the tray's reading of
  // the queues and memory, not the one taken when the page opened.
  const self = S;
  self.onSystem = (event) => {
    if (S !== self) return;
    const before = systemShown(self.system);
    self.system = event.detail;
    if (systemShown(self.system) !== before) renderQueue();
  };
  window.addEventListener(SYSTEM_READ, self.onSystem);
}

/** The parts of SystemInfo the queue table and its warnings show. */
const systemShown = (system) =>
  JSON.stringify([system.queues, fmtBytes(system.memory.admissible_bytes), fmtBytes(system.memory.default_fit_cap_bytes)]);

export function unmount() {
  if (S && S.onSystem) window.removeEventListener(SYSTEM_READ, S.onSystem);
  S = null;
}

function newState(container, { clone, caps, system, dataset, preview, previewError }) {
  const request = clone?.request ?? null;
  // The Prepare page's choices apply only to the dataset it was set for.
  const stored = store.get("preprocessing", {}) ?? {};
  const prep = clone
    ? { preprocessing: request?.preprocessing ?? {}, savgol_window: request?.params?.savgol_window ?? 0, derivatives: request?.params?.derivatives ?? null }
    : stored.dataset_id === dataset.dataset_id
      ? stored
      : {};
  const applicability = new Map((preview?.applicability ?? []).map((a) => [`${a.algorithm}/${a.mode}`, a]));
  const state = {
    container,
    clone,
    caps,
    system,
    dataset,
    applicability,
    previewError,
    prep: {
      preprocessing: prep.preprocessing ?? {},
      savgol: prep.savgol_window ?? 0,
      // "both" is Prepare's preselection of modes, not a value of kd's derivatives parameter.
      derivatives: prep.derivatives === "both" ? null : prep.derivatives ?? null,
      choice: prep.derivatives ?? null,
      alsoRaw: Boolean(prep.also_raw),
    },
    selected: new Set(clone ? [] : initialSelection(prep)),
    seedsText: request ? request.seeds.join(", ") : store.get("search.seeds", "0"),
    sketchText: request?.sketch ? JSON.stringify(request.sketch, null, 1) : "",
    configs: new Map(), // key -> config (form and limits), kept while the page lives
    checks: new Map(), // key -> { sig, result | error }
    cloneKey: null,
    el: {},
  };
  if (clone) {
    const mode = request?.mode ?? firstAcceptedMode(state, clone.algorithm);
    state.cloneKey = mode ? `${clone.algorithm}/${mode}` : null;
    if (mode) state.selected.add(state.cloneKey);
  }
  return state;
}

function firstAcceptedMode(state, algorithm) {
  const info = state.caps.algorithms.find((a) => a.algorithm === algorithm);
  return info?.modes.find((m) => modeBlock(state, info, m) === null)?.name ?? null;
}

// ---------------------------------------------------------------------------
// Layout

function render() {
  const el = S.el;
  el.checkTop = button("Check settings", { onClick: checkAll });
  el.queueTop = button("Queue runs", { kind: "primary", onClick: submitAll });
  el.algos = h("div", { class: "algo-list" });
  el.algoMeta = h("span", { class: "meta" });
  el.configs = h("div", { class: "stack" });
  el.limits = h("div");
  el.sketchTable = h("div");
  el.queue = h("div");
  el.checksOut = h("div", { class: "checks" });
  el.summary = h("div", { class: "txt" });
  el.checkBottom = button("Check settings", { onClick: checkAll });
  el.queueBottom = button("Queue runs", { kind: "primary", onClick: submitAll });
  el.accepted = h("div");

  fill(S.container, 
    pageHead({ title: "Search", sub: subLine(), crumbs: S.clone ? [["Run " + S.clone.run_id, href("result", S.clone.run_id)], ["Edit and rerun", null]] : null, actions: [el.checkTop, el.queueTop] }),
    warningBlock(S.previewError ? [`KD's applicability of the modes could not be read, so no mode is greyed for this data: ${S.previewError.message}`] : []),
    S.clone ? cloneNote() : null,
    h(
      "div",
      { class: "split search-split" },
      h("section", { class: "panel algos", "aria-label": "Algorithms and modes" }, h("header", null, h("h2", null, "Algorithms"), el.algoMeta), el.algos, h("p", { class: "note", style: "margin-top:.7rem" }, "A mode is greyed when KD's declaration refuses this data or the Sketch, with KD's reason. What this machine lacks is a warning; the run can still be queued.")),
      h("div", { class: "stack" }, seedsPanel(), el.configs, panel({ title: "Limits and recording", meta: "applied to each run" }, el.limits), sketchPanel(), panel({ title: "Runs to queue", meta: "one job per algorithm, mode and seed" }, el.queue, el.checksOut), el.accepted, h("div", { class: "summary-bar" }, el.summary, el.checkBottom, el.queueBottom)),
    ),
  );
  update({ structure: true });
}

/** The stored selection, replaced by Prepare's preselected modes when Prepare
 *  changed them since this page last read them. */
function initialSelection(prep) {
  const selection = store.get("search.selection", []);
  if (!prep.modes) return selection;
  const fromPrepare = prep.modes.map((m) => `${m.algorithm}/${m.mode}`);
  if (JSON.stringify(fromPrepare) === JSON.stringify(store.get("search.prepModes", null))) return selection;
  store.set("search.prepModes", fromPrepare);
  return fromPrepare;
}

function subLine() {
  const d = S.dataset;
  const deriv = DERIVATIVE_TEXT[S.prep.choice] ?? (S.prep.choice ? S.prep.choice : "each mode's own derivatives");
  const smoothing = S.prep.preprocessing?.smoothing;
  const smoothText = smoothing ? SMOOTHING_TEXT[smoothing.method] ?? smoothing.method : S.prep.savgol ? `Savitzky-Golay window ${S.prep.savgol}` : "no smoothing";
  const asLoaded = smoothing || S.prep.savgol ? "smoothed before each fit" : "as loaded";
  return h(
    "span",
    null,
    "Data: ",
    h("span", { class: "mono" }, d.label),
    ` ${asLoaded}${S.prep.alsoRaw ? ", and each run again on the data as loaded" : ""}. Prepare: ${deriv}, ${smoothText} `,
    h("a", { href: href("prepare") }, "(change)"),
    ".",
    d.reference_available ? ` Reference equation: ${d.reference_shown ? "shown" : "hidden"}.` : null,
  );
}

function cloneNote() {
  const run = S.clone;
  return h(
    "p",
    { class: "note clone-note" },
    `Settings from run ${run.run_id} (${run.algorithm}, seed ${run.seed ?? "n/a"}). The new runs start from scratch; `,
    run.request ? "the form holds that run's request." : "that run was not started here, so the form holds the parameters of its sealed record.",
  );
}

/** Re-render what depends on the selection, the seeds, the forms and the limits. */
function update({ structure = false } = {}) {
  if (!S) return;
  if (structure) {
    renderAlgos();
    renderConfigs();
    renderLimits();
    renderSketchTable();
    if (!S.clone) store.set("search.selection", [...S.selected]);
  }
  renderQueue();
}

// ---------------------------------------------------------------------------
// Algorithm and mode list

/** Why kd refuses this mode here, or null. */
function modeBlock(state, info, mode) {
  const a = state.applicability.get(`${info.algorithm}/${mode.name}`);
  if (a && !a.accepts) return a.reason ?? "KD's declaration refuses this data.";
  if (!a && !state.previewError && !mode.selectable_from_data) return "This mode is not selectable from input data.";
  const used = sketchClauses(state).used;
  const refused = used.filter((c) => mode.sketch[c] === "unsupported");
  if (refused.length) return `The Sketch uses ${refused.join(", ")}, which this mode declares unsupported.`;
  return null;
}

function renderAlgos() {
  const items = S.caps.algorithms.map((info) => {
    const modes = info.modes.map((mode) => {
      const key = `${info.algorithm}/${mode.name}`;
      const block = modeBlock(S, info, mode);
      const box = h("input", { type: "checkbox", checked: S.selected.has(key) && !block, disabled: block !== null, "aria-label": `${info.algorithm} ${mode.name}` });
      box.addEventListener("change", () => {
        if (box.checked) S.selected.add(key);
        else S.selected.delete(key);
        update({ structure: true });
      });
      return h("label", { class: block ? "mode off" : "mode" }, box, h("span", null, h("span", { class: "mono" }, mode.name), block ? h("span", { class: "why" }, block) : null));
    });
    return h(
      "div",
      { class: "alg" },
      h("div", { class: "name" }, h("b", null, info.algorithm), h("span", { class: "meta" }, `${info.cost_class}, ${info.score_kind} (${info.score_direction})`)),
      info.warnings.length ? h("ul", { class: "alg-warn" }, info.warnings.map((w) => h("li", null, w))) : null,
      h("div", { class: "modes" }, modes),
    );
  });
  fill(S.el.algos, ...items);
  const n = activeKeys().length;
  S.el.algoMeta.textContent = `${n} mode${n === 1 ? "" : "s"} selected`;
}

/** Selected keys that kd does not refuse, in the order of the capability list. */
function activeKeys() {
  const keys = [];
  for (const info of S.caps.algorithms) {
    for (const mode of info.modes) {
      const key = `${info.algorithm}/${mode.name}`;
      if (S.selected.has(key) && modeBlock(S, info, mode) === null) keys.push(key);
    }
  }
  return keys;
}

function lookup(key) {
  const [algorithm, modeName] = key.split("/");
  const info = S.caps.algorithms.find((a) => a.algorithm === algorithm);
  return { info, mode: info.modes.find((m) => m.name === modeName) };
}

// ---------------------------------------------------------------------------
// Seeds

function seedsPanel() {
  const input = h("input", { type: "text", value: S.seedsText, class: "seeds-input", "aria-label": "Seeds" });
  S.el.seedsNote = h("span", { class: "note" });
  input.addEventListener("input", () => {
    S.seedsText = input.value;
    if (!S.clone) store.set("search.seeds", input.value);
    update();
  });
  return panel({ title: "Seeds", meta: "one run per algorithm, mode and seed" }, h("div", { class: "seeds-row" }, h("label", null, "Seeds ", input), S.el.seedsNote));
}

function seeds() {
  try {
    return { seeds: parseSeeds(S.seedsText), error: null };
  } catch (err) {
    return { seeds: [], error: err.message };
  }
}

// ---------------------------------------------------------------------------
// Parameter panels, one per selected algorithm and mode

/** Form fields: the schema's fields without seed and derivatives (set by the
 *  seed list and the mode), plus kd.Model keywords that are not config fields. */
function formFields(info) {
  const key = (f) => f.facade_param || f.name;
  const fields = info.fields.filter((f) => !["seed", "derivatives"].includes(key(f)));
  const taken = new Set(fields.map(key));
  const extra = [];
  for (const p of info.facade_params) {
    if (taken.has(p.name) || !["max_iterations", "answer_selection"].includes(p.effect)) continue;
    const budget = p.effect === "max_iterations";
    extra.push({
      name: p.name,
      json_kind: p.kind,
      default: p.default,
      literal_values: p.literal_values,
      nullable: false,
      required: false,
      settable_from: "facade",
      facade_param: null,
      resume_tier: budget ? "resume_safe" : "not declared",
      knob: budget,
      description: budget ? `Iteration budget in ${info.segmentation.unit} (kd.Model keyword).` : `kd.Model keyword (${p.effect.replace("_", " ")}).`,
      object_fields: null,
    });
  }
  return [...extra, ...fields];
}

/** Parameters the mode and the Prepare step fix: they select the mode or carry
 *  the derivative method and the Savitzky-Golay window. */
function fixedParams(info, mode) {
  const a = S.applicability.get(`${info.algorithm}/${mode.name}`);
  const fixed = { ...mode.selected_by, ...(a?.params ?? {}) };
  for (const [k, v] of Object.entries(fixed)) if (v === SET) delete fixed[k];
  const facade = new Set(info.facade_params.map((p) => p.name));
  if (facade.has("derivatives") && S.prep.derivatives && mode.derivative_providers.includes(S.prep.derivatives)) fixed.derivatives = S.prep.derivatives;
  if (facade.has("savgol_window") && S.prep.savgol) fixed.savgol_window = S.prep.savgol;
  return fixed;
}

function makeConfig(key) {
  const { info, mode } = lookup(key);
  const request = S.clone?.request;
  const initial = S.clone && S.clone.algorithm === info.algorithm ? request?.params ?? S.clone.config ?? {} : {};
  const fixed = fixedParams(info, mode);
  const form = schemaForm(formFields(info), { values: initial });
  for (const tr of form.element.querySelectorAll("tr")) {
    const name = tr.querySelector("td.name")?.textContent;
    if (name in fixed) {
      tr.classList.add("off");
      tr.querySelectorAll("input, select, textarea").forEach((c) => (c.disabled = true));
      tr.querySelector("td.desc").prepend(h("span", { class: "fixed-by" }, "Set by the mode. "));
    }
  }
  form.onChange(() => update());
  const limits = {
    capGiB: request?.memory_cap_bytes ? request.memory_cap_bytes / GIB : null,
    maxMinutes: request?.max_seconds ? request.max_seconds / 60 : null,
    hardMinutes: request?.hard_seconds ? request.hard_seconds / 60 : null,
    checkpointEvery: request?.checkpoint_every ?? null,
    device: request?.device ?? "cpu",
  };
  return { key, info, mode, fixed, form, limits, isClone: key === S.cloneKey, panel: configPanel(info, mode, fixed, form) };
}

function configPanel(info, mode, fixed, form) {
  const a = S.applicability.get(`${info.algorithm}/${mode.name}`);
  const needs = [...new Set([...mode.required_params, ...Object.entries(a?.params ?? {}).filter(([, v]) => v === SET).map(([k]) => k)])];
  const pythonOnly = new Set(info.fields.filter((f) => f.settable_from === "python" || f.json_kind === null).map((f) => f.facade_param || f.name));
  const generations = info.facade_params.find((p) => p.name === "generations");
  const facts = [
    h("span", null, "Native score ", h("b", null, info.score_kind), info.score_direction === "min" ? ", lower is better" : ", higher is better"),
    h("span", null, "Cost ", h("b", null, info.cost_class)),
    h("span", null, "Derivatives ", h("b", null, fixed.derivatives ?? mode.provider_kind ?? "none")),
    info.one_shot ? h("span", null, "One-shot fit") : h("span", null, "Progress unit ", h("b", null, info.segmentation.unit)),
    fixed.savgol_window ? h("span", null, "Savitzky-Golay window ", h("b", null, fixed.savgol_window)) : null,
  ];
  const notes = [
    h("p", { class: "note" }, info.summary, generations?.effect === "unused" ? [" ", h("code", null, "generations"), " is not used by this algorithm."] : null),
    mode.description ? h("p", { class: "note" }, `Mode ${mode.name}: ${mode.description}`) : null,
    S.prep.derivatives && mode.derivative_providers.length && !mode.derivative_providers.includes(S.prep.derivatives)
      ? h("p", { class: "note" }, `Prepare chose ${S.prep.derivatives}; this mode takes derivatives by ${mode.derivative_providers.join(" or ")}, so it uses ${fixed.derivatives ?? mode.provider_kind}.`)
      : null,
    needs.length ? h("p", { class: "note" }, "This mode needs ", needs.map((n, i) => [i ? ", " : null, h("code", null, n), pythonOnly.has(n) ? " (set from Python only)" : null]), ".") : null,
    Object.keys(mode.refused_fields).length ? h("p", { class: "note" }, "This mode refuses a changed value of ", Object.keys(mode.refused_fields).map((n, i) => [i ? ", " : null, h("code", null, n)]), ".") : null,
  ];
  const warn = h("div", { class: "cfg-warn" });
  const el = h(
    "section",
    { class: "panel cfg" },
    h("div", { class: "algo-head" }, h("h2", null, info.algorithm, " ", h("span", { class: "mono mode-name" }, `${mode.name} mode`)), h("div", { class: "facts" }, facts)),
    h("div", { class: "cfg-notes" }, notes),
    warn,
    form.element,
  );
  el.warn = warn;
  return el;
}

function renderConfigs() {
  const keys = activeKeys();
  for (const key of keys) if (!S.configs.has(key)) S.configs.set(key, makeConfig(key));
  fill(S.el.configs, ...keys.map((k) => S.configs.get(k).panel));
  if (!keys.length) fill(S.el.configs, panel({ title: "Parameters" }, h("p", { class: "empty" }, "Select at least one mode on the left. Each selected mode gets its own parameter table here.")));
}

/** params of FitRequest: the changed form values and the fixed ones. */
function paramsOf(cfg) {
  return { ...cfg.form.changes(), ...cfg.fixed };
}

function refusedChanges(cfg) {
  const changed = cfg.form.changes();
  return Object.keys(cfg.mode.refused_fields).filter((k) => k in changed);
}

// ---------------------------------------------------------------------------
// Limits and recording

function renderLimits() {
  const cfgs = activeKeys().map((k) => S.configs.get(k));
  if (!cfgs.length) {
    fill(S.el.limits, h("p", { class: "empty" }, "The limits of each selected mode appear here."));
    return;
  }
  const mem = S.system.memory;
  const env = S.caps.environment;
  const head = h("thead", null, h("tr", null, h("th", null, "Setting"), cfgs.map((c) => h("th", null, `${c.info.algorithm}, ${c.mode.name}`)), h("th", null, "What happens")));
  const row = (label, cell, desc) => h("tr", null, h("td", null, label), cfgs.map((c) => h("td", null, cell(c))), h("td", { class: "desc" }, desc));
  const number = (c, prop, { placeholder, disabled, unit }) => {
    const input = h("input", { type: "text", class: "num", inputmode: "decimal", value: c.limits[prop] ?? "", placeholder, disabled, "aria-label": `${prop} ${c.key}` });
    input.addEventListener("input", () => {
      c.limits[prop] = input.value.trim() === "" ? null : Number(input.value);
      update();
    });
    return [input, " ", unit];
  };
  const gpus = env.gpus.map((g) => ({ value: `cuda:${g.index}`, label: `cuda:${g.index} (${g.name})` }));
  const device = (c) => {
    const select = h("select", { "aria-label": `device ${c.key}` }, [{ value: "cpu", label: "cpu" }, ...gpus].map((o) => h("option", { value: o.value, selected: c.limits.device === o.value }, o.label)));
    select.addEventListener("change", () => {
      c.limits.device = select.value;
      update();
    });
    return select;
  };
  const body = h(
    "tbody",
    null,
    row("Memory cap", (c) => number(c, "capGiB", { placeholder: (mem.default_fit_cap_bytes / GIB).toFixed(1), unit: "GiB" }), `The run is ended when its processes use more than the cap; other applications are not affected. Empty takes the default that fits this machine now, ${fmtBytes(mem.default_fit_cap_bytes)}.`),
    row("Time budget", (c) => number(c, "maxMinutes", { placeholder: c.info.one_shot ? "n/a" : "none", disabled: c.info.one_shot, unit: "min" }), ["KD's soft budget (", h("code", null, "max_seconds"), "): the run stops at the end of an iteration and its result is sealed normally. A one-shot fit has no iterations, so the budget does not apply."]),
    row("Hard time limit", (c) => number(c, "hardMinutes", { placeholder: "auto", unit: "min" }), `The supervisor ends the run's processes when this time has passed since it started. No equation is sealed; the checkpoints written so far stay. Empty: the time budget plus ${HARD_SLACK_MINUTES} min, or no limit without a time budget.`),
    row("Checkpoints", (c) => (c.info.one_shot ? h("span", { class: "note" }, "not available") : number(c, "checkpointEvery", { placeholder: "default", unit: c.info.segmentation.unit })), "Iterations between checkpoints; empty takes the server's default. Needed to resume after a stop or a machine restart. A one-shot fit cannot resume."),
    row("Device", device, env.cuda_available ? "Where the fit computes." : "No CUDA device was found; fits run on the CPU."),
    h("tr", null, h("td", null, "Machine"), h("td", { colspan: cfgs.length }, h("select", { disabled: true }, h("option", null, "this machine"))), h("td", { class: "desc" }, "Runs and their files stay on this machine.")),
  );
  S.el.limitsWarn = h("div");
  fill(S.el.limits, 
    env.memory_cap_kind === "sampled" ? warningBlock([`The cap is enforced by sampling the run's memory, so a run can pass it briefly before it is ended. ${env.memory_cap_reason ?? ""}`.trim()]) : null,
    h("div", { class: "tablebox" }, h("table", { class: "t params limits" }, head, body)),
    S.el.limitsWarn,
  );
}

function capBytes(cfg) {
  const giB = cfg.limits.capGiB;
  return giB === null || !Number.isFinite(giB) || giB <= 0 ? null : Math.round(giB * GIB);
}

function limitProblems(cfg) {
  const out = [];
  const bad = (v) => v !== null && (!Number.isFinite(v) || v <= 0);
  if (bad(cfg.limits.capGiB)) out.push(`${cfg.key}: the memory cap must be a positive number of GiB.`);
  if (bad(cfg.limits.maxMinutes)) out.push(`${cfg.key}: the time budget must be a positive number of minutes.`);
  if (bad(cfg.limits.hardMinutes)) out.push(`${cfg.key}: the hard time limit must be a positive number of minutes.`);
  if (bad(cfg.limits.checkpointEvery) || (cfg.limits.checkpointEvery !== null && !Number.isInteger(cfg.limits.checkpointEvery))) out.push(`${cfg.key}: checkpoints need a whole number of ${cfg.info.segmentation.unit}.`);
  return out;
}

// ---------------------------------------------------------------------------
// Sketch

function sketchPanel() {
  const area = h("textarea", { class: "sketch-input mono", rows: 5, placeholder: '{"schema": "kd-sketch-v2", "pinned": [...], "anchored": [...], "holes": [...]}', "aria-label": "Sketch document" }, S.sketchText);
  S.el.sketchState = h("span");
  area.addEventListener("input", () => {
    S.sketchText = area.value;
    update({ structure: true });
  });
  return h(
    "section",
    { class: "panel" },
    h(
      "details",
      { open: true },
      h("summary", null, "Sketch: terms known in advance ", S.el.sketchState),
      h("p", { class: "note", style: "margin:.4rem 0 .6rem" }, "A Sketch pins terms with fixed coefficients, requires anchor terms, or bounds the number of free terms. Paste a KD sketch document; KD checks it in the pre-launch check. How each selected mode enforces each kind of clause:"),
      S.el.sketchTable,
      h("p", { class: "note", style: "margin:.5rem 0" }, "A mode whose entry is ", h("code", null, "unsupported"), " is disabled in the algorithm list as soon as the Sketch uses a clause of that kind."),
      area,
    ),
  );
}

/** The sketch document and the clause kinds it uses (the keys kd reads:
 *  pinned, anchored, holes and the constraints of the holes). */
function sketchClauses(state) {
  const text = state.sketchText.trim();
  if (!text) return { doc: null, used: [], error: null };
  let doc;
  try {
    doc = JSON.parse(text);
  } catch (err) {
    return { doc: null, used: [], error: `The Sketch is not valid JSON: ${err.message}` };
  }
  if (doc === null || typeof doc !== "object" || Array.isArray(doc)) return { doc: null, used: [], error: "The Sketch must be a JSON object." };
  const used = [];
  const holes = Array.isArray(doc.holes) ? doc.holes : [];
  const constraints = holes.map((hole) => hole?.constraint ?? {});
  if (Array.isArray(doc.pinned) && doc.pinned.length) used.push("fixed_terms");
  if (Array.isArray(doc.anchored) && doc.anchored.length) used.push("anchors");
  if (holes.length) used.push("hole_count");
  if (constraints.some((c) => c.max_deriv_order != null)) used.push("derivative_order");
  if (constraints.some((c) => c.operators != null)) used.push("operator_set");
  if (constraints.some((c) => c.fields != null || c.axes != null)) used.push("field_axis_set");
  return { doc, used, error: null };
}

function renderSketchTable() {
  const { used, error, doc } = sketchClauses(S);
  S.el.sketchState.textContent = error ? "(not readable)" : doc ? `(uses ${used.length ? used.join(", ") : "no clause"})` : "(none set)";
  const keys = S.caps.algorithms.flatMap((info) => info.modes.map((m) => `${info.algorithm}/${m.name}`)).filter((k) => S.selected.has(k));
  if (!keys.length) {
    fill(S.el.sketchTable, h("p", { class: "empty" }, "Select a mode to see how it enforces each clause."));
    return;
  }
  const columns = [{ label: "Clause", render: (c) => h("span", { class: used.includes(c) ? "used" : null }, c), className: "mono" }];
  for (const k of keys) {
    const { info, mode } = lookup(k);
    columns.push({ label: `${info.algorithm}, ${mode.name}`, render: (c) => h("span", { class: mode.sketch[c] === "unsupported" ? "unsupported" : null }, mode.sketch[c]) });
  }
  fill(S.el.sketchTable, table(columns, SKETCH_CLAUSES, { className: "sk" }), error ? h("p", { class: "field-error" }, error) : null);
}

// ---------------------------------------------------------------------------
// Requests

function preflightRequest(cfg) {
  return {
    dataset_id: S.dataset.dataset_id,
    algorithm: cfg.info.algorithm,
    mode: cfg.mode.name,
    params: paramsOf(cfg),
    device: cfg.limits.device,
    preprocessing: S.prep.preprocessing,
    sketch: sketchClauses(S).doc,
    schema_digest: S.caps.schema_digest,
  };
}

function fitRequest(cfg, seedList) {
  const { dataset_id, algorithm, mode, params, device, preprocessing, sketch, schema_digest } = preflightRequest(cfg);
  return {
    kind: "fit",
    dataset_id,
    algorithm,
    mode,
    params,
    seeds: seedList,
    device,
    preprocessing,
    sketch,
    max_seconds: cfg.info.one_shot || cfg.limits.maxMinutes === null ? null : cfg.limits.maxMinutes * 60,
    hard_seconds: cfg.limits.hardMinutes === null ? null : cfg.limits.hardMinutes * 60,
    memory_cap_bytes: capBytes(cfg),
    checkpoint_every: cfg.info.one_shot ? null : cfg.limits.checkpointEvery,
    schema_digest,
    note: S.clone ? `edit and rerun of ${S.clone.run_id}` : null,
  };
}

/** Prepare's "also queue each run on the data as loaded" (not for edit and rerun). */
const alsoRaw = (cfg) => S.prep.alsoRaw && !cfg.isClone;

/** The same fit on the data as loaded: no smoothing and no Savitzky-Golay window. */
function rawRequest(body) {
  const params = { ...body.params };
  delete params.savgol_window;
  return { ...body, params, preprocessing: { smoothing: null }, note: "on the data as loaded" };
}

/** What stops the request from being built at all (unreadable inputs). */
function inputProblems(cfgs) {
  const problems = [];
  const { error } = seeds();
  if (error) problems.push(`Seeds: ${error}`);
  const sketch = sketchClauses(S);
  if (sketch.error) problems.push(sketch.error);
  for (const cfg of cfgs) {
    for (const e of cfg.form.errors()) problems.push(`${cfg.info.algorithm}, ${cfg.mode.name}: ${e.key}: ${e.message}`);
    problems.push(...limitProblems(cfg));
  }
  return problems;
}

/** Run kd's pre-launch check of every selected configuration; cached per request. */
async function preflightAll(cfgs) {
  await Promise.all(
    cfgs.map(async (cfg) => {
      const body = preflightRequest(cfg);
      const sig = JSON.stringify(body);
      if (S.checks.get(cfg.key)?.sig === sig) return;
      try {
        S.checks.set(cfg.key, { sig, result: await post("/preflight", body), error: null });
      } catch (err) {
        if (!(err instanceof ApiError)) throw err;
        S.checks.set(cfg.key, { sig, result: null, error: err });
      }
    }),
  );
}

async function checkAll() {
  const cfgs = activeKeys().map((k) => S.configs.get(k));
  const problems = inputProblems(cfgs);
  if (!cfgs.length) throw new Error("Select at least one mode.");
  if (problems.length) throw new Error(problems[0]);
  await preflightAll(cfgs);
  if (!S) return;
  update();
  const refused = cfgs.filter((c) => S.checks.get(c.key).result?.ok === false);
  toast(refused.length ? `KD refuses ${refused.length} of ${cfgs.length} configurations; see "Runs to queue".` : `KD accepts ${cfgs.length === 1 ? "the configuration" : `all ${cfgs.length} configurations`}.`, refused.length ? "error" : null);
}

async function submitAll() {
  const cfgs = activeKeys().map((k) => S.configs.get(k));
  if (!cfgs.length) throw new Error("Select at least one mode.");
  const problems = inputProblems(cfgs);
  if (problems.length) throw new Error(problems[0]);
  await preflightAll(cfgs);
  if (!S) return;
  update();
  const refused = cfgs.filter((c) => S.checks.get(c.key).result?.ok === false);
  if (refused.length) throw new Error(`KD refuses ${refused.map((c) => `${c.info.algorithm}, ${c.mode.name}`).join("; ")}. Nothing was queued.`);
  const seedList = seeds().seeds;
  const accepted = [];
  const failed = [];
  for (const cfg of cfgs) {
    const path = cfg.isClone ? `/runs/${encodeURIComponent(S.clone.run_id)}/clone` : "/jobs";
    const bodies = [fitRequest(cfg, seedList)];
    if (alsoRaw(cfg)) bodies.push(rawRequest(bodies[0]));
    for (const body of bodies) {
      try {
        const answer = await post(path, body);
        accepted.push(...answer.jobs);
      } catch (err) {
        if (!(err instanceof ApiError)) throw err;
        failed.push({ cfg, err });
      }
    }
  }
  if (!S) return;
  renderAccepted(accepted, failed);
  toast(failed.length ? `${accepted.length} runs queued; ${failed.length} configuration${failed.length === 1 ? "" : "s"} not queued.` : `${accepted.length} run${accepted.length === 1 ? "" : "s"} queued.`, failed.length ? "error" : null);
}

function renderAccepted(jobs, failed) {
  const rows = jobs.map((job) => ({ job }));
  fill(S.el.accepted, 
    panel(
      { title: "Queued", meta: `${jobs.length} job${jobs.length === 1 ? "" : "s"}`, actions: [linkButton("All runs", href("jobs"), { small: true })] },
      failed.map(({ cfg, err }) => h("div", { class: "refusal" }, h("b", null, `${cfg.info.algorithm}, ${cfg.mode.name} was not queued: `), err.message, err.field ? ` (field ${err.field})` : "")),
      jobs.length
        ? table(
            [
              { label: "Job", render: ({ job }) => h("a", { href: href("run", job.job_id), class: "mono" }, job.job_id) },
              { label: "Algorithm", render: ({ job }) => job.algorithm },
              { label: "Seed", num: true, render: ({ job }) => job.seed },
              { label: "Memory cap", num: true, render: ({ job }) => fmtBytes(job.limits.memory_cap_bytes) },
              { label: "State", render: ({ job }) => (job.waiting ? `${job.state}: ${job.waiting.message}` : job.state) },
            ],
            rows,
          )
        : null,
      jobs.flatMap((j) => j.warnings).length ? warningBlock([...new Set(jobs.flatMap((j) => j.warnings))]) : null,
    ),
  );
  S.el.accepted.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------------------
// Runs to queue, check results and the summary bar

function renderQueue() {
  const cfgs = activeKeys().map((k) => S.configs.get(k));
  const { seeds: seedList, error } = seeds();
  const fitQueue = S.system.queues.find((q) => q.queue === "fit") ?? { slots: 1, running: 0, queued: 0 };
  const ahead = fitQueue.running + fitQueue.queued;
  const rows = [];
  for (const cfg of cfgs) {
    for (const seed of seedList) rows.push({ cfg, seed, raw: false });
    if (alsoRaw(cfg)) for (const seed of seedList) rows.push({ cfg, seed, raw: true });
  }
  const starts = (i) => {
    if (i === 0) return ahead < fitQueue.slots ? "now" : fitQueue.queued ? `after ${fitQueue.queued} queued run${fitQueue.queued === 1 ? "" : "s"}` : "after the running run ends";
    return ahead + i < fitQueue.slots ? "now" : `after run ${i}`;
  };
  const defaultCap = S.system.memory.default_fit_cap_bytes;
  fill(S.el.queue, 
    table(
      [
        { label: "#", render: (_, i) => i + 1 },
        { label: "Algorithm", render: (r) => r.cfg.info.algorithm },
        { label: "Mode", render: (r) => h("span", { class: "mono" }, r.cfg.mode.name) },
        { label: "Seed", num: true, render: (r) => r.seed },
        S.prep.alsoRaw ? { label: "Data", render: (r) => (r.raw ? "as loaded" : "smoothed") } : null,
        { label: "Memory cap", num: true, render: (r) => fmtBytes(capBytes(r.cfg) ?? defaultCap) },
        { label: "Starts", render: (_, i) => starts(i) },
      ],
      rows,
      { empty: error ? `Seeds: ${error}` : "Select a mode and give at least one seed." },
    ),
  );
  S.el.seedsNote.textContent = error ? error : `${seedList.length} seed${seedList.length === 1 ? "" : "s"} for each selected mode. Ranges such as 0-4 are accepted.`;
  S.el.seedsNote.className = error ? "field-error" : "note";
  renderConfigWarnings(cfgs);
  renderChecks(cfgs);
  renderSummary(rows.length, fitQueue, inputProblems(cfgs));
}

function renderConfigWarnings(cfgs) {
  const mem = S.system.memory;
  for (const cfg of cfgs) {
    const refused = refusedChanges(cfg);
    fill(cfg.panel.warn, warningBlock(refused.length ? [`Mode ${cfg.mode.name} declares that it refuses a changed ${refused.join(", ")}; KD will refuse this configuration unless it keeps the default.`] : []));
  }
  const needs = cfgs
    .map((c) => ({ c, need: (capBytes(c) ?? mem.default_fit_cap_bytes) + mem.supervisor_allowance_bytes }))
    .filter(({ need }) => need > mem.admissible_bytes)
    .map(({ c, need }) => `${c.info.algorithm}, ${c.mode.name}: needs ${fmtBytes(need)} with the supervisor; ${fmtBytes(mem.admissible_bytes)} can start beside the running jobs now. The run waits until memory frees, or starts with a warning when nothing runs ahead of it.`);
  if (S.el.limitsWarn) fill(S.el.limitsWarn, warningBlock(needs));
}

function renderChecks(cfgs) {
  const items = [];
  for (const cfg of cfgs) {
    const check = S.checks.get(cfg.key);
    const label = `${cfg.info.algorithm}, ${cfg.mode.name}`;
    if (!check || check.sig !== JSON.stringify(preflightRequest(cfg))) continue;
    if (check.error) {
      items.push(warningBlock([`${label}: the pre-launch check could not run (${check.error.message}). The run can still be queued; a settings error then appears in the run's status and log.`]));
      continue;
    }
    const r = check.result;
    if (r.ok) items.push(h("p", { class: "ok-line" }, `${label}: accepted by KD`, r.resolved_mode && r.resolved_mode !== cfg.mode.name ? ` (KD selects mode ${r.resolved_mode})` : "", "."));
    for (const refusal of r.refusals) items.push(h("div", { class: "refusal", role: "alert" }, h("b", null, `${label}, refused by KD: `), refusal.message, refusal.field ? h("span", { class: "mono field" }, ` ${refusal.field}`) : null));
    if (r.warnings.length) items.push(warningBlock(r.warnings.map((w) => `${label}: ${w}`)));
  }
  fill(S.el.checksOut, ...items);
}

function renderSummary(n, fitQueue, problems) {
  const label = `Queue ${n} run${n === 1 ? "" : "s"}`;
  for (const b of [S.el.queueTop, S.el.queueBottom]) {
    b.textContent = label;
    b.disabled = n === 0;
  }
  for (const b of [S.el.checkTop, S.el.checkBottom]) b.disabled = n === 0;
  const ahead = [fitQueue.running ? `${fitQueue.running} running` : null, fitQueue.queued ? `${fitQueue.queued} queued` : null].filter(Boolean).join(" and ");
  fill(S.el.summary, 
    h("b", null, `${n} run${n === 1 ? "" : "s"}`),
    ` go to the queue${ahead ? ` after ${ahead} run${fitQueue.running + fitQueue.queued === 1 ? "" : "s"}` : ""}. Fits at a time: ${fitQueue.slots}. Each run can be stopped, resumed or rerun on its own.`,
    problems.length ? h("span", { class: "field-error" }, problems[0]) : null,
  );
}

