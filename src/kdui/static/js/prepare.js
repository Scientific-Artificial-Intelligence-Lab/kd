// Prepare page (design 2.3, mockup prepare.html): the derivative method, which
// only preselects modes on the Search page because kd ties the provider to the
// mode, and the smoothing (none, Savitzky-Golay window, low-pass, network).
// The noise estimate, the low-pass spectrum and the before and after views of
// the field and of its left-hand-side derivative are the answer of
// POST /preprocess-preview; the page draws them and computes nothing.
// The choices are written to store key "preprocessing" on every change.

import { get, post, store } from "./api.js";
import { button, emptyState, errorState, fmtNumber, h, href, kv, linkButton, loadingState, pageHead, panel, table, tabs, warningBlock } from "./ui.js";
import { renderTex } from "./math.js";
import { disposeWithin, fieldPlot, linePlot } from "./plots.js";

// Page styles of the mockup that app.css does not carry, scoped to this page.

const KIND_TEXT = {
  finite_diff: "Stencils on the grid. Nothing is trained.",
  autograd: "Trains a network on the field first, then differentiates it. Uses the GPU when one is found. Each mode has its own training settings on the Search page.",
};
const KIND_LABEL = { finite_diff: "finite differences", autograd: "autograd", both: "both" };

let page = null;

export async function mount(container, params) {
  const stored = store.get("dataset");
  const datasetId = params.id || (stored && stored.dataset_id);
  if (!datasetId) {
    container.append(
      pageHead({ title: "Prepare", sub: "Choose how derivatives are taken and whether the data are smoothed first." }),
      emptyState("No dataset is chosen yet. Choose one on the Data page first.", linkButton("Choose a dataset", href("data"), { kind: "primary" })),
    );
    return;
  }
  const waiting = loadingState("Reading the dataset and the algorithms");
  container.append(waiting);
  let preview;
  let caps;
  try {
    [preview, caps] = await Promise.all([get(`/datasets/${encodeURIComponent(datasetId)}/preview`), get("/capabilities")]);
  } catch (err) {
    if (!container.isConnected) return;
    const retry = () => {
      container.replaceChildren();
      return mount(container, params);
    };
    container.replaceChildren(
      pageHead({ title: "Prepare", sub: "Choose how derivatives are taken and whether the data are smoothed first." }),
      errorState(err, { title: "KD could not read this dataset.", retry }),
      h("p", { class: "note" }, h("a", { href: href("data") }, "Choose another dataset on the Data page.")),
    );
    return;
  }
  waiting.remove();
  if (!container.isConnected) return;
  const label = stored && stored.dataset_id === datasetId ? stored.label : preview.report.name;
  page = createState(datasetId, label, preview, caps.snapshot);
  build(container, page);
  saveChoices(page);
  requestPreview(page);
}

export function unmount() {
  if (!page) return;
  page.alive = false;
  for (const ctl of [page.previewAbort, page.autogradAbort]) if (ctl) ctl.abort();
  page = null;
}

// ---------------------------------------------------------------------------
// State

function createState(datasetId, label, preview, snapshot) {
  const report = preview.report;
  const saved = store.get("preprocessing");
  const form = saved && saved.dataset_id === datasetId && saved.form ? saved.form : null;
  const lowpassAxes = (report.axes || []).filter((a) => a.name !== report.lhs_axis);
  const defaults = {
    kind: "none",
    derivatives: "finite_diff",
    savgol: 0,
    modes: Object.fromEntries(lowpassAxes.map((a) => [a.name, a.n])),
    timeWindow: 0,
    seed: 0,
    alsoRaw: false,
  };
  return {
    alive: true,
    datasetId,
    label,
    preview,
    report,
    snapshot,
    lowpassAxes,
    form: { ...defaults, ...(form || {}), modes: { ...defaults.modes, ...((form && form.modes) || {}) } },
    tab: "loaded",
    seq: 0,
    result: null,
    resultError: null,
    loading: false,
    autograd: null,
    autogradError: null,
    autogradLoading: false,
    nodes: {},
  };
}

/** The Preprocessing model of a fit request for the current form. */
function preprocessingOf(form) {
  if (form.kind === "lowpass") return { smoothing: { method: "lowpass", modes: { ...form.modes }, time_window: form.timeWindow } };
  if (form.kind === "network") return { smoothing: { method: "network", seed: form.seed } };
  return { smoothing: null };
}

const savgolOf = (form) => (form.kind === "savgol" ? form.savgol : 0);

/** Accepted modes of this dataset with their provider kind (from the capability snapshot). */
function acceptedModes(st) {
  const kinds = new Map();
  for (const alg of st.snapshot ? st.snapshot.algorithms : []) {
    for (const mode of alg.modes) kinds.set(`${alg.algorithm}/${mode.name}`, mode.provider_kind);
  }
  return st.preview.applicability
    .filter((a) => a.accepts)
    .map((a) => ({ algorithm: a.algorithm, mode: a.mode, kind: kinds.get(`${a.algorithm}/${a.mode}`) || "unknown" }));
}

function preselected(st) {
  const wanted = st.form.derivatives;
  return acceptedModes(st).filter((m) => wanted === "both" ? m.kind === "finite_diff" || m.kind === "autograd" : m.kind === wanted);
}

/** Store key "preprocessing", read by the Search page. */
function saveChoices(st) {
  store.set("preprocessing", {
    dataset_id: st.datasetId,
    preprocessing: preprocessingOf(st.form),
    savgol_window: savgolOf(st.form),
    derivatives: st.form.derivatives,
    also_raw: st.form.kind !== "none" && st.form.alsoRaw,
    modes: preselected(st).map(({ algorithm, mode }) => ({ algorithm, mode })),
    form: st.form,
  });
}

// ---------------------------------------------------------------------------
// Requests

function previewBody(st, derivatives) {
  const field = st.report.lhs_field || (st.preview.slices[0] && st.preview.slices[0].field);
  const region = st.preview.slices.find((s) => s.field === field) || st.preview.slices[0];
  return { dataset_id: st.datasetId, preprocessing: preprocessingOf(st.form), savgol_window: savgolOf(st.form), derivatives, region };
}

async function requestPreview(st) {
  if (st.previewAbort) st.previewAbort.abort();
  const ctl = new AbortController();
  st.previewAbort = ctl;
  const seq = ++st.seq;
  st.loading = true;
  if (st.resultError) st.resultError = null;
  if (st.result) renderStatus(st);
  else renderPreviewParts(st);
  try {
    const result = await post("/preprocess-preview", previewBody(st, "finite_diff"), { signal: ctl.signal });
    if (!st.alive || seq !== st.seq) return;
    st.result = result;
    st.resultError = null;
  } catch (err) {
    if (err.name === "AbortError" || !st.alive || seq !== st.seq) return;
    st.resultError = err;
  }
  st.loading = false;
  renderPreviewParts(st);
}

async function requestAutograd(st) {
  if (st.autograd || st.autogradLoading) return;
  const ctl = new AbortController();
  st.autogradAbort = ctl;
  st.autogradLoading = true;
  st.autogradError = null;
  renderTarget(st);
  try {
    st.autograd = await post("/preprocess-preview", previewBody(st, "autograd"), { signal: ctl.signal });
  } catch (err) {
    if (err.name === "AbortError" || !st.alive) return;
    st.autogradError = err;
  }
  st.autogradLoading = false;
  if (st.alive) renderTarget(st);
}

/** A form change: save it, redraw what depends on it, and ask for a new preview. */
function changed(st, { preview = true } = {}) {
  saveChoices(st);
  renderDerivatives(st);
  renderSummary(st);
  syncSmoothing(st);
  if (preview) requestPreview(st);
}

// ---------------------------------------------------------------------------
// Layout

function build(container, st) {
  const name = st.label;
  const n = st.nodes;
  n.deriv = h("div");
  n.smoothing = buildSmoothing(st);
  n.target = h("div", { class: "preview-state" });
  n.field = h("div", { class: "preview-state" });
  n.noise = h("div");
  n.spectrum = h("div");
  n.summary = h("div");
  n.warnings = h("div");
  n.status = h("div");
  const lhsTex = st.report.lhs_label ? h("span") : null;
  if (lhsTex) renderTex(lhsTex, lhsLatex(st));
  container.append(
    pageHead({
      title: "Prepare",
      sub: `How KD takes derivatives of ${name}, and whether the field is smoothed before the search. Both choices are passed to every run started from Search.`,
      actions: [linkButton("Continue to Search", href("search"), { kind: "primary" })],
    }),
    n.warnings,
    n.status,
    h(
      "div",
      { class: "split-wide" },
      h(
        "div",
        { class: "stack" },
        panel({ title: "Derivatives", meta: "chosen with the mode" }, n.deriv),
        panel({ title: "Smoothing", meta: "applied to the data before the search" }, n.smoothing),
        panel({ title: lhsTex ? h("span", null, "Target ", lhsTex) : "Target", meta: "what the searches fit, with the choices above" }, n.target),
        panel({ title: "Field, before and after", meta: "the same region as loaded and after the smoothing" }, n.field),
      ),
      h(
        "div",
        { class: "stack" },
        sidePanel("Noise and step gaps", "kd.preview_report", n.noise),
        sidePanel("Energy left out by the lowest m modes", "kd.lowpass_spectrum", n.spectrum),
        sidePanel("Runs will use", null, n.summary),
      ),
    ),
  );
  renderDerivatives(st);
  syncSmoothing(st);
  renderSummary(st);
  renderPreviewParts(st);
}

function sidePanel(title, meta, body) {
  return h("section", { class: "panel" }, h("header", null, h("h3", null, title), meta && h("span", { class: "meta" }, meta)), body);
}

function lhsLatex(st) {
  const r = st.report;
  if (!r.lhs_field || !r.lhs_axis) return r.lhs_label || "";
  return r.lhs_order > 1 ? `${r.lhs_field}_{${r.lhs_axis.repeat(r.lhs_order)}}` : `${r.lhs_field}_{${r.lhs_axis}}`;
}

// ---------------------------------------------------------------------------
// Derivatives

function renderDerivatives(st) {
  const accepted = acceptedModes(st);
  const kinds = [...new Set(accepted.map((m) => m.kind))].sort((a, b) => (KIND_TEXT[b] ? 1 : 0) - (KIND_TEXT[a] ? 1 : 0));
  const wanted = st.form.derivatives;
  const isPre = (kind) => (wanted === "both" ? kind === "finite_diff" || kind === "autograd" : kind === wanted);
  const seg = h(
    "div",
    { class: "seg", role: "radiogroup", "aria-label": "Preselect modes", style: "margin-bottom:.8rem" },
    ["finite_diff", "autograd", "both"].map((choice) =>
      h("button", { type: "button", "aria-pressed": String(choice === wanted), onClick: () => { st.form.derivatives = choice; changed(st, { preview: false }); } }, KIND_LABEL[choice]),
    ),
  );
  const rows = kinds.map((kind) => ({ kind, modes: accepted.filter((m) => m.kind === kind) }));
  const refused = st.preview.applicability.filter((a) => !a.accepts).length;
  st.nodes.deriv.replaceChildren(
    h("p", { class: "note", style: "margin:0 0 .7rem" }, "In KD the derivative provider belongs to the mode: each mode declares one, and a mode owned by a plugin refuses another. The choice here preselects modes on the Search page."),
    seg,
    table(
      [
        { label: "Provider", render: (r) => h("code", null, r.kind) },
        { label: `Modes that accept ${st.label}`, render: (r) => h("span", { class: "mono", style: "font-size:.84rem" }, r.modes.map((m) => `${m.algorithm} ${m.mode}`).join(", ")) },
        { label: "What it does", render: (r) => KIND_TEXT[r.kind] || "Declared by the mode." },
      ],
      rows,
      { className: "modes", rowClass: (r) => (isPre(r.kind) ? "pre" : null), empty: "No mode accepts this dataset." },
    ),
    refused
      ? h("p", { class: "note", style: "margin:.5rem 0 0" }, `${refused} other modes do not accept ${st.label}; `, h("a", { href: href("data", st.datasetId) }, "the Data page"), " gives KD's reasons.")
      : null,
  );
}

// ---------------------------------------------------------------------------
// Smoothing

function smoothingRefusals(st) {
  const r = st.report;
  const grid = r.topology === "grid";
  const fields = (r.fields || []).length;
  return {
    lowpass: grid ? null : `kd.lowpass_smooth takes grid data only; ${st.label} is ${r.topology} data.`,
    network: !grid ? `kd.network_smooth takes grid data only; ${st.label} is ${r.topology} data.` : fields > 1 ? `kd.network_smooth takes a single field; ${st.label} has ${fields}.` : null,
  };
}

function intInput(value, onValue, label) {
  const input = h("input", { type: "number", class: "num", step: "1", value: String(value), "aria-label": label });
  input.addEventListener("change", () => {
    const v = Number(input.value);
    if (!Number.isInteger(v)) {
      input.setCustomValidity("Enter a whole number.");
      input.reportValidity();
      return;
    }
    input.setCustomValidity("");
    onValue(v);
  });
  return input;
}

function option(st, kind, nameNodes, what, knobs, refusal) {
  const radio = h("input", { type: "radio", name: "kd-smoothing", value: kind, "aria-label": kind, disabled: Boolean(refusal) });
  radio.addEventListener("change", () => {
    st.form.kind = kind;
    changed(st);
  });
  const name = h("span", { class: "name", onClick: () => !refusal && radio.click() }, nameNodes);
  const li = h("li", { "data-kind": kind, class: refusal ? "off" : null }, radio, name, h("span", { class: "what" }, what), knobs && h("span", { class: "knobs" }, knobs), refusal && h("span", { class: "why" }, refusal));
  return li;
}

function buildSmoothing(st) {
  const f = st.form;
  const refusals = smoothingRefusals(st);
  const lowAxes = st.lowpassAxes;
  const modeKnobs = lowAxes.map((a) => h("label", null, `modes along ${a.name} `, intInput(f.modes[a.name], (v) => { f.modes[a.name] = v; changed(st, { preview: f.kind === "lowpass" }); }, `modes along ${a.name}`), ` of ${a.n}`));
  const spectralText = lowAxes.map((a) => `${a.name} (${a.periodic ? "real FFT, since it is periodic" : "DCT"})`).join(", ");
  const tAxis = st.report.lhs_axis || "t";
  const list = h(
    "ul",
    { class: "opts", role: "radiogroup", "aria-label": "Smoothing" },
    option(st, "none", "None", "The search reads the field as loaded.", null, null),
    option(st, "savgol", ["Savitzky-Golay window ", h("code", null, "savgol_window")], "A run parameter every algorithm accepts; 0 turns it off.", h("label", null, "window ", intInput(f.savgol, (v) => { f.savgol = v; changed(st, { preview: f.kind === "savgol" }); }, "savgol_window")), null),
    option(
      st,
      "lowpass",
      ["Low-pass ", h("code", null, "kd.lowpass_smooth")],
      `Keeps the lowest modes along ${spectralText || "each axis but the left-hand-side axis"}; an optional Savitzky-Golay window along ${tAxis}. The spectrum on the right shows what each count leaves out.`,
      [...modeKnobs, h("label", null, "time_window ", intInput(f.timeWindow, (v) => { f.timeWindow = v; changed(st, { preview: f.kind === "lowpass" }); }, "time_window"))],
      refusals.lowpass,
    ),
    option(
      st,
      "network",
      ["Network ", h("code", null, "kd.network_smooth")],
      "Trains one field network on every grid point and samples it. Every run trains it again before its search, under the run's own memory cap, so it has no preview on this page.",
      [h("label", null, "seed ", intInput(f.seed, (v) => { f.seed = v; changed(st, { preview: false }); }, "network seed")), h("span", null, "recipe: KD default field recipe")],
      refusals.network,
    ),
  );
  const also = h("input", { type: "checkbox" });
  also.addEventListener("change", () => {
    f.alsoRaw = also.checked;
    changed(st, { preview: false });
  });
  st.nodes.smoothingRadios = list;
  st.nodes.also = also;
  return h(
    "div",
    null,
    list,
    h("label", { class: "also" }, also, "Also queue each run on the data as loaded, next to the smoothed copy"),
    h("p", { class: "note", style: "margin:.4rem 0 0" }, h("code", null, "kd.select_lowpass_modes"), " chooses the mode count by refitting a set of terms at several counts. It needs those terms, so it is offered on a result page with that result's equation."),
  );
}

/** Reflect the form in the smoothing list: selected option, enabled knobs, the also-raw box. */
function syncSmoothing(st) {
  const f = st.form;
  for (const li of st.nodes.smoothingRadios.children) {
    const sel = li.dataset.kind === f.kind;
    li.classList.toggle("sel", sel);
    li.querySelector("input[type=radio]").checked = sel;
    for (const input of li.querySelectorAll(".knobs input")) input.disabled = !sel;
  }
  st.nodes.also.disabled = f.kind === "none";
  st.nodes.also.checked = f.kind !== "none" && f.alsoRaw;
}

// ---------------------------------------------------------------------------
// Preview views

function renderPreviewParts(st) {
  renderStatus(st);
  renderWarnings(st);
  renderTarget(st);
  renderField(st);
  renderNoise(st);
  renderSpectrum(st);
}

function renderStatus(st) {
  st.nodes.status.replaceChildren(st.loading && st.result ? h("p", { class: "note", role: "status" }, "Updating the preview for the new choices.") : "");
}

function renderWarnings(st) {
  const list = [...st.report.warnings, ...(st.result ? st.result.warnings : [])];
  st.nodes.warnings.replaceChildren(...[warningBlock([...new Set(list)])].filter(Boolean));
}

/** The loading or error view in place of a preview part; kd's error is shown once, under Target. */
function previewProblem(st, { primary = false } = {}) {
  if (st.resultError && !primary) return h("p", { class: "note" }, "No view: KD could not compute the preview; its message is under Target.");
  if (st.resultError) return errorState(st.resultError, { title: "KD could not compute the preview.", retry: () => requestPreview(st) });
  if (!st.result) return loadingState("Computing the preview");
  return null;
}

function heat(slice, caption) {
  const el = h("div", { class: "plot" });
  queueMicrotask(() => el.isConnected && fieldPlot(el, slice));
  return h("div", null, el, h("p", { class: "plot-caption" }, caption || samplingText(slice)));
}

function samplingText(slice) {
  const s = slice.sampling;
  const shape = slice.full_shape.join(" × ");
  const shown = slice.topology === "grid" && slice.axes.length === 2 ? `${slice.coordinates[slice.axes[0]].length} × ${slice.coordinates[slice.axes[1]].length}` : String(s.n_shown);
  const what = s.method === "none" ? `All ${s.n_total} values` : `Drawn from ${shown} of the ${s.n_total} values`;
  const missing = slice.missing_count ? `; ${slice.missing_count} not finite` : "";
  return `${what} of the ${shape} grid (axes ${slice.axes.join(", ")})${missing}.`;
}

function renderTarget(st) {
  const box = st.nodes.target;
  disposeWithin(box);
  const f = st.form;
  const smoothed = f.kind !== "none";
  if (st.tab === "smoothed" && !smoothed) st.tab = "loaded";
  const strip = tabs(
    [
      { id: "loaded", label: "As loaded, finite differences" },
      { id: "smoothed", label: "Smoothed copy", disabled: !smoothed },
      { id: "autograd", label: "autograd" },
    ],
    {
      selected: st.tab,
      onSelect: (id) => {
        st.tab = id;
        if (id === "autograd") requestAutograd(st);
        renderTarget(st);
      },
    },
  );
  box.replaceChildren(strip.element, targetBody(st), h("p", { class: "note", style: "margin:.5rem 0 0" }, "With a smoothing option chosen, the Smoothed copy tab shows the derivative the runs will fit. The autograd tab is the derivative of a network trained on the field."));
}

function targetBody(st) {
  if (st.tab === "autograd") {
    if (st.autogradError) return errorState(st.autogradError, { title: "KD could not compute the autograd preview.", retry: () => { st.autogradError = null; requestAutograd(st); } });
    if (!st.autograd) return loadingState("Computing the autograd preview");
    const slice = st.autograd.lhs_before;
    return slice ? heat(slice) : h("div", null, warningBlock(st.autograd.warnings) || emptyState("KD returned no autograd view for this dataset."));
  }
  const problem = previewProblem(st, { primary: true });
  if (problem) return problem;
  const r = st.result;
  const slice = st.tab === "smoothed" ? r.lhs_after : r.lhs_before;
  if (!slice) return warningBlock(r.warnings) || emptyState("KD returned no view of this derivative for these choices.");
  return heat(slice);
}

function renderField(st) {
  const box = st.nodes.field;
  disposeWithin(box);
  const problem = previewProblem(st);
  if (problem) {
    box.replaceChildren(problem);
    return;
  }
  const r = st.result;
  const after = r.field_after
    ? heat(r.field_after)
    : h("div", { class: "note" }, st.form.kind === "none" ? "No smoothing is chosen; the runs read the field as loaded." : "KD returned no smoothed view for this choice; see the warnings above.");
  box.replaceChildren(
    h("div", { class: "pair" }, h("div", null, h("h4", null, "As loaded"), heat(r.field_before)), h("div", null, h("h4", null, "After the smoothing"), after)),
  );
}

function renderNoise(st) {
  const r = st.report;
  const field = (r.fields || []).find((f) => f.name === r.lhs_field) || (r.fields || [])[0];
  if (!field) {
    st.nodes.noise.replaceChildren(h("p", { class: "note" }, "KD reports no field statistics for this dataset."));
    return;
  }
  const axes = Object.keys(field.noise_estimate);
  const noise = st.result ? st.result.noise_estimate : field.noise_estimate;
  const rows = [
    ["Noise estimate", noise],
    st.result && st.result.noise_estimate_after ? ["Noise estimate after the smoothing", st.result.noise_estimate_after] : null,
    ["Step gap, first derivative", field.first_derivative_gap],
    ["Step gap, second derivative", field.second_derivative_gap],
  ].filter(Boolean);
  st.nodes.noise.replaceChildren(
    table([{ label: "", render: (row) => row[0] }, ...axes.map((a) => ({ label: a, num: true, render: (row) => fmtNumber(row[1][a]) }))], rows),
    h("p", { class: "note", style: "margin:.5rem 0 0" }, "Noise estimate is relative, on the scale of the ", h("code", null, "kd.add_noise"), " level. Step gap compares finite differences taken with step 1 and step 2."),
  );
}

function renderSpectrum(st) {
  const box = st.nodes.spectrum;
  disposeWithin(box);
  const problem = previewProblem(st);
  if (problem) {
    box.replaceChildren(problem);
    return;
  }
  const curves = st.result.spectrum;
  if (!curves.length) {
    box.replaceChildren(h("p", { class: "note" }, "KD has no low-pass spectrum for this data: it is not a grid."));
    return;
  }
  const axes = new Map((st.report.axes || []).map((a) => [a.name, a]));
  const kept = (c) => (st.form.kind === "lowpass" && c.axis in st.form.modes ? st.form.modes[c.axis] : Math.min(32, c.missed.length));
  const series = curves.map((c) => ({
    name: `along ${c.axis} (${axes.get(c.axis) && axes.get(c.axis).periodic ? "FFT" : "DCT"})`,
    x: c.missed.map((_, i) => i + 1),
    y: c.missed.map((v) => (v > 0 ? v : null)),
  }));
  const marks = st.form.kind === "lowpass" ? curves.filter((c) => c.axis in st.form.modes).map((c) => ({ x: kept(c), y: c.missed[kept(c) - 1] })).filter((m) => m.y > 0) : [];
  const el = h("div", { class: "plot tall" });
  const readout = curves.map((c) => `Along ${c.axis}, ${kept(c)} modes leave out ${fmtNumber(c.missed[kept(c) - 1])}.`).join(" ");
  box.replaceChildren(
    el,
    h("p", { class: "note", style: "margin:.4rem 0 0" }, `From kd.lowpass_spectrum of the field as loaded. ${readout} A curve that stops falling has reached the noise floor of the data.`),
  );
  queueMicrotask(() => el.isConnected && linePlot(el, { series, xLabel: "modes kept, m", yLabel: "energy left out", logY: true, marks, xMax: Math.max(...curves.map((c) => c.missed.length)) }));
}

// ---------------------------------------------------------------------------
// Summary

function smoothingText(st) {
  const f = st.form;
  if (f.kind === "savgol") return `Savitzky-Golay window ${f.savgol} (savgol_window)`;
  if (f.kind === "lowpass") {
    const modes = Object.entries(f.modes).map(([a, m]) => `${m} modes along ${a}`).join(", ");
    return `low-pass, ${modes}${f.timeWindow ? `, time_window ${f.timeWindow}` : ""}`;
  }
  if (f.kind === "network") return `network smoothing, seed ${f.seed}, trained in every run`;
  return "none";
}

function renderSummary(st) {
  const f = st.form;
  const fp = st.preview.fingerprint;
  const derivText =
    f.derivatives === "both" ? "each mode's own provider; finite-difference and autograd modes preselected" : `each mode's own provider; ${f.derivatives === "finite_diff" ? "finite-difference" : "autograd"} modes preselected`;
  st.nodes.summary.replaceChildren(
    kv([
      ["Derivatives", derivText],
      ["Smoothing", smoothingText(st)],
      f.kind !== "none" ? ["Data as loaded", f.alsoRaw ? "each run is queued a second time on the data as loaded" : "not queued separately"] : null,
      ["Data", h("span", null, `${st.label} as loaded, fingerprint `, h("span", { class: "mono", style: "font-size:.8rem", title: fp }, `…${fp.slice(-8)}`))],
      ["Preselected", `${preselected(st).length} modes`],
    ]),
    h("div", { style: "margin-top:.6rem" }, button("Reset to none", { small: true, onClick: () => resetForm(st) })),
  );
}

function resetForm(st) {
  st.form.kind = "none";
  st.form.alsoRaw = false;
  st.form.derivatives = "finite_diff";
  changed(st);
}
