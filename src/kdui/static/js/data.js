// Data step: choose a catalog dataset or read a file, map
// its arrays, see what KD reads from it (kd.preview_report), its field drawn
// from the slice route, and which algorithm modes accept it with KD's reason
// for each that does not. The reference equation of a catalog dataset is a
// plain per-dataset toggle. Every number on this page comes from the API.
//
// Addresses: #/data (the current dataset of the store, else the catalog),
// #/data/<dataset_id>, #/data?view=file, #/data?view=file&upload=<upload_id>.

import { get, post, upload, store } from "./api.js";
import {
  h, frag, clear, href, go, replaceRoute, fmtNumber, fmtBytes, button, linkButton, tag, pageHead, panel, kv,
  table, tabs, warningBlock, codeBlock, loadingState, emptyState, errorState, toast, equation,
} from "./ui.js";
import { fieldPlot, linePlot } from "./plots.js";

const SUB = "Choose a catalog dataset or read a file, check what KD reads from it, and see which algorithms accept it.";
const SUPPORTED = "KD reads numpy (.npy, .npz), MATLAB (.mat, including v7.3), HDF5, CSV and XLSX files. The file is read in a separate process.";


// A view belongs to one mount; after a newer mount it is no longer connected,
// so a late answer of an earlier dataset does not overwrite the current one.
const alive = (view) => view.container.isConnected;

/** "u_tt" -> "u_{tt}" for typesetting a derivative label. */
const texLabel = (label) => label.replace(/_([A-Za-z0-9]+)/g, "_{$1}");


// ---------------------------------------------------------------------------
// Mount

export async function mount(container, params) {
  let datasetId = params.id || null;
  if (!datasetId && !params.view && !params.upload) {
    const current = store.get("dataset");
    if (current && current.dataset_id) {
      datasetId = current.dataset_id;
      replaceRoute("data", datasetId);
    }
  }
  // Prepare needs a chosen dataset; the file view before "Load and check" has none.
  const prepare = datasetId || store.get("dataset") ? linkButton("Continue to Prepare", href("prepare"), { kind: "primary" }) : null;
  const left = h("div", { class: "stack" });
  const right = h("div", { class: "stack" }, loadingState());
  const below = h("div");
  container.append(pageHead({ title: "Data", sub: SUB, actions: [prepare] }), h("div", { class: "split" }, left, right), below);

  const list = await get("/datasets");
  if (!container.isConnected) return;
  const view = { container, list, left, right, below, datasetId, uploadId: params.upload || null, fileMode: params.view === "file" || Boolean(params.upload) };
  if (datasetId) await showDataset(view);
  else await showWithoutDataset(view);
}

export function unmount() {}

async function showWithoutDataset(view) {
  renderSource(view, null, null, null);
  if (view.uploadId) {
    await showFile(view, null);
    return;
  }
  view.right.replaceChildren(
    view.fileMode
      ? panel({ title: "File" }, emptyState("Upload a file or give its path on this machine. Its arrays and the mapping form appear here."))
      : panel({ title: "Dataset" }, emptyState("Choose a dataset from the catalog, or switch to File to read your own data.")),
  );
}

// ---------------------------------------------------------------------------
// Source panel (left): catalog list or file input

function renderSource(view, detail, uploadInfo, inspection) {
  const catalogId = detail && detail.source.origin === "catalog" ? detail.source.catalog_id : null;
  const body = h("div");
  const showMode = (mode) => {
    seg.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    body.replaceChildren(mode === "file" ? fileSource(view, uploadInfo, inspection) : catalogSource(view, catalogId));
  };
  const seg = ["catalog", "file"].map((mode) =>
    h("button", { type: "button", "data-mode": mode, "aria-pressed": "false", onClick: () => showMode(mode) }, mode === "catalog" ? "Catalog" : "File"),
  );
  const fileMode = view.fileMode || (detail && detail.source.origin === "upload");
  showMode(fileMode ? "file" : "catalog");
  view.left.replaceChildren(frag(
    panel({ title: "Source" }, h("div", { class: "seg", role: "group", "aria-label": "Source type", style: "margin-bottom:.7rem" }, seg), body),
    earlierFiles(view),
  ));
}

function catalogSource(view, selectedCatalogId) {
  const filter = h("input", { type: "text", placeholder: "Filter by id or tag", "aria-label": "Filter datasets", style: "width:100%;margin-bottom:.6rem" });
  const box = h("div");
  const columns = [
    { label: "Dataset", render: (d) => [h("span", { class: "mono" }, d.catalog_id), h("span", { class: "sub" }, [d.tags.slice(0, 2).join(", "), d.tier === "remote" ? ", downloads on first use" : ""].join(""))] },
    { label: "Layout", render: (d) => [d.topology, h("span", { class: "sub" }, d.axes.join(", "))] },
    { label: "LHS", render: (d) => (d.lhs ? equation(texLabel(d.lhs), { size: "inline" }) : h("span", { class: "note" }, "none")) },
  ];
  const draw = () => {
    const q = filter.value.trim().toLowerCase();
    const rows = view.list.catalog.filter((d) => !q || d.catalog_id.includes(q) || d.tags.some((t) => t.includes(q)));
    const t = table(columns, rows, {
      empty: "No catalog dataset matches.",
      className: "catalog",
      rowClass: (d) => (d.catalog_id === selectedCatalogId ? "sel" : null),
      onRowClick: (d) => chooseCatalog(view, d.catalog_id).catch((err) => toast(err.message, "error")),
    });
    t.querySelector("table").prepend(h("colgroup", null, h("col", { style: "width:56%" }), h("col", { style: "width:28%" }), h("col", { style: "width:16%" })));
    box.replaceChildren(t);
  };
  filter.addEventListener("input", draw);
  draw();
  return frag(filter, box, h("p", { class: "note", style: "margin-top:.6rem" }, "The list shows no equations. A dataset's reference equation is shown only on request, per dataset."));
}

/** Same source and recipe give the same dataset_id: reuse a registered one, else register. */
async function chooseCatalog(view, catalogId) {
  const known = view.list.registered.find((d) => d.source.origin === "catalog" && d.source.catalog_id === catalogId);
  if (known) {
    go("data", known.dataset_id);
    return;
  }
  const detail = await post("/datasets", { source: { origin: "catalog", catalog_id: catalogId } });
  go("data", detail.dataset_id);
}

function fileSource(view, uploadInfo, inspection) {
  const chooser = h("input", { type: "file", style: "display:none", "aria-label": "File to upload" });
  chooser.addEventListener("change", async () => {
    const file = chooser.files[0];
    if (!file) return;
    toast(`Uploading ${file.name} (${fmtBytes(file.size)})`);
    try {
      const answer = await upload("/uploads", file, { filename: file.name });
      go("data", null, { view: "file", upload: answer.upload_id });
    } catch (err) {
      toast(err.message, err.code === "payload_too_large" ? "warn" : "error", 12000);
    }
  });
  const path = h("input", { type: "text", placeholder: "Path on this machine", "aria-label": "Path of a file on this machine", value: uploadInfo && uploadInfo.path ? uploadInfo.path : null });
  const importPath = async () => {
    if (!path.value.trim()) throw new Error("Give the path of a file on this machine.");
    const answer = await post("/uploads/import", { path: path.value.trim() });
    go("data", null, { view: "file", upload: answer.upload_id });
  };
  path.addEventListener("keydown", (event) => {
    if (event.key === "Enter") importPath().catch((err) => toast(err.message, "error", 12000));
  });
  return frag(
    h("div", { class: "pathrow" }, path, button("Use path", { small: true, onClick: importPath })),
    h("div", { class: "pathrow", style: "margin-top:.5rem" }, button("Upload a file", { small: true, onClick: () => chooser.click() }), chooser),
    h("p", { class: "note", style: "margin-top:.4rem" }, "A path is read in place, with no size limit. An upload is copied into KD's workspace; above the size limit in kd-ui's settings, use a path."),
    uploadInfo &&
      h(
        "div",
        { style: "margin-top:.8rem" },
        kv([
          ["File", h("span", { class: "mono" }, uploadInfo.filename)],
          uploadInfo.path ? ["Path", h("span", { class: "mono" }, uploadInfo.path)] : ["Stored", "uploaded copy"],
          inspection && ["Container", inspection.container],
          ["Size", fmtBytes(uploadInfo.size_bytes)],
          inspection && ["Layout", inspection.layouts.length ? inspection.layouts.join(", ") : "none recognized; give the mapping"],
          inspection && ["Arrays", String(inspection.arrays.length)],
        ]),
      ),
    h("p", { class: "note", style: "margin-top:.7rem" }, SUPPORTED),
  );
}

function earlierFiles(view) {
  const files = view.list.registered.filter((d) => d.source.origin === "upload");
  if (!files.length) return null;
  return panel(
    { title: "Datasets from files" },
    h("ul", { class: "earlier", style: "margin:0;padding-left:1.1rem" }, files.map((d) => h("li", null, h("a", { href: href("data", d.dataset_id), class: d.dataset_id === view.datasetId ? "mono" : null }, d.label)))),
  );
}

// ---------------------------------------------------------------------------
// One dataset: summary, field, report, applicability

async function showDataset(view) {
  let detail;
  try {
    detail = await get(`/datasets/${encodeURIComponent(view.datasetId)}`);
  } catch (err) {
    if (!alive(view)) return;
    renderSource(view, null, null, null);
    view.right.replaceChildren(errorState(err, { title: "This dataset could not be read.", retry: () => showDataset(view) }));
    return;
  }
  if (!alive(view)) return;
  const catalog = detail.source.origin === "catalog" ? view.list.catalog.find((d) => d.catalog_id === detail.source.catalog_id) : null;
  const summary = summaryPanel(view, detail, catalog, null);
  const guess = catalog ? [catalog.topology, ...catalog.axes, catalog.lhs].filter(Boolean).join(", ") : "file";
  store.set("dataset", { dataset_id: detail.dataset_id, label: detail.label, summary: guess });
  const previewArea = h("div", { class: "stack" }, loadingState("KD is loading the dataset and computing its report"));
  if (detail.source.origin === "upload") {
    view.uploadId = detail.source.upload_id;
    view.right.replaceChildren(loadingState());
    await showFile(view, detail);
    if (!alive(view)) return;
    view.right.append(panel({ title: "What KD read", meta: "kd.preview_report, after Load and check" }, ...summary.nodes), previewArea);
  } else {
    renderSource(view, detail, null, null);
    view.right.replaceChildren(h("section", { class: "panel" }, ...summary.nodes), previewArea);
  }
  await loadPreview(view, detail, catalog, summary, previewArea);
}

async function loadPreview(view, detail, catalog, summary, previewArea) {
  const capabilities = get("/capabilities").catch((err) => ({ error: err }));
  let preview;
  try {
    preview = await get(`/datasets/${encodeURIComponent(detail.dataset_id)}/preview`);
  } catch (err) {
    if (!alive(view)) return;
    previewArea.replaceChildren(errorState(err, { title: "KD could not load this dataset.", retry: () => loadPreview(view, detail, catalog, summary, previewArea) }));
    view.below.replaceChildren();
    return;
  }
  if (!alive(view)) return;
  summary.setFingerprint(preview.fingerprint);
  store.set("dataset", { dataset_id: detail.dataset_id, label: detail.label, summary: shortSummary(preview.report) });
  previewArea.replaceChildren(fieldPanel(detail, preview), h("div", { class: "grid-2" }, axesPanel(preview.report), fieldStatsPanels(preview.report)));
  view.below.replaceChildren(h("div", { class: "full-width" }, loadingState("Reading the algorithms")));
  const caps = await capabilities;
  if (!alive(view)) return;
  view.below.replaceChildren(h("div", { class: "full-width" }, applicabilityPanel(detail, preview, caps)));
}

/** "grid, x 256, t 201, u_t" for the header. */
function shortSummary(report) {
  const parts = [report.topology];
  for (const axis of report.axes || []) parts.push(`${axis.name} ${axis.n}`);
  if (report.lhs_label) parts.push(report.lhs_label);
  return parts.join(", ");
}

function loadCall(detail) {
  if (detail.source.origin === "catalog") {
    const lhs = detail.recipe.lhs ? `, lhs="${detail.recipe.lhs}"` : "";
    return `kd.load("${detail.source.catalog_id}"${lhs})`;
  }
  return null;
}

function summaryPanel(view, detail, catalog, fingerprint) {
  const fp = h("dd", { class: "mono", style: "word-break:break-all" }, fingerprint || detail.fingerprint || "known after the first preview");
  const call = loadCall(detail);
  const tags = catalog ? [catalog.topology, catalog.tier, ...catalog.tags] : ["file"];
  const head = h(
    "div",
    { class: "ds-head" },
    h("div", null, h("h2", null, detail.label), h("div", { class: "tags" }, tags.map((t) => tag(t)))),
    h("dl", { class: "kv", style: "max-width:34rem" }, call && [h("dt", null, "Loaded with"), h("dd", null, h("code", null, call))], h("dt", null, "Fingerprint"), fp),
  );
  const ref = detail.reference_available ? referenceBox(detail) : null;
  return {
    nodes: [head, ref],
    setFingerprint: (value) => {
      fp.textContent = value;
    },
  };
}

function referenceBox(detail) {
  const box = h("div", { class: "ref" });
  const draw = (d) => {
    const shown = d.reference_shown && d.reference_equation;
    const toggle = button(shown ? "Hide reference equation" : "Show reference equation", {
      onClick: async () => {
        const next = await post(`/datasets/${encodeURIComponent(d.dataset_id)}/reference`, { shown: !d.reference_shown });
        if (box.isConnected) draw(next);
      },
    });
    const what = shown
      ? h("div", { class: "what" }, equation(d.reference_equation.latex, { size: "mid" }), h("b", null, "Reference equation shown. "), "A fit submitted while it is shown includes it in its report. Hide it again at any time.")
      : h("div", { class: "what" }, h("b", null, "Reference equation hidden. "), "The catalog lists an equation for this dataset. It stays hidden unless you show it here.");
    box.replaceChildren(what, toggle);
  };
  draw(detail);
  return box;
}

// ---------------------------------------------------------------------------
// Field preview: one tab per default slice, time slices

function fieldPanel(detail, preview) {
  const report = preview.report;
  const axesByName = Object.fromEntries((report.axes || []).map((a) => [a.name, a]));
  const content = h("div");
  const items = preview.slices.map((spec, i) => ({ id: `slice-${i}`, label: `${spec.field}(${spec.axes.join(", ")})` }));
  const sweep = timeSliceSpec(preview, axesByName);
  if (sweep) items.push({ id: "time", label: report.lhs_axis === "t" ? "Time slices" : `${report.lhs_axis} slices` });
  let current = null;
  const select = (id) => {
    if (current) current.dispose();
    current = null;
    clear(content);
    if (id === "time") current = timeSlices(content, detail, sweep, axesByName);
    else current = sliceView(content, detail, preview.slices[Number(id.slice(6))], axesByName);
  };
  const strip = tabs(items, { selected: items[0].id, onSelect: select });
  const element = panel(
    { title: "Field", meta: "Sampled by the server, drawn in the browser" },
    preview.slices.length ? strip.element : null,
    preview.slices.length ? content : emptyState("KD gave no field to draw for this dataset."),
  );
  if (preview.slices.length) select(items[0].id);
  return element;
}

function samplingNote(slice) {
  const s = slice.sampling;
  const shape = slice.full_shape.join(" × ");
  let text;
  if (s.method === "none") text = `Drawn from all ${s.n_shown.toLocaleString()} values of this slice (field shape ${shape}).`;
  else if (s.method === "stride") {
    const steps = Object.entries(s.stride || {}).map(([axis, k]) => `every ${k} on ${axis}`).join(", ");
    text = `Drawn from ${s.n_shown.toLocaleString()} of ${s.n_total.toLocaleString()} values (${steps}; field shape ${shape}).`;
  } else text = `Drawn from a random ${s.n_shown.toLocaleString()} of ${s.n_total.toLocaleString()} points.`;
  const missing = slice.missing_count ? ` ${slice.missing_count} values shown are not finite and are left blank.` : "";
  return `${text}${missing} Searches use the full data. Hover for values.`;
}

/** Plot of one slice with a control per fixed axis (index along it). */
function sliceView(content, detail, spec, axesByName) {
  const plotEl = h("div", { class: "plot tall" });
  const note = h("p", { class: "plot-caption" });
  const fixed = { ...spec.fixed };
  let handle = null;
  let disposed = false;
  const load = async () => {
    const slice = await post(`/datasets/${encodeURIComponent(detail.dataset_id)}/slice`, { field: spec.field, axes: spec.axes, fixed });
    if (disposed || !plotEl.isConnected) return;
    if (handle) handle.dispose();
    handle = fieldPlot(plotEl, slice);
    note.textContent = samplingNote(slice);
  };
  const controls = Object.keys(fixed).map((axis) => {
    const n = axesByName[axis] ? axesByName[axis].n : null;
    const label = h("span", null);
    const show = () => {
      label.textContent = `${axis} index ${fixed[axis]}${n ? ` of ${n - 1}${coordinateHint(axesByName[axis], fixed[axis])}` : ""}`;
    };
    const input = h("input", { type: "range", min: 0, max: n ? n - 1 : fixed[axis], value: fixed[axis], "aria-label": `Index along ${axis}` });
    input.addEventListener("input", () => {
      fixed[axis] = Number(input.value);
      show();
    });
    input.addEventListener("change", () => load().catch((err) => note.replaceChildren(errorState(err, { title: "The slice could not be drawn." }))));
    show();
    return h("label", { class: "slicectl" }, input, label);
  });
  content.append(frag(controls.length ? h("div", { class: "slicectl" }, controls) : null, plotEl, note));
  plotEl.append(loadingState("Sampling the field"));
  load().catch((err) => plotEl.replaceChildren(errorState(err, { title: "The field could not be sampled.", retry: () => load() })));
  return {
    dispose() {
      disposed = true;
      if (handle) handle.dispose();
    },
  };
}

/** For a uniform axis, the coordinate of an index as reported min plus index times step. */
function coordinateHint(axis, index) {
  if (!axis || axis.spacing !== "uniform" || axis.min === null || axis.step_mean === null) return "";
  return ` (${axis.name} = ${fmtNumber(axis.min + index * axis.step_mean)})`;
}

/** Curves of the first field over one spatial axis at a few points of the LHS axis. */
function timeSliceSpec(preview, axesByName) {
  const report = preview.report;
  const t = report.lhs_axis;
  if (report.topology !== "grid" || !t || !axesByName[t] || !preview.slices.length) return null;
  const spec = preview.slices.find((s) => s.field === report.lhs_field) || preview.slices[0];
  const space = spec.axes.find((a) => a !== t);
  if (!space) return null;
  const others = Object.fromEntries(Object.entries(spec.fixed).filter(([a]) => a !== t));
  const n = axesByName[t].n;
  const picks = [...new Set([0, Math.round((n - 1) / 3), Math.round((2 * (n - 1)) / 3), n - 1])];
  return { field: spec.field, space, t, others, picks };
}

function timeSlices(content, detail, sweep, axesByName) {
  const plotEl = h("div", { class: "plot tall" }, loadingState("Sampling the field"));
  const note = h("p", { class: "plot-caption" });
  content.append(plotEl, note);
  let handle = null;
  let disposed = false;
  const load = async () => {
    const series = [];
    for (const k of sweep.picks) {
      const fixed = { ...sweep.others, [sweep.t]: k };
      const slice = await post(`/datasets/${encodeURIComponent(detail.dataset_id)}/slice`, { field: sweep.field, axes: [sweep.space], fixed });
      if (disposed) return;
      const hint = coordinateHint(axesByName[sweep.t], k);
      series.push({ name: hint ? hint.slice(2, -1) : `${sweep.t} index ${k}`, x: slice.coordinates[sweep.space], y: slice.values });
    }
    if (disposed || !plotEl.isConnected) return;
    clear(plotEl);
    handle = linePlot(plotEl, { series, xLabel: sweep.space, yLabel: sweep.field });
    note.textContent = `${sweep.field} over ${sweep.space} at ${series.length} points of ${sweep.t}. Searches use the full data. Hover for values.`;
  };
  load().catch((err) => plotEl.replaceChildren(errorState(err, { title: "The field could not be sampled.", retry: () => load() })));
  return {
    dispose() {
      disposed = true;
      if (handle) handle.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// kd.preview_report: axes, fields

function axesPanel(report) {
  const lhs = report.lhs_label
    ? [equation(texLabel(report.lhs_label), { size: "inline" }), `, order ${report.lhs_order} in ${report.lhs_axis}`]
    : `none (order ${report.lhs_order})`;
  const warnings = report.warnings.length ? warningBlock(report.warnings) : "none";
  if (!report.axes) return panel({ title: "Axes", meta: "kd.preview_report" }, h("p", { class: "note" }, `This ${report.topology} dataset reports no axes.`), kv([["Left-hand side", lhs], ["Warnings", warnings]]));
  const columns = [
    { label: "Axis", key: "name" },
    { label: "Points", num: true, render: (a) => String(a.n) },
    { label: "Range", render: (a) => `${fmtNumber(a.min)} to ${fmtNumber(a.max)}` },
    { label: "Step", num: true, render: (a) => [fmtNumber(a.step_mean), h("span", { class: "sub" }, a.spacing)] },
    { label: "Periodic", render: (a) => (a.periodic ? "yes" : "no") },
  ];
  return panel({ title: "Axes", meta: "kd.preview_report" }, table(columns, report.axes), h("div", { style: "margin-top:.8rem" }, kv([["Left-hand side", lhs], ["Warnings", warnings]])));
}

function fieldStatsPanels(report) {
  const fields = report.fields || [];
  if (!fields.length) return panel({ title: "Fields", meta: "kd.preview_report" }, h("p", { class: "note" }, "KD reports no field statistics for this dataset."));
  return h("div", { class: "stack" }, fields.map(fieldStatsPanel));
}

function fieldStatsPanel(f) {
  const axes = [...new Set([...Object.keys(f.noise_estimate), ...Object.keys(f.first_derivative_gap), ...Object.keys(f.second_derivative_gap)])];
  const rows = [
    ["Noise estimate", f.noise_estimate],
    ["Step gap, first derivative", f.first_derivative_gap],
    ["Step gap, second derivative", f.second_derivative_gap],
  ];
  const columns = [{ label: "Per axis", render: (r) => r[0] }, ...axes.map((a) => ({ label: a, num: true, render: (r) => fmtNumber(r[1][a] ?? null) }))];
  return panel(
    { title: `Field ${f.name}`, meta: "kd.preview_report" },
    kv([
      ["dtype, shape", `${f.dtype}, ${f.shape.join(" × ")}`],
      ["min, max", `${fmtNumber(f.min)}, ${fmtNumber(f.max)}`],
      ["mean", fmtNumber(f.mean)],
      ["NaN, Inf", `${f.nan_count}, ${f.inf_count}`],
    ]),
    axes.length ? h("div", { style: "margin-top:.8rem" }, table(columns, rows)) : null,
    h("p", { class: "note", style: "margin-top:.5rem" }, "Noise estimate is relative, on the scale of the ", h("code", null, "kd.add_noise"), " level. Step gap is the relative difference between finite differences taken with step 1 and step 2."),
  );
}

// ---------------------------------------------------------------------------
// Applicability: kd.mode_accepts per mode, joined with the capability snapshot

function applicabilityPanel(detail, preview, caps) {
  const verdicts = preview.applicability;
  const accepted = verdicts.filter((v) => v.accepts).length;
  const snapshot = caps.error ? null : caps.snapshot;
  const algorithms = snapshot ? snapshot.algorithms : [...new Set(verdicts.map((v) => v.algorithm))].map((algorithm) => ({ algorithm, modes: [], warnings: [] }));
  const body = h("tbody");
  for (const info of algorithms) {
    const rows = verdicts.filter((v) => v.algorithm === info.algorithm);
    rows.forEach((v, i) => body.append(applicabilityRow(info, v, i, rows.length, Boolean(snapshot))));
  }
  const missing = caps.error
    ? warningBlock([`The algorithm list could not be read: ${caps.error.message}`])
    : snapshot
      ? null
      : warningBlock(["KD is still collecting its algorithm list; cost, score and what this machine has appear after the next refresh."]);
  return panel(
    { title: "Algorithms for this data", meta: `${accepted} of ${verdicts.length} modes accept ${detail.label}` },
    h("p", { class: "note", style: "margin-bottom:.7rem" }, "Each mode's declaration is checked against the data's layout, equation form, LHS order, fields and axes. A refused mode shows KD's reason. The last column is what this machine has for the algorithm; a missing item is a warning, not a refusal."),
    missing,
    h(
      "div",
      { class: "tablebox" },
      h(
        "table",
        { class: "t acc" },
        h("colgroup", null, ["21%", "12%", "10%", "7%", "11%", "23%", "16%"].map((w) => h("col", { style: `width:${w}` }))),
        h("thead", null, h("tr", null, ["Algorithm", "Mode", "Derivatives", "Cost", "Native score", "For this data", "This machine"].map((t) => h("th", null, t)))),
        body,
      ),
    ),
  );
}

function applicabilityRow(info, verdict, index, count, known) {
  const mode = info.modes.find((m) => m.name === verdict.mode);
  const span = { rowspan: count };
  const needs = Object.entries(verdict.params).filter(([, value]) => typeof value === "string" && /^<.*>$/.test(value));
  const forData = verdict.accepts
    ? [h("span", { class: "yes" }, "Accepts"), needs.length ? h("span", { class: "sub" }, `Needs ${needs.map(([k]) => k).join(", ")} to be set`) : null]
    : h("span", { class: "reason" }, verdict.reason || "Refused");
  const score = known ? [info.score_kind, h("span", { class: "sub" }, info.score_direction === "min" ? "lower is better" : "higher is better")] : "";
  const env = info.warnings.length ? h("span", { class: "env bad" }, info.warnings.join(" ")) : h("span", { class: "env" }, known ? "nothing extra needed" : "");
  return h(
    "tr",
    { class: verdict.accepts ? null : "off" },
    index === 0 ? h("td", span, h("b", null, info.algorithm), info.summary ? h("span", { class: "sub" }, info.summary) : null) : null,
    h("td", { class: "mono" }, verdict.mode),
    h("td", null, mode ? mode.provider_kind || "none" : ""),
    index === 0 ? h("td", span, info.cost_class || "") : null,
    index === 0 ? h("td", span, score) : null,
    h("td", null, forData),
    index === 0 ? h("td", span, env) : null,
  );
}

// ---------------------------------------------------------------------------
// File view: arrays, mapping form, the equivalent kd.load call

async function showFile(view, detail) {
  const id = encodeURIComponent(view.uploadId);
  let info;
  let inspection;
  try {
    [info, inspection] = await Promise.all([get(`/uploads/${id}`), get(`/uploads/${id}/inspection`)]);
  } catch (err) {
    if (!alive(view)) return;
    renderSource({ ...view, fileMode: true }, detail, null, null);
    view.right.replaceChildren(errorState(err, { title: "KD could not read this file.", retry: () => showFile(view, detail) }));
    return;
  }
  if (!alive(view)) return;
  renderSource({ ...view, fileMode: true }, detail, info, inspection);
  const mapping = mappingState(inspection, info, detail);
  view.right.replaceChildren(arraysPanel(mapping), mappingPanel(view, mapping));
}

const MULTI = (shape) => shape.filter((n) => n > 1);
/** A guess at a name: "tt" -> "t", "uu" -> "u", others unchanged. */
const guessName = (key) => (/^(.)\1+$/.test(key) ? key[0] : key);

function mappingState(inspection, info, detail) {
  const recipe = detail ? detail.recipe : null;
  const roles = new Map();
  for (const a of inspection.arrays) {
    let role = { use: MULTI(a.shape).length > 1 ? "field" : "coord", name: guessName(a.key) };
    if (recipe && (recipe.coords || recipe.fields)) {
      const coord = Object.entries(recipe.coords || {}).find(([, key]) => key === a.key);
      const field = Object.entries(recipe.fields || {}).find(([, key]) => key === a.key);
      role = coord ? { use: "coord", name: coord[0] } : field ? { use: "field", name: field[0] } : { use: "ignore", name: guessName(a.key) };
    }
    roles.set(a.key, role);
  }
  const state = {
    inspection,
    info,
    roles,
    kind: (recipe && recipe.kind) || "pde",
    layout: (recipe && recipe.layout) || "",
    fieldAxes: recipe && recipe.field_axes ? [...recipe.field_axes] : null,
    lhs: recipe ? recipe.lhs || "" : null,
    periodic: new Set((recipe && recipe.periodic) || []),
    label: detail ? detail.label : info.filename.replace(/\.[^.]+$/, ""),
    listeners: new Set(),
  };
  if (state.lhs === null) {
    const fields = namesOf(state, "field");
    state.lhs = fields.length && namesOf(state, "coord").includes("t") ? `${fields[0]}_t` : "";
  }
  return state;
}

const namesOf = (state, use) => [...state.roles.values()].filter((r) => r.use === use).map((r) => r.name);
const arrayOf = (state, key) => state.inspection.arrays.find((a) => a.key === key);
const changed = (state) => state.listeners.forEach((fn) => fn());

function arraysPanel(state) {
  const useCell = (a) => {
    const role = state.roles.get(a.key);
    const select = h(
      "select",
      { "aria-label": `Use of ${a.key}` },
      [["coord", "coordinate"], ["field", "field"], ["ignore", "ignore"]].map(([v, l]) => h("option", { value: v, selected: role.use === v }, l)),
    );
    const name = h("input", { type: "text", value: role.name, "aria-label": `Name for ${a.key}`, disabled: role.use === "ignore" });
    select.addEventListener("change", () => {
      role.use = select.value;
      name.disabled = role.use === "ignore";
      state.fieldAxes = null;
      changed(state);
    });
    name.addEventListener("input", () => {
      role.name = name.value.trim();
      state.fieldAxes = null;
      changed(state);
    });
    return h("span", { style: "white-space:nowrap" }, select, name);
  };
  const columns = [
    { label: "Key", render: (a) => h("span", { class: "mono" }, a.key) },
    { label: "Shape", render: (a) => h("span", { class: "mono" }, a.shape.join(" × ")) },
    { label: "dtype", render: (a) => h("span", { class: "mono" }, a.dtype) },
    { label: "min", num: true, render: (a) => fmtNumber(a.min) },
    { label: "max", num: true, render: (a) => fmtNumber(a.max) },
    { label: "NaN, Inf", num: true, render: (a) => `${a.nan_count}, ${a.inf_count}` },
    { label: "Use as", render: useCell },
  ];
  return panel({ title: "Arrays in the file", meta: "kd.inspect_file" }, table(columns, state.inspection.arrays, { className: "map", empty: "KD found no arrays in this file." }));
}

/** Coordinate length: the largest dimension of its array. */
function coordLength(state, name) {
  const key = [...state.roles].find(([, r]) => r.use === "coord" && r.name === name);
  return key ? Math.max(...arrayOf(state, key[0]).shape) : null;
}

/** The field's dimensions (lengths > 1) and the axis chosen for each. */
function fieldDims(state) {
  const fieldKey = [...state.roles].find(([, r]) => r.use === "field");
  if (!fieldKey) return null;
  const dims = MULTI(arrayOf(state, fieldKey[0]).shape);
  const coords = namesOf(state, "coord");
  if (!state.fieldAxes || state.fieldAxes.length !== dims.length) {
    const free = [...coords];
    state.fieldAxes = dims.map((n) => {
      const match = free.find((c) => coordLength(state, c) === n) || free[0] || "";
      free.splice(free.indexOf(match), 1);
      return match;
    });
  }
  return { key: fieldKey[0], name: fieldKey[1].name, dims, coords };
}

function recipeOf(state) {
  const recipe = { kind: state.kind };
  if (state.layout) recipe.layout = state.layout;
  else {
    const coords = {};
    const fields = {};
    for (const [key, role] of state.roles) {
      if (role.use === "coord") coords[role.name] = key;
      if (role.use === "field") fields[role.name] = key;
    }
    const dims = fieldDims(state);
    const order = dims ? state.fieldAxes.filter((a) => a in coords) : [];
    // A table has no coordinates and kd refuses coords= for it, even empty.
    if (Object.keys(coords).length) recipe.coords = Object.fromEntries([...order, ...Object.keys(coords).filter((a) => !order.includes(a))].map((a) => [a, coords[a]]));
    recipe.fields = fields;
    if (dims && dims.dims.length > 1) recipe.field_axes = state.fieldAxes;
  }
  if (state.lhs) recipe.lhs = state.lhs;
  if (state.periodic.size) recipe.periodic = [...state.periodic];
  return recipe;
}

function pythonCall(state) {
  const r = recipeOf(state);
  const json = (v) => JSON.stringify(v).replace(/,/g, ", ").replace(/:/g, ": ");
  const tuple = (list) => `(${list.map((s) => JSON.stringify(s)).join(", ")}${list.length === 1 ? "," : ""})`;
  const source = state.info.path || state.info.filename;
  const lines = [`    ${JSON.stringify(source)},`];
  if (r.kind !== "pde") lines.push(`    kind=${JSON.stringify(r.kind)},`);
  if (r.layout) lines.push(`    layout=${JSON.stringify(r.layout)},`);
  if (r.coords) lines.push(`    coords=${json(r.coords)},`);
  if (r.fields) lines.push(`    fields=${json(r.fields)},`);
  if (r.field_axes) lines.push(`    field_axes=${tuple(r.field_axes)},`);
  if (r.lhs) lines.push(`    lhs=${JSON.stringify(r.lhs)},`);
  if (r.periodic) lines.push(`    periodic=${tuple(r.periodic)},`);
  return `kd.load(\n${lines.join("\n")}\n)`;
}

function mappingPanel(view, state) {
  const grid = h("div", { class: "formgrid" });
  const callBox = h("div");
  const drawCall = () => callBox.replaceChildren(codeBlock(pythonCall(state)));
  const draw = () => {
    grid.replaceChildren(...mappingRows(state));
    drawCall();
  };
  draw.codeOnly = drawCall;
  state.listeners.add(draw);
  draw();
  const submit = button("Load and check", {
    kind: "primary",
    onClick: async () => {
      if (!state.label.trim()) throw new Error("Give the dataset a name.");
      const detail = await post("/datasets", { source: { origin: "upload", upload_id: state.info.upload_id }, recipe: recipeOf(state), label: state.label.trim() });
      go("data", detail.dataset_id);
    },
  });
  return panel(
    { title: "Mapping", meta: "the arguments of kd.load" },
    grid,
    h("h3", { style: "margin:1.1rem 0 .5rem" }, "The same call in Python"),
    callBox,
    h("div", { class: "formrow-actions" }, submit),
  );
}

function mappingRows(state) {
  const rows = [];
  const kind = h("select", { "aria-label": "Kind of data" }, ["pde", "tabular", "trajectory"].map((k) => h("option", { value: k, selected: state.kind === k }, k)));
  kind.addEventListener("change", () => {
    state.kind = kind.value;
    changed(state);
  });
  rows.push(h("div", { class: "lab" }, "Kind"), h("div", null, kind));
  if (state.inspection.layouts.length) {
    const layout = h("select", { "aria-label": "Layout" }, h("option", { value: "", selected: !state.layout }, "my mapping (the arrays table)"), state.inspection.layouts.map((l) => h("option", { value: l, selected: state.layout === l }, `layout ${l}, recognized by KD`)));
    layout.addEventListener("change", () => {
      state.layout = layout.value;
      changed(state);
    });
    rows.push(h("div", { class: "lab" }, "Layout"), h("div", null, layout));
  }
  // A table has columns, not a field on grid axes.
  if (!state.layout && state.kind !== "tabular") rows.push(...dimensionRows(state));
  rows.push(...lhsRows(state));
  const coords = namesOf(state, "coord");
  if (coords.length) {
    rows.push(
      h("div", { class: "lab" }, "Periodic axes"),
      h(
        "div",
        { class: "inline" },
        coords.map((c) => {
          const box = h("input", { type: "checkbox", checked: state.periodic.has(c) });
          box.addEventListener("change", () => {
            if (box.checked) state.periodic.add(c);
            else state.periodic.delete(c);
            changed(state);
          });
          return h("label", { class: "check" }, box, ` ${c}`);
        }),
      ),
    );
  }
  const label = h("input", { type: "text", value: state.label, "aria-label": "Dataset name", style: "width:18rem" });
  label.addEventListener("input", () => {
    state.label = label.value;
  });
  rows.push(h("div", { class: "lab" }, "Dataset name"), h("div", null, label), h("div", { class: "help" }, "The name shown in KD's lists and run pages."));
  return rows;
}

function dimensionRows(state) {
  const dims = fieldDims(state);
  if (!dims) return [h("div", { class: "lab" }, "Field"), h("div", { class: "note" }, "Mark one array as a field in the table above.")];
  const selects = dims.dims.map((n, i) =>
    h(
      "span",
      { class: "dim" },
      `dimension ${i + 1}, length ${n}`,
      (() => {
        const select = h("select", { "aria-label": `Axis of dimension ${i + 1}` }, dims.coords.map((c) => h("option", { value: c, selected: state.fieldAxes[i] === c }, c)));
        select.addEventListener("change", () => {
          state.fieldAxes[i] = select.value;
          changed(state);
        });
        return select;
      })(),
    ),
  );
  const mismatches = dims.dims
    .map((n, i) => [n, state.fieldAxes[i], coordLength(state, state.fieldAxes[i])])
    .filter(([n, , len]) => len !== n)
    .map(([n, axis, len]) => `dimension of length ${n} is set to ${axis || "no axis"}${len ? `, which has ${len} points` : ""}`);
  const check = mismatches.length
    ? warningBlock([`Lengths differ: ${mismatches.join("; ")}. KD will report it when you load.`])
    : h("div", { class: "okline" }, `Lengths match the coordinates: ${dims.dims.map((n, i) => `${state.fieldAxes[i]} ${n}`).join(", ")}.`);
  return [h("div", { class: "lab" }, `Field ${dims.name}`), h("div", null, h("div", { class: "dims" }, selects), check)];
}

function lhsRows(state) {
  const fields = namesOf(state, "field");
  const coords = namesOf(state, "coord");
  const suggestions = fields.flatMap((f) => (coords.includes("t") ? [`${f}_t`, `${f}_tt`] : []));
  const input = h("input", { type: "text", value: state.lhs, list: "kd-data-lhs", "aria-label": "Left-hand side", placeholder: "for example u_t", style: "width:12rem" });
  input.addEventListener("input", () => {
    state.lhs = input.value.trim();
    redrawCall(state);
  });
  const quick = suggestions.map((s) =>
    button(s, {
      small: true,
      onClick: () => {
        state.lhs = s;
        changed(state);
      },
    }),
  );
  return [
    h("div", { class: "lab" }, "Left-hand side"),
    h("div", { class: "inline" }, input, h("datalist", { id: "kd-data-lhs" }, suggestions.map((s) => h("option", { value: s }))), quick),
    h("div", { class: "help" }, "A derivative label such as u_t, or a column of a table. Leave empty to let KD use its default."),
  ];
}

/** Redraw only the Python call (typing in a text box must keep its focus). */
function redrawCall(state) {
  for (const fn of state.listeners) if (fn.codeOnly) fn.codeOnly();
}
