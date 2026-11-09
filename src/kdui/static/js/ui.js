// Shared UI pieces: DOM building, number formats, navigation links, states
// (loading, empty, error), status badges, meters, tables, the equation
// element and the parameter form generated from the capability schema.
// Pages import what they need; README.md lists every export.

import { renderTex } from "./math.js";

// ---------------------------------------------------------------------------
// DOM

/**
 * Create an element. attrs: class, style (string), text, html (trusted
 * strings only), dataset via "data-*", boolean attributes as true/false,
 * event handlers as on<Event> functions (onClick, onInput, ...).
 * Children: nodes, strings, numbers, arrays; null/undefined/false are skipped.
 */
export function h(tag, attrs = null, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "html") el.innerHTML = value;
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2).toLowerCase(), value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, String(value));
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    if (Array.isArray(child)) append(el, child);
    else el.append(child instanceof Node ? child : String(child));
  }
}

/** DocumentFragment of children (same child rules as h), for replaceChildren/append. */
export function frag(...children) {
  const f = document.createDocumentFragment();
  append(f, children);
  return f;
}

/** Remove every child of `el`; returns `el`. */
export function clear(el) {
  el.replaceChildren();
  return el;
}

// ---------------------------------------------------------------------------
// Formats (the KD report's conventions)

/** 4 significant digits, Python-style exponent below 1e-3 and from 1e5: 1.118e-05. */
export function fmtNumber(x, sig = 4) {
  if (x === null || x === undefined) return "n/a";
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return "0";
  const a = Math.abs(x);
  if (a < 1e-3 || a >= 1e5) return x.toExponential(sig - 1).replace(/e([+-])(\d)$/, "e$10$2");
  return String(Number(x.toPrecision(sig)));
}

/** Bytes as GiB with one decimal ("16.0 GiB"); below 1 GiB as MiB (one decimal
 *  below 10 MiB), below 1 MiB as KiB. */
export function fmtBytes(bytes) {
  if (bytes === null || bytes === undefined) return "n/a";
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  const mib = bytes / 1024 ** 2;
  if (mib < 10) return `${mib.toFixed(1)} MiB`;
  const gib = bytes / 1024 ** 3;
  return gib >= 1 ? `${gib.toFixed(1)} GiB` : `${Math.round(mib)} MiB`;
}

/** Seconds as "19.5 s", "5 min 3 s" or "2 h 4 min". */
export function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return "n/a";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m} min ${Math.round(seconds - 60 * m)} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

/** An ISO time as local "10:02" today, else "6 Nov 10:02". */
export function fmtTime(iso) {
  if (!iso) return "n/a";
  const d = new Date(iso);
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
  if (d.toDateString() === new Date().toDateString()) return time;
  return `${d.toLocaleDateString([], { day: "numeric", month: "short" })} ${time}`;
}

/** Seconds between two ISO times (end defaults to now). */
export function elapsedSeconds(startIso, endIso = null) {
  const end = endIso ? new Date(endIso) : new Date();
  return (end - new Date(startIso)) / 1000;
}

/**
 * Parse a seed list such as "0, 1, 2" or "0-4, 7" into integers.
 * Throws Error with a message for the user on anything else.
 */
export function parseSeeds(text) {
  const seeds = [];
  for (const part of text.split(",").map((s) => s.trim()).filter(Boolean)) {
    const range = /^(\d+)\s*-\s*(\d+)$/.exec(part);
    if (range) {
      const [lo, hi] = [Number(range[1]), Number(range[2])];
      if (hi < lo) throw new Error(`"${part}": the range ends before it starts.`);
      for (let s = lo; s <= hi; s += 1) seeds.push(s);
    } else if (/^\d+$/.test(part)) seeds.push(Number(part));
    else throw new Error(`"${part}" is not a seed. Write whole numbers such as 0, 1, 2 or a range such as 0-4.`);
  }
  if (!seeds.length) throw new Error("Give at least one seed.");
  return [...new Set(seeds)];
}

// ---------------------------------------------------------------------------
// Navigation (hash routes, see README.md)

/** "#/page/id?key=value". id and query are optional. */
export function href(page, id = null, query = null) {
  let hash = `#/${page}`;
  if (id !== null && id !== undefined) hash += `/${encodeURIComponent(id)}`;
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(query || {})) if (v !== null && v !== undefined) q.set(k, Array.isArray(v) ? v.join(",") : v);
  const qs = q.toString();
  return qs ? `${hash}?${qs}` : hash;
}

/** Navigate: the shell unmounts the current page and mounts the target. */
export function go(page, id = null, query = null) {
  window.location.hash = href(page, id, query);
}

/** Change the address without remounting (the page already shows that state). */
export function replaceRoute(page, id = null, query = null) {
  const { pathname, search } = window.location;
  window.history.replaceState(null, "", `${pathname}${search}${href(page, id, query)}`);
}

// ---------------------------------------------------------------------------
// Building blocks

/** Button. kind: "primary" | "danger" | null. An async onClick disables the
 *  button while it runs and shows a rejected promise as an error toast. */
export function button(label, { kind = null, small = false, onClick = null, disabled = false, title = null, type = "button" } = {}) {
  const cls = ["btn", kind, small ? "small" : null].filter(Boolean).join(" ");
  const el = h("button", { class: cls, type, disabled, title }, label);
  if (onClick) {
    el.addEventListener("click", async (event) => {
      el.disabled = true;
      try {
        await onClick(event);
      } catch (err) {
        toast(err.message, "error");
      } finally {
        if (el.isConnected) el.disabled = disabled;
      }
    });
  }
  return el;
}

/** Link styled as a button. */
export const linkButton = (label, hash, { kind = null, small = false } = {}) =>
  h("a", { class: ["btn", kind, small ? "small" : null].filter(Boolean).join(" "), href: hash }, label);

/** Small tag. kind: null | "ok" | "no" | "host". */
export const tag = (text, kind = null) => h("span", { class: kind ? `tag ${kind}` : "tag" }, text);

/** Page heading row: { title, sub, crumbs: [[label, hash]...], actions: [nodes] }. */
export function pageHead({ title, sub = null, crumbs = null, actions = null }) {
  return h(
    "div",
    { class: "page-head" },
    h(
      "div",
      null,
      crumbs && h("div", { class: "crumbs" }, crumbs.map(([label, hash], i) => [i ? " / " : null, hash ? h("a", { href: hash }, label) : label])),
      h("h1", null, title),
      sub && h("p", { class: "sub" }, sub),
    ),
    actions && h("div", { class: "actions" }, actions),
  );
}

/** Card with a header: panel({ title, meta, actions }, ...children). */
export function panel({ title = null, meta = null, actions = null, id = null } = {}, ...children) {
  const header =
    title || meta || actions
      ? h("header", null, title && h("h2", null, title), meta && h("span", { class: "meta" }, meta), actions && h("div", { class: "actions" }, actions))
      : null;
  return h("section", { class: "panel", id }, header, ...children);
}

/** Key/value list: kv([[label, value], ...]); value may be a node; null rows are skipped. */
export const kv = (pairs) =>
  h("dl", { class: "kv" }, pairs.filter(Boolean).map(([label, value]) => [h("dt", null, label), h("dd", null, value ?? "n/a")]));

/**
 * Table. columns: [{ label, key, render(row, index) -> node|string, num, className }];
 * null columns are left out.
 * opts: { empty (text when no rows), rowClass(row) -> string, onRowClick(row) }.
 */
export function table(columns, rows, { empty = "Nothing to show.", rowClass = null, onRowClick = null, className = "" } = {}) {
  columns = columns.filter(Boolean);
  const head = h("thead", null, h("tr", null, columns.map((c) => h("th", { class: c.num ? "num" : null }, c.label))));
  const body = h(
    "tbody",
    null,
    rows.length
      ? rows.map((row, i) => {
          const tr = h(
            "tr",
            { class: rowClass ? rowClass(row) : null },
            columns.map((c) => h("td", { class: [c.num ? "num" : null, c.className].filter(Boolean).join(" ") || null }, c.render ? c.render(row, i) : row[c.key])),
          );
          if (onRowClick) {
            tr.style.cursor = "pointer";
            tr.addEventListener("click", (event) => {
              if (!event.target.closest("a, button, input, select, label")) onRowClick(row);
            });
          }
          return tr;
        })
      : h("tr", null, h("td", { colspan: columns.length, class: "empty" }, empty)),
  );
  return h("div", { class: "tablebox" }, h("table", { class: `t ${className}`.trim() }, head, body));
}

/** Tab strip. items: [{ id, label, disabled }]. Returns { element, select(id) }. */
export function tabs(items, { selected, onSelect }) {
  const buttons = items.map((item) =>
    h("button", { type: "button", role: "tab", class: item.disabled ? "off" : null, disabled: item.disabled, "aria-selected": String(item.id === selected), onClick: () => select(item.id) }, item.label),
  );
  const element = h("div", { class: "tabs", role: "tablist" }, buttons);
  function select(id) {
    buttons.forEach((b, i) => b.setAttribute("aria-selected", String(items[i].id === id)));
    onSelect(id);
  }
  return { element, select };
}

/** Warning block for a list of short texts (warnings from the API). Null when empty. */
export function warningBlock(warnings) {
  if (!warnings || !warnings.length) return null;
  return h("div", { class: "warning", role: "note" }, warnings.length === 1 ? warnings[0] : h("ul", { class: "warnings" }, warnings.map((w) => h("li", null, w))));
}

/** The dashed placeholder of a part that comes later (design section 4.9). */
export const reserved = (title, text) => h("div", { class: "reserved" }, h("b", null, title), h("span", null, text));

/** Copyable code block. */
export function codeBlock(text) {
  const pre = h("pre", { class: "mono", style: "margin:0;padding:.6rem .8rem;background:var(--code);border-radius:4px;overflow-x:auto" }, text);
  const copy = button("Copy", { small: true, onClick: () => navigator.clipboard.writeText(text).then(() => toast("Copied.")) });
  return h("div", { class: "stack" }, pre, h("div", null, copy));
}

// ---------------------------------------------------------------------------
// States

export const loadingState = (text = "Loading") => h("p", { class: "state loading", role: "status" }, text);

/** Empty state: what is missing and, optionally, the action that fills it. */
export const emptyState = (text, action = null) => h("div", { class: "state empty" }, h("p", null, text), action && h("div", { class: "actions" }, action));

/**
 * Error state for a failed request or a refusal. Shows the server's message
 * verbatim (kd's own text for code kd_error). opts: { title, retry() }.
 */
export function errorState(err, { title = "This could not be loaded.", retry = null } = {}) {
  const detail = err && err.field ? ` (field ${err.field})` : "";
  return h(
    "div",
    { class: "state error", role: "alert" },
    h("p", { class: "what" }, title),
    h("p", { class: "msg" }, `${err ? err.message : "Unknown error"}${detail}`),
    retry && h("div", { class: "actions" }, button("Try again", { small: true, onClick: retry })),
  );
}

/** Short message at the bottom left. kind: null | "warn" | "error". */
export function toast(message, kind = null, ms = 6000) {
  const box = document.getElementById("kd-toasts");
  const el = h("div", { class: kind ? `toast ${kind}` : "toast", role: kind === "error" ? "alert" : "status" }, message);
  box.append(el);
  setTimeout(() => el.remove(), ms);
  return el;
}

// ---------------------------------------------------------------------------
// Jobs, scores, memory

const STATE_VIEW = {
  queued: ["queued", "Queued"],
  running: ["running", "Running"],
  completed: ["completed", "Completed"],
  budget_stopped: ["budget", "Budget reached"],
  cancelled: ["cancelled", "Cancelled"],
  failed: ["failed", "Failed"],
  interrupted: ["interrupted", "Interrupted"],
};

/** Status badge: shape plus text (design 3.2). A running job in phase
 *  starting or preparing is shown as "Preparing". */
export function statusBadge(state, phase = null) {
  let [cls, label] = STATE_VIEW[state] || ["queued", state];
  if (state === "running" && (phase === "starting" || phase === "preparing")) [cls, label] = ["preparing", phase === "starting" ? "Starting" : "Preparing"];
  return h("span", { class: `st ${cls}` }, label);
}

const STOP_TEXT = {
  iterations_exhausted: "all iterations done",
  algorithm_done: "the algorithm finished",
  early_stopping: "stopped early by the algorithm",
  user_stop: "stopped by you, result kept",
  // kd's own word for a run a callback stopped; in kd-ui that is the stop control.
  callback: "stopped by a callback (the stop control for runs started here), result kept",
  wall_clock_budget: "time budget reached",
  wall_clock_budget_before_search: "time budget reached before the search",
  evaluation_budget: "evaluation budget reached",
  finished: "finished",
  cancelled_while_queued: "removed from the queue",
  killed: "force-stopped",
  error: "kd raised an error",
  hard_timeout: "hard time limit reached",
  memory_limit: "memory limit reached",
  system_killed: "ended by the system, probably out of memory",
  terminated: "the worker ended without a result",
  process_lost: "its processes are gone",
  machine_restarted: "the machine restarted",
};

/** Plain text of a StopReason. */
export const stopReasonText = (reason) => (reason ? STOP_TEXT[reason] || reason : "");

/** The smoothing a fit ran with, as Prepare sets it: the request's preprocessing
 *  (low-pass or network) and kd's savgol_window parameter. null when neither. */
export function smoothingText(smoothing, savgolWindow) {
  const parts = [];
  if (smoothing && smoothing.method === "lowpass") parts.push(`low-pass, ${typeof smoothing.modes === "number" ? `${smoothing.modes} modes` : Object.entries(smoothing.modes).map(([a, m]) => `${a} ${m}`).join(", ")}`);
  else if (smoothing && smoothing.method === "network") parts.push(`network smoothing, seed ${smoothing.seed}`);
  else if (smoothing) parts.push(smoothing.method);
  if (savgolWindow) parts.push(`Savitzky-Golay window ${savgolWindow}`);
  return parts.length ? parts.join(" and ") : null;
}

/** A native score with its own name and direction: "-8.935 AIC (lower is better)". */
export function nativeScore(score) {
  if (!score) return h("span", { class: "note" }, "no score yet");
  const better = score.direction === "min" ? "lower is better" : "higher is better";
  return h("span", { title: `${score.kind}, the algorithm's own score (${better})` }, fmtNumber(score.value), " ", h("span", { class: "note" }, `${score.kind} (${better})`));
}

/** Title of a job: "sga on burgers, seed 3407"; verification and comparison by run. */
export function jobTitle(job, datasetLabel = null) {
  if (job.kind === "fit") {
    const on = datasetLabel || job.dataset_id || "a dataset";
    return `${job.algorithm} on ${on}${job.seed !== null ? `, seed ${job.seed}` : ""}`;
  }
  if (job.kind === "verify") return `Verification of ${job.run_id}`;
  return "Comparison";
}

/** Thin progress bar for a fraction in [0, 1]. kind: null | "mem". */
export function progressBar(fraction, kind = null) {
  const pct = Math.max(0, Math.min(1, fraction)) * 100;
  return h("div", { class: kind ? `progress ${kind}` : "progress", role: "presentation" }, h("span", { style: `width:${pct.toFixed(1)}%` }));
}

/** Memory of one job against its cap: text line plus bar. */
export function memoryMeter(usedBytes, capBytes) {
  return [
    h("div", { class: "facts" }, usedBytes === null ? `Cap ${fmtBytes(capBytes)}` : `Memory ${fmtBytes(usedBytes)} of ${fmtBytes(capBytes)} cap`),
    usedBytes === null ? null : progressBar(usedBytes / capBytes, "mem"),
  ];
}

/**
 * The machine's memory as a segmented bar with legend, from SystemInfo.memory
 * (MemoryBudget): memory in use now, the reserve kept free, and what a new job
 * could get beside the running ones; the caps of running jobs as a legend line.
 */
export function machineMemory(memory, { jobsAtATime = null } = {}) {
  const total = memory.total_bytes;
  const inUse = Math.max(0, total - memory.available_bytes);
  const segments = [
    ["other", "In use now", inUse, "var(--context)"],
    ["reserve", "Kept free (reserve)", memory.reserve_bytes, "var(--axis)"],
    ["free", "Free for a new job", memory.admissible_bytes, null],
  ];
  const pct = (v) => `${((100 * v) / total).toFixed(2)}%`;
  const label = segments.map(([, name, v]) => `${name} ${fmtBytes(v)}`).join(", ");
  return h(
    "div",
    { class: "machine" },
    h("div", { class: "bar", role: "img", "aria-label": `Memory: ${label}, total ${fmtBytes(total)}` }, segments.map(([cls, , v]) => h("span", { class: cls, style: `width:${pct(v)}` }))),
    segments.map(([, name, v, color]) =>
      h("div", { class: "legend-row" }, h("i", { style: color ? `background:${color}` : "box-shadow:inset 0 0 0 1px var(--edge)" }), h("span", null, name), h("span", null, fmtBytes(v))),
    ),
    h("div", { class: "legend-row" }, h("i", { style: "background:var(--accent)" }), h("span", null, "Caps of running KD jobs"), h("span", null, fmtBytes(memory.running_cap_bytes))),
    h("div", { class: "line" }, h("span", null, `Total ${fmtBytes(total)}`), jobsAtATime !== null && h("span", null, `Fits at a time: ${jobsAtATime}`)),
  );
}

// ---------------------------------------------------------------------------
// Equations

/**
 * Typeset equation element from kd's display LaTeX.
 * size: "large" (1.75rem, the design's law) | "mid" | "small" | "inline".
 */
export function equation(latex, { size = "large" } = {}) {
  if (size === "inline") return renderTex(h("span", { class: "tex-inline" }), latex);
  const el = h("div", { class: size === "large" ? "law" : `law ${size}` });
  return renderTex(el, latex);
}

// ---------------------------------------------------------------------------
// Parameter form from the capability schema (ParameterField list)

const RESUME_TEXT = { resume_safe: "yes", init_only: "no", identity_breaking: "no, new run" };

const formKey = (field) => field.facade_param || field.name;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const shown = (value) => (value === null || value === undefined ? "" : typeof value === "object" ? JSON.stringify(value) : String(value));

/** Split on commas outside parentheses and brackets: "mul(u, u_x), u_xx" is two items. */
function splitTopLevel(text) {
  const out = [];
  let depth = 0;
  let cur = "";
  for (const ch of text) {
    if (ch === "(" || ch === "[") depth += 1;
    if (ch === ")" || ch === "]") depth -= 1;
    if (ch === "," && depth === 0) {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  out.push(cur);
  return out.map((s) => s.trim()).filter(Boolean);
}

function parseValue(field, raw) {
  const kind = field.json_kind;
  if (kind === "bool") return raw;
  if (raw.trim() === "") {
    if (field.nullable) return null;
    if (kind === "str") return "";
    if (kind.endsWith("_list")) return [];
    throw new Error("A value is needed.");
  }
  if (field.literal_values) return field.literal_values.find((v) => shown(v) === raw);
  const number = (text, int) => {
    const v = Number(text);
    if (text.trim() === "" || !Number.isFinite(v) || (int && !Number.isInteger(v))) throw new Error(int ? "A whole number is needed." : "A number is needed.");
    return v;
  };
  if (kind === "int") return number(raw, true);
  if (kind === "float") return number(raw, false);
  if (kind === "str") return raw;
  if (kind === "str_list") return splitTopLevel(raw);
  if (kind === "int_list" || kind === "float_list") return raw.split(",").map((s) => s.trim()).filter(Boolean).map((s) => number(s, kind === "int_list"));
  try {
    return JSON.parse(raw); // dict, object
  } catch {
    throw new Error("JSON is needed here, for example {\"key\": 1}.");
  }
}

function fieldControl(field, value, locked) {
  const kind = field.json_kind;
  const common = { name: field.name, disabled: locked, title: locked ? "Fixed for this run" : null, "aria-label": field.name };
  if (field.literal_values) {
    const options = field.literal_values.map((v) => h("option", { value: shown(v), selected: same(v, value) }, shown(v)));
    if (field.nullable) options.unshift(h("option", { value: "", selected: value === null }, "(none)"));
    return h("select", common, options);
  }
  if (kind === "bool") return h("input", { ...common, type: "checkbox", checked: Boolean(value) });
  if (kind === "dict" || kind === "object") return h("textarea", { ...common, rows: 2 }, shown(value));
  const text = kind.endsWith("_list") && Array.isArray(value) ? value.join(", ") : shown(value);
  const numeric = kind === "int" || kind === "float";
  return h("input", { ...common, type: "text", value: text, class: numeric ? "num" : null, inputmode: numeric ? "decimal" : null, placeholder: field.nullable ? "none" : null });
}

/**
 * Form for an algorithm's parameters, generated from AlgorithmInfo.fields.
 * Fields settable from Python only are left out. Fields with knob=true form
 * the basic table; the rest fold under "More parameters (n)".
 *
 * opts: { values: {key: value} (initial values, by form key or field name),
 *         lockNotResumeSafe: true on a resume form (fields that are not
 *         resume_safe are disabled), resumeColumn: show "Change on resume" }
 * Form key of a field: its facade_param when set, else its name; this is the
 * kd.Model keyword that FitRequest.params expects.
 *
 * Returns { element, values(), changes(), errors(), set(values), onChange(fn) }:
 * values() all shown fields; changes() only those that differ from the
 * default; errors() [{ key, message }] for inputs that could not be read.
 */
export function schemaForm(fields, { values = {}, lockNotResumeSafe = false, resumeColumn = true } = {}) {
  const visible = fields.filter((f) => f.settable_from !== "python" && f.json_kind !== null);
  const rows = new Map(); // key -> { field, tr, control, error, value, problem }
  const listeners = new Set();

  const read = (row) => {
    const { field, control } = row;
    const raw = field.json_kind === "bool" ? control.checked : control.value;
    try {
      row.value = parseValue(field, raw);
      row.problem = null;
    } catch (err) {
      row.problem = err.message;
    }
    row.tr.classList.toggle("changed", row.problem === null && !same(row.value, field.default));
    row.tr.classList.toggle("invalid", row.problem !== null);
    row.error.textContent = row.problem || "";
  };

  const makeRow = (field) => {
    const key = formKey(field);
    const initial = key in values ? values[key] : field.name in values ? values[field.name] : field.default;
    const locked = lockNotResumeSafe && field.resume_tier !== "resume_safe";
    const control = fieldControl(field, initial, locked);
    const error = h("span", { class: "field-error" });
    const tr = h(
      "tr",
      { class: locked ? "off" : null },
      h("td", { class: "name" }, key),
      h("td", null, control, error),
      h("td", { class: "def" }, shown(field.default) || '""'),
      resumeColumn ? h("td", null, RESUME_TEXT[field.resume_tier] || field.resume_tier) : null,
      h("td", { class: "desc" }, field.description || ""),
    );
    const row = { field, tr, control, error, value: initial, problem: null };
    control.addEventListener(field.json_kind === "bool" || field.literal_values ? "change" : "input", () => {
      read(row);
      for (const fn of listeners) fn();
    });
    rows.set(key, row);
    read(row);
    return tr;
  };

  const tableOf = (group) =>
    h(
      "div",
      { class: "tablebox" },
      h(
        "table",
        { class: "t params" },
        h("thead", null, h("tr", null, h("th", null, "Parameter"), h("th", null, "Value"), h("th", null, "Default"), resumeColumn ? h("th", null, "Change on resume") : null, h("th", null, "Description"))),
        h("tbody", null, group.map(makeRow)),
      ),
    );

  const basic = visible.filter((f) => f.knob);
  const more = visible.filter((f) => !f.knob);
  const element = h(
    "div",
    { class: "schema-form" },
    basic.length ? tableOf(basic) : null,
    more.length ? h("details", null, h("summary", null, `More parameters (${more.length})`), tableOf(more)) : null,
  );

  const collect = (onlyChanged) => {
    const out = {};
    for (const [key, row] of rows) {
      if (row.problem !== null || row.control.disabled) continue;
      if (onlyChanged && same(row.value, row.field.default)) continue;
      out[key] = row.value;
    }
    return out;
  };

  return {
    element,
    values: () => collect(false),
    changes: () => collect(true),
    errors: () => [...rows].filter(([, r]) => r.problem !== null).map(([key, r]) => ({ key, message: r.problem })),
    set(newValues) {
      for (const [key, value] of Object.entries(newValues)) {
        const row = rows.get(key);
        if (!row) continue;
        const replacement = fieldControl(row.field, value, row.control.disabled);
        row.control.replaceWith(replacement);
        replacement.addEventListener(row.field.json_kind === "bool" || row.field.literal_values ? "change" : "input", () => {
          read(row);
          for (const fn of listeners) fn();
        });
        row.control = replacement;
        read(row);
      }
    },
    onChange(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}
