# kdui frontend: shell and page contract

No framework and no build step: native ES modules, `fetch`, hash navigation.
Everything is served from `kdui/static`; no file refers to another host.
`index.html` loads `vendor/katex/katex.min.css`, `app.css`, the KaTeX UMD
script (`window.katex`) and `js/app.js`.

| File | Part | Content |
|---|---|---|
| `app.js` | shell | session start, router, header with the six steps, run tray |
| `api.js` | shell | every URL: API client, token exchange, pollers, client store |
| `ui.js` | shell | DOM builder, formats, states, badges, meters, tables, equation, schema form |
| `plots.js` | shell | ECharts wrappers that take scientific data |
| `math.js` | shell | KaTeX typesetting |
| `data.js`, `prepare.js`, `search.js`, `jobs.js`, `run.js`, `result.js`, `compare.js` | pages | one page each |

## Page contract

Each page module exports

```js
export function mount(container, params) { ... } // may be async
export function unmount() { ... } // may be async
```

- `container` is a fresh `<div class="page" data-page="<name>">` inside `<main>`.
  Build everything inside it. It is detached when the user navigates away, so
  a late async render into it is harmless; check `container.isConnected`
  before starting new work after an `await` if it matters.
- `unmount()` stops what the page started: pollers (`handle.stop()`), timers,
  window listeners, store subscriptions. The shell disposes ECharts instances
  left inside the container itself.
- A rejected `mount` is shown by the shell as an error state with "Try again".
- Navigating to the same page with other parameters unmounts and mounts again,
  unless the page exports `revisit(params)` and it returns true: then the page
  has shown the new parameters in place (the result page scrolls to another
  section of the same run). To record state in the address without
  remounting, use `replaceRoute`.
- Pages never hard-code `/api/v1` URLs and never call `fetch` directly: use `api.js`.
- Light theme only; use the classes of `app.css` (they are the design's
  `kd-ui.css`: `.panel`, `.grid-2`, `.split`, `.split-wide`, `table.t`,
  `.tablebox`, `dl.kv`, `.reads`, `.law`, `.ledger`, `.st`, `.tag`, `.btn`,
  `.tabs`, `.seg`, `.warning`, `.note`, `.reserved`, `.section-title`, `figure.fig`, ...).
  A page's own rules sit at the end of `app.css`, each scoped to
  `[data-page="<name>"]`; pages do not inject `<style>` elements.

## Routes

`#/<page>/<id>?<query>`. The router passes `params = { id, path, ...query }`:
`id` is the first segment after the page (decoded), `path` all segments after
the page, the query items as strings.

| Address | Page | `id` |
|---|---|---|
| `#/data`, `#/data/<dataset_id>`; e.g. `?view=file`, `?upload=<upload_id>` | data | dataset |
| `#/prepare` | prepare | |
| `#/search`; `?clone=<run_id>` for edit and rerun | search | |
| `#/jobs` | jobs (queue, running, ended jobs) | |
| `#/run/<job_id>` | run (one job, live) | job |
| `#/result/<run_id>`; `?section=report` | result (Verify step; Report step with the section) | run |
| `#/compare`; `?runs=<id>,<id>` | compare (history and comparison) | |

The query keys above other than `section` are read by the pages;
the shell itself reads only `section` (to underline Report instead of Verify).
The tray links a running or queued job to `#/run/<job_id>` and an ended fit
with a run to `#/result/<run_id>`. Unknown pages show an error state.

## Client store (`store` in api.js)

Kept in `localStorage`, shared across tabs. The header reacts to these keys:

| Key | Value | Written by |
|---|---|---|
| `dataset` | `{ dataset_id, label, summary }`; `summary` is a short line such as `"grid, x 256, t 201, u_t"` shown next to the name | data page when a dataset is chosen |
| `preprocessing` | `{ dataset_id, preprocessing, savgol_window, derivatives, also_raw, modes, form }`: `preprocessing` is the fit request's `Preprocessing`; `derivatives` is Prepare's choice `finite_diff`, `autograd` or `both`; `modes` the preselected `{ algorithm, mode }` pairs; `form` restores the Prepare form. Search applies it only when `dataset_id` is the dataset searched on | prepare page |
| `run` | last run id shown on the result page (Verify and Report steps link to it) | the shell, on `#/result/<id>` |

Pages may add keys of their own; use a page prefix (`search.seeds`).

## api.js

| Name | Signature | Notes |
|---|---|---|
| `get` | `get(path, query?, {signal}?) -> Promise<json>` | `path` like `"/jobs"`; query values `null` are dropped, arrays joined with `,` |
| `post` | `post(path, body = {}, {signal, query}?) -> Promise<json>` | JSON body |
| `upload` | `upload(path, blob, query?) -> Promise<json>` | raw body, e.g. `upload("/uploads", file, { filename: file.name })` |
| `request` | `request(method, path, { query, body, raw, signal }) -> Promise<json\|text\|null>` | the general form |
| `ApiError` | `class ApiError extends Error { status, code, message, field }` | every failure; `code` is the `ErrorCode` (`kd_error` carries kd's text verbatim), `"network"` when unreachable |
| `apiUrl` | `apiUrl(path, query?) -> string` | absolute URL, base-relative to the page |
| `artifactUrl` | `artifactUrl(artifactId) -> string` | for `<a>`, `<img>`, `<iframe>` of an artifact |
| `openSession` | `openSession() -> Promise<SessionInfo>` | called by the shell only |
| `pollEvery` | `pollEvery(fn, { interval = 2000, hiddenInterval = 10000, onError }) -> { stop(), now() }` | one call at a time; `fn` returning `true` stops; a 401 stops |
| `pollCursor` | `pollCursor(path, { onPage, isDone, cursor, interval = 1000, hiddenInterval = 5000, onError }) -> { stop(), now() }` | follows `EventsPage` / `LogPage`; drains while `has_more`; `isDone(page)` (may be async) ends it; handle `page.reset` in `onPage` |
| `followJob` | `followJob(jobId, { onJob, onEvents, onLog, interval = 1000, hiddenInterval = 5000, onError }) -> { stop(), now() }` | one loop: `GET /jobs/{id}`, then drains events and log; stops after the round in which the job is terminal |
| `TERMINAL_STATES` | `Set<string>` | `completed`, `budget_stopped`, `cancelled`, `failed`, `interrupted` |
| `JOBS_CHANGED` | `"kd:jobs-changed"` | window event sent after a successful `POST /jobs`, `/jobs/{id}/stop|kill`, `/runs/{id}/resume|clone`; the tray refreshes at once |
| `SYSTEM_READ` | `"kd:system-read"` | window event the shell sends each time the tray has read `GET /system` (every 2 s); `detail` is the `SystemInfo` |
| `store` | `store.get(key, fallback = null)`, `store.set(key, value)` (`null` removes), `store.on(key, fn) -> unsubscribe` | see above |

## ui.js

DOM and navigation

| Name | Signature |
|---|---|
| `h` | `h(tag, attrs?, ...children) -> Element`; attrs: `class`, `style` (string), `text`, `html` (trusted only), `on<Event>` handlers, booleans as present/absent, `null`/`false` skipped; children: nodes, strings, numbers, nested arrays |
| `frag` | `frag(...children) -> DocumentFragment` (use with `replaceChildren`, which does not flatten arrays) |
| `clear` | `clear(el) -> el` |
| `href` | `href(page, id?, query?) -> "#/page/id?k=v"` |
| `go` | `go(page, id?, query?)` navigates (remounts) |
| `replaceRoute` | `replaceRoute(page, id?, query?)` changes the address without remounting |

Formats (the KD report's conventions)

| Name | Signature |
|---|---|
| `fmtNumber` | `fmtNumber(x, sig = 4) -> string`: `1.118e-05`, `0.1002`, `"n/a"` for null |
| `fmtBytes` | `fmtBytes(bytes) -> "16.0 GiB"` (MiB below 1 GiB, KiB below 1 MiB) |
| `fmtDuration` | `fmtDuration(seconds) -> "19.5 s" \| "5 min 3 s" \| "2 h 4 min"` |
| `fmtTime` | `fmtTime(iso) -> "10:02"` today, else `"6 Nov 10:02"` |
| `elapsedSeconds` | `elapsedSeconds(startIso, endIso = null) -> number` |
| `parseSeeds` | `parseSeeds("0, 1, 2" \| "0-4, 7") -> number[]`; throws `Error` with a message for the user |

Building blocks

| Name | Signature |
|---|---|
| `button` | `button(label, { kind: "primary"\|"danger"\|null, small, onClick, disabled, title, type }) -> <button>`; an async `onClick` disables the button while it runs and turns a rejection into an error toast |
| `linkButton` | `linkButton(label, hash, { kind, small }) -> <a class="btn">` |
| `tag` | `tag(text, kind = null)`; kind `"ok"`, `"no"`, `"host"` |
| `pageHead` | `pageHead({ title, sub, crumbs: [[label, hash\|null], ...], actions: [nodes] })` |
| `panel` | `panel({ title, meta, actions, id }, ...children) -> <section class="panel">` |
| `kv` | `kv([[label, value], ...])` (`null` entries skipped) |
| `table` | `table(columns, rows, { empty, rowClass(row), onRowClick(row), className })`; column `{ label, key, render(row, i), num, className }` (null columns are left out); returns `div.tablebox` |
| `tabs` | `tabs([{ id, label, disabled }], { selected, onSelect(id) }) -> { element, select(id) }` |
| `warningBlock` | `warningBlock(warnings: string[]) -> Element \| null` (null when empty); for every `warnings` list of the API |
| `reserved` | `reserved(title, text)`: the dashed placeholder of a part that comes later (the compare page's consensus) |
| `codeBlock` | `codeBlock(text)`: code with a Copy button (e.g. the equivalent `kd.load` call) |

States

| Name | Signature |
|---|---|
| `loadingState` | `loadingState(text = "Loading")` |
| `emptyState` | `emptyState(text, action = null)` |
| `errorState` | `errorState(err, { title, retry })`: shows `err.message` verbatim (kd's own text) and the field |
| `toast` | `toast(message, kind = null \| "warn" \| "error", ms = 6000)` |

Jobs, scores, memory

| Name | Signature |
|---|---|
| `statusBadge` | `statusBadge(state, phase = null)`: shape plus text (design 3.2); `running` in phase `starting`/`preparing` shows as Starting/Preparing |
| `stopReasonText` | `stopReasonText(stopReason) -> string` ("memory limit reached", "ended by the system, probably out of memory", ...) |
| `nativeScore` | `nativeScore(NativeScore) -> Element`: value, kind and direction ("lower is better") |
| `jobTitle` | `jobTitle(job, datasetLabel = null) -> "sga on burgers, seed 3407"` |
| `progressBar` | `progressBar(fraction, kind = null \| "mem")`; only for a real fraction, never an invented percentage |
| `memoryMeter` | `memoryMeter(usedBytes \| null, capBytes) -> [nodes]`: "Memory 1.5 GiB of 16.0 GiB cap" plus bar |
| `machineMemory` | `machineMemory(SystemInfo.memory, { jobsAtATime })`: segmented bar and legend |

Equations and forms

| Name | Signature |
|---|---|
| `equation` | `equation(latex, { size: "large"\|"mid"\|"small"\|"inline" })`: kd's display LaTeX, typeset |
| `schemaForm` | `schemaForm(AlgorithmInfo.fields, { values, lockNotResumeSafe = false, resumeColumn = true }) -> { element, values(), changes(), errors(), set(values), onChange(fn) }` |

`schemaForm` leaves out fields settable from Python only (`settable_from ==
"python"` or `json_kind == null`). `knob` fields form the basic table, the
rest fold under "More parameters (n)". Keys are the `kd.Model` keywords that
`FitRequest.params` expects: the field's `facade_param` when set, else its
`name`. `values()` gives every shown field, `changes()` only those that differ
from the default (send these as `params`), `errors()` lists inputs that could
not be read (`[{ key, message }]`). `literal_values` become a select, `bool` a
checkbox, lists a comma-separated text, `dict`/`object` JSON in a textarea.
Changed rows carry a "changed" mark. The "Change on resume" column maps
`resume_tier`: `resume_safe` yes, `init_only` no, `identity_breaking` "no, new
run". On a resume form pass `lockNotResumeSafe: true`: the other fields are
disabled ("Fixed for this run") and left out of `values()`.

## plots.js

All take data, not chart options, and return `{ chart, update(data), dispose() }`.
Charts resize with their element; give it a height (`.plot` 260 px,
`.plot.tall` 360 px, `.plot.short` 180 px). Colours are `style.py`'s.

| Name | Signature |
|---|---|
| `linePlot` | `linePlot(el, { series: [{ name, x, y, step }], xLabel, yLabel, logY, marks: [{ x, y }], remainingFrom, xMax })`: one y axis; several series get a legend and end labels; `remainingFrom`..`xMax` shaded as "remaining budget" |
| `scatterPlot` | `scatterPlot(el, { series: [{ name, x, y, labels }], xLabel, yLabel, logX, logY })` |
| `dotPlot` | `dotPlot(el, { rows: [{ label, value, note }], xLabel, logX = true })`: one value per row (e.g. re-measured nmse); `value: null` is "not measured"; sets the element height |
| `fieldHeatmap` | `fieldHeatmap(el, FieldSlice, { diverging = false })`: grid over two axes, rows (`axes[0]`) upward, columns (`axes[1]`) to the right; viridis, or kd_div symmetric about zero; `null` cells blank |
| `fieldPlot` | `fieldPlot(el, FieldSlice, opts)`: line for one axis, heatmap for a grid, scatter coloured by value for scattered points |
| `disposeWithin` | `disposeWithin(root)` (the shell calls it on unmount) |
| `SERIES`, `VIRIDIS`, `KD_DIV` | colour lists |

Only charts that update or need hover are drawn here (live progress, previews,
comparison summaries). Figures of a finished run are KD's report figures
(SVG artifacts), shown as images, not redrawn. A sampled slice is for display:
compute nothing from it.

## math.js

| Name | Signature |
|---|---|
| `renderTex` | `renderTex(el, latex, { display = false }) -> el`; unparsable LaTeX is shown as plain text (class `tex-fallback`) |
| `texToHtml` | `texToHtml(latex, { display }) -> string` |
| `typeset` | `typeset(root = document)`: renders every `[data-tex]` element (`data-display="true"` for display style) |

## Vendored libraries

`vendor/echarts/echarts.esm.min.js` (Apache ECharts 6.1.0, ESM build, with
`LICENSE`, `NOTICE` and the bundled d3 licence) and `vendor/katex/` (KaTeX
0.16.11 from `website/docs/javascripts/katex`: `katex.min.js`, `katex.min.css`,
the WOFF2 fonts, `LICENSE`). The KaTeX stylesheet also names WOFF and TTF
fonts; browsers take the WOFF2 file first, so only those ship.
