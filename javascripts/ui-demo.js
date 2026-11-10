/* A static interface preview. Only bundled samples are fetched; Run never starts a job. */
document.addEventListener("DOMContentLoaded", async () => {
  const root = document.querySelector("[data-kd-demo]");
  if (!root) return;
  let data;
  try {
    const response = await fetch(root.dataset.source);
    if (!response.ok) throw new Error(`Sample data: HTTP ${response.status}`);
    data = await response.json();
  } catch (error) {
    root.textContent = `The preview could not load: ${error.message}. See the KD Studio quick start for local setup.`;
    return;
  }
  const initial = () => ({ step: "data", dataset: null, time: 0, view: "field" });
  let state = initial();
  root.innerHTML = `
    <header class="demo-top"><strong class="demo-brand">KD Studio</strong><span class="demo-dataset">No dataset selected</span><span class="demo-badge">Preview</span><button type="button" class="demo-reset">Reset</button></header>
    <div class="demo-notice"></div>
    <div class="demo-all-algorithms" aria-label="Available algorithms"></div>
    <nav class="demo-nav" aria-label="Preview steps">
      <button type="button" data-step="data"><span>1</span>Data</button>
      <button type="button" data-step="prepare"><span>2</span>Prepare</button>
      <button type="button" data-step="search"><span>3</span>Search</button>
      <button type="button" data-step="example"><span>4</span>Recorded result</button>
    </nav>
    <div class="demo-body">
      <section data-panel="data" aria-label="Preview data">
        <div class="demo-section-head"><h3>Choose a field</h3><button type="button" class="demo-primary" data-next="prepare">Continue to Prepare</button></div>
        <div class="demo-grid"><aside class="demo-box"><h4>Sample catalog</h4><div class="demo-samples"></div><p class="demo-small">Your own files can be loaded in the local app. This preview uses bundled examples.</p>
          <details><summary>More datasets and file inputs</summary><label>Full KD catalog<select class="demo-catalog" aria-label="Full KD catalog"></select></label><p class="demo-catalog-info demo-small"></p><pre class="demo-catalog-code"></pre>
            <label>Your file format<select class="demo-file-format" aria-label="File format"><option>NPZ / NPY</option><option>MATLAB</option><option>HDF5</option><option>CSV</option><option>XLSX</option></select></label><p class="demo-file-hint demo-small"></p><p class="demo-small">The local app supports grid and scattered fields, feature tables and ODE trajectories. File mapping and uploads are performed locally.</p><a class="demo-data-guide">Data loading guide</a></details></aside>
          <div class="demo-box demo-preview"><div class="demo-empty"><strong>Select a dataset</strong><p>Burgers and KdV are available as browser previews.</p><span>No installation required.</span></div>
            <div class="demo-field" hidden><div class="demo-section-head"><h4 class="demo-field-title"></h4><span class="demo-original">Original field</span></div>
              <div class="demo-view" role="group" aria-label="Field view"><button type="button" data-view="field">Field</button><button type="button" data-view="profile">Time profile</button></div>
              <canvas class="demo-field-chart" height="280" role="img" aria-label="Observed sample field"></canvas>
              <div class="demo-scale"><span></span><i aria-hidden="true"></i><span></span></div>
              <label class="demo-time" hidden>Time <output></output><input type="range" min="0" value="0" aria-label="Preview time"></label>
              <p class="demo-field-description demo-small"></p>
            </div>
          </div></div>
      </section>
      <section data-panel="prepare" aria-label="Preview preparation" hidden>
        <div class="demo-section-head"><h3>Prepare the input</h3><button type="button" class="demo-primary" data-next="search">Continue to Search</button></div>
        <div class="demo-grid"><form class="demo-box demo-preparation"><h4>Input choices</h4>
          <label>Derivatives<select name="derivatives" aria-label="Derivative method"></select></label>
          <label>Smoothing<select name="smoothing"></select></label>
          <label class="demo-window" hidden>Window length<input name="window" type="number" min="5" max="65" step="2" value="7" required disabled><small>Odd values from 5 to 65 in this preview.</small></label>
          <div class="demo-lowpass" hidden><label>Spatial modes to keep<input name="modes" type="number" min="1" step="1" value="32" required disabled></label><label>Time-axis window<select name="time_window" disabled><option value="0">Off</option><option value="7">7 points</option></select></label></div>
          <label class="demo-network" hidden>Network seed<input name="network_seed" type="number" min="0" step="1" value="0" required disabled><small>Trains KD's default field-network recipe locally.</small></label>
        </form><div class="demo-box"><h4>Input for the search</h4><dl class="demo-prep-summary"></dl><p class="demo-prep-note demo-small"></p><button type="button" data-next="data">Inspect the field</button></div></div>
        <div class="demo-box demo-processing"><div class="demo-section-head"><h4>Processing example</h4><label>Show<select class="demo-processing-view" aria-label="Processing example view"><option value="field">Field before and after</option><option value="target">Target derivative</option><option value="spectrum">Spatial spectrum</option></select></label></div><p class="demo-processing-caption demo-small"></p><div class="demo-processing-presets"></div><canvas class="demo-processing-chart" height="245" role="img" aria-label="Recorded processing example"></canvas><div class="demo-processing-legend"></div><p class="demo-processing-hint demo-small" role="status"></p>
          <details><summary>Other processing tools</summary><p>Noise and grid-spacing inspection are available when loading your data. Automatic low-pass selection (<code>kd.select_lowpass_modes</code>) refits a supplied candidate equation at several cutoffs, so it is used after choosing candidate terms. The local UI can queue each run on the original data and its smoothed copy for comparison. Neural smoothing and automatic differentiation train a field network locally; DLGA can also reuse a supplied trained surrogate.</p><a class="demo-derivatives-guide">Derivative sources and setup</a></details></div>
      </section>
      <section data-panel="search" aria-label="Preview search" hidden>
        <div class="demo-section-head"><h3>Set up a search</h3><span class="demo-original">Changes stay in this browser</span></div>
        <div class="demo-grid"><form class="demo-box demo-search"><h4>Algorithm and parameters</h4>
          <label>Algorithm<select name="algorithm" aria-label="Preview algorithm"></select></label>
          <label>Mode<select name="mode" aria-label="Algorithm mode"></select></label>
          <p class="demo-mode-summary demo-small"></p><p class="demo-mode-status demo-small" role="status"></p>
          <label class="demo-generations">Generations<input name="generations" type="number" min="1" step="1" required><small class="demo-generation-help"></small></label>
          <p class="demo-one-shot demo-small" hidden>This algorithm performs a one-shot fit; it does not use a generation budget.</p>
          <label>Seed<input name="seed" type="number" min="0" step="1" required></label>
          <details class="demo-parameter-details"><summary>Main algorithm parameters</summary><div class="demo-extra-params"></div></details>
          <p class="demo-requirements demo-small"></p><a class="demo-algorithm-guide">Setup and worked example</a>
        </form><div class="demo-box"><h4>Your configuration</h4><dl class="demo-run-summary"></dl>
          <div class="demo-actions"><button type="button" class="demo-check">Check settings</button><button type="button" class="demo-primary demo-run">Run search</button></div>
          <p class="demo-feedback" role="status" aria-live="polite"></p>
        </div></div>
      </section>
      <section data-panel="example" aria-label="Recorded example" hidden>
        <div class="demo-section-head"><h3>A recorded Burgers result</h3><span class="demo-badge">Recorded example</span></div>
        <p class="demo-example-provenance"></p>
        <div class="demo-box"><div class="demo-equation"></div><div class="demo-metrics"></div>
          <canvas class="demo-coefficients" height="190" role="img" aria-label="Coefficients of the recorded equation"></canvas>
          <p class="demo-small">NMSE measures the fitted time-derivative target. Forward error measures the relative L2 error of the integrated field; these are different checks.</p>
        </div>
        <div class="demo-actions"><button type="button" data-next="data">Edit configuration</button><a class="demo-local-link">KD Studio quick start</a></div>
      </section>
    </div>
    <dialog class="demo-dialog" aria-labelledby="demo-dialog-title">
      <button type="button" class="demo-close" aria-label="Close run information">×</button>
      <span class="demo-badge">Preview mode</span><h3 id="demo-dialog-title">Run locally</h3>
      <p>No search was started. The command below installs and starts the local KD Studio app. The recorded Burgers result uses its own settings.</p><p class="demo-local-requirements"></p>
      <h4>Run KD Studio locally</h4><pre class="demo-install"></pre><a class="demo-local-link">KD Studio quick start</a>
      <details class="demo-recipe-details"><summary>Use these settings in Python</summary><pre class="demo-recipe"></pre><button type="button" class="demo-copy">Copy Python recipe</button><span class="demo-copy-status" role="status"></span></details><p class="demo-recipe-hint demo-small"></p><a class="demo-dialog-guide">Selected algorithm guide</a>
      <div class="demo-actions"><button type="button" class="demo-show-example">Recorded Burgers result</button><button type="button" class="demo-back">Edit settings</button></div>
    </dialog>`;

  const $ = (selector) => root.querySelector(selector);
  const prep = $(".demo-preparation");
  const form = $(".demo-search");
  const algorithm = () => data.algorithms.find((a) => a.id === form.elements.algorithm.value);
  const mode = () => algorithm().modes.find((m) => m.name === form.elements.mode.value);
  const sample = () => data.datasets.find((d) => d.id === state.dataset);
  const smoothing = () => prep.elements.smoothing.value === "savgol" ? Number(prep.elements.window.value) : 0;
  const names = { sga: "SGA-PDE", dlga: "DLGA-PDE", discover: "DISCOVER", eqgpt: "EqGPT", llm4ed: "LLM4ED", pysr: "PySR", pysindy: "PySINDy" };
  const providerNames = { finite_diff: "Finite differences", fd2: "FD2 (second-order accuracy)", autograd: "Automatic differentiation", none: "No derivatives (this mode)" };
  const usesBudget = () => algorithm().params.generations.effect !== "unused";
  const knobCache = new Map();
  let knobKey = "";
  const localURL = root.dataset.guide;
  const docsBase = new URL("../", new URL(root.dataset.source, document.baseURI));
  root.querySelectorAll(".demo-local-link").forEach((link) => link.href = localURL);
  $(".demo-data-guide").href = new URL("data/shape/", docsBase).href;
  $(".demo-derivatives-guide").href = new URL("data/derivatives/", docsBase).href;
  $(".demo-notice").textContent = `The preview provides settings for ${data.algorithms.length} algorithms. Searches and training run in the local KD Studio app.`;
  $(".demo-all-algorithms").textContent = data.algorithms.map((a) => names[a.id]).join(" · ");
  data.algorithms.forEach((a) => form.elements.algorithm.add(new Option(names[a.id], a.id)));
  const providers = [...new Set(data.algorithms.flatMap((a) => a.modes.flatMap((m) => [m.provider_kind, ...m.derivative_providers])))];
  providers.forEach((p) => prep.elements.derivatives.add(new Option(providerNames[p] || p, p)));
  data.processing_options.forEach((p) => prep.elements.smoothing.add(new Option(p.label, p.id)));
  data.catalog.forEach((d) => $(".demo-catalog").add(new Option(`${d.id} · ${d.topology}`, d.id)));
  function catalogHint() {
    const d = data.catalog.find((row) => row.id === $(".demo-catalog").value);
    $(".demo-catalog-info").textContent = `${d.axes.join(", ")} coordinates; ${d.lhs ? "target " + d.lhs + "; " : ""}${d.fmt} data; ${d.tier === "remote" ? "downloaded on first use" : "bundled"}. Load this dataset in the local app. The two sample buttons above have browser previews.`;
    $(".demo-catalog-code").textContent = `data = kd.load(${JSON.stringify(d.id)})`;
  }
  function fileHint() {
    const name = $(".demo-file-format").value;
    $(".demo-file-hint").textContent = `${name}: choose File in the local UI, inspect the arrays or columns, then map coordinates, fields and the target. MATLAB v7.3 uses HDF5; CSV and XLSX can also hold tables or ODE trajectories. Files are not uploaded by this preview.`;
  }
  $(".demo-catalog").addEventListener("change", catalogHint);
  $(".demo-file-format").addEventListener("change", fileHint);
  data.datasets.forEach((d) => {
    const button = document.createElement("button");
    button.type = "button"; button.dataset.dataset = d.id;
    const name = document.createElement("strong"); name.textContent = d.id === "kdv" ? "KdV" : "Burgers";
    const shape = document.createElement("span"); shape.textContent = `${d.shape[0]} × ${d.shape[1]} observed values`;
    button.append(name, shape);
    button.addEventListener("click", () => { state.dataset = d.id; state.time = 0; state.view = "field"; update(); });
    $(".demo-samples").append(button);
  });
  function defaults() {
    knobCache.clear(); knobKey = "";
    prep.reset(); form.reset();
    form.elements.algorithm.value = "sga";
    form.elements.generations.value = data.recorded.generations;
    form.elements.seed.value = data.algorithms.find((a) => a.id === "sga").params.seed.default;
    prep.elements.smoothing.value = "none";
    $(".demo-processing-view").value = "field";
    $(".demo-catalog").selectedIndex = 0;
    $(".demo-file-format").selectedIndex = 0;
    setAlgorithm(); catalogHint(); fileHint();
  }
  function allowedProviders(m) { return [...new Set([m.provider_kind, ...m.derivative_providers])]; }
  function rememberKnobs() {
    if (!knobKey) return;
    const values = {};
    root.querySelectorAll("[data-knob]").forEach((input) => values[input.dataset.knob] = input.type === "checkbox" ? input.checked : input.value);
    knobCache.set(knobKey, values);
  }
  function setAlgorithm() {
    rememberKnobs();
    form.elements.mode.replaceChildren();
    algorithm().modes.forEach((m) => form.elements.mode.add(new Option(m.name, m.name)));
    setMode();
  }
  function setMode() {
    rememberKnobs();
    const m = mode();
    if (!allowedProviders(m).includes(prep.elements.derivatives.value)) prep.elements.derivatives.value = m.provider_kind;
    renderKnobs();
  }
  function renderKnobs() {
    const a = algorithm(), m = mode(); knobKey = `${a.id}/${m.name}`;
    const values = knobCache.get(knobKey) || {};
    $(".demo-extra-params").replaceChildren();
    for (const field of a.fields) {
      const label = document.createElement("label"); label.textContent = field.facade_param || field.name;
      if (!["int", "float", "str", "bool"].includes(field.json_kind)) {
        const hint = document.createElement("small"); hint.textContent = "Structured setting: configure this in the local app. " + field.description;
        label.append(hint); $(".demo-extra-params").append(label); continue;
      }
      const input = document.createElement(field.literal_values ? "select" : "input");
      input.dataset.knob = field.name;
      if (field.literal_values) {
        if (field.nullable) input.add(new Option("Not set", ""));
        field.literal_values.forEach((v) => input.add(new Option(String(v), String(v))));
      } else input.type = field.json_kind === "bool" ? "checkbox" : ["int", "float"].includes(field.json_kind) ? "number" : "text";
      if (input.type === "number") input.step = field.json_kind === "int" ? "1" : "any";
      const value = values[field.name] ?? field.default;
      if (input.type === "checkbox") input.checked = Boolean(value); else input.value = value === null ? "" : value;
      input.required = m.required_params.includes(field.name) || !field.nullable;
      if (input.type === "checkbox") input.required = false;
      input.disabled = field.name in m.refused_fields;
      const hint = document.createElement("small"); hint.textContent = `${field.description}${input.disabled ? " Fixed in this mode." : ""}${m.required_params.includes(field.name) ? " Required by this mode." : ""}`;
      label.append(input, hint); $(".demo-extra-params").append(label);
    }
  }
  function localRequirements() {
    const a = algorithm(), m = mode(), notes = [];
    if (a.requirements.extras.length) notes.push(`Install sail-kd[${a.requirements.extras.join(",")}]; PySR downloads Julia on first use.`);
    a.requirements.environment.filter((e) => e.required).forEach((e) => notes.push(`Configure ${e.name}${e.unless_argument ? " or an explicit " + e.unless_argument : ""} locally. This preview never asks for credentials.`));
    if (a.id === "eqgpt") notes.push("EqGPT needs pretrained assets, downloaded on first use or supplied locally.");
    if (m.provider_kind === "autograd") notes.push("This mode uses a trained field network for derivatives.");
    if (m.preferred_device === "cuda") notes.push("A GPU is preferred for this mode.");
    return notes.join(" ");
  }
  function setupIssues() {
    const d = sample(), a = algorithm(), m = mode(), issues = [];
    if (d && !d.applicability[`${a.id}/${m.name}`].accepts) issues.push(d.applicability[`${a.id}/${m.name}`].reason);
    if (!allowedProviders(m).includes(prep.elements.derivatives.value)) issues.push(`Select ${allowedProviders(m).map((p) => providerNames[p]).join(" or ")} for this mode, or select another mode.`);
    m.required_params.forEach((name) => {
      const field = a.fields.find((f) => f.name === name);
      if (!field || !["int", "float", "str", "bool"].includes(field.json_kind)) issues.push(`Configure ${name} in the local app; see this algorithm's guide.`);
    });
    return issues;
  }
  function parameterChanges() {
    const changes = {};
    algorithm().fields.forEach((field) => {
      const input = root.querySelector(`[data-knob="${field.name}"]`);
      if (!input || input.disabled) return;
      const value = input.type === "checkbox" ? input.checked : input.value === "" ? null : ["int", "float"].includes(field.json_kind) ? Number(input.value) : input.value;
      if (value !== field.default || mode().required_params.includes(field.name)) changes[field.facade_param || field.name] = value;
    });
    return changes;
  }
  function definitionList(target, rows) {
    target.replaceChildren();
    rows.forEach(([label, value]) => {
      const dt = document.createElement("dt"); dt.textContent = label;
      const dd = document.createElement("dd"); dd.textContent = value;
      target.append(dt, dd);
    });
  }
  function show(step, focus = false) {
    state.step = step;
    root.querySelectorAll("[data-panel]").forEach((panel) => panel.hidden = panel.dataset.panel !== step);
    root.querySelectorAll(".demo-nav button").forEach((button) => {
      button.setAttribute("aria-current", button.dataset.step === step ? "step" : "false");
    });
    update();
    if (focus) {
      root.scrollIntoView({ block: "start", behavior: "auto" });
      const heading = $(`[data-panel="${step}"] h3`); heading.tabIndex = -1; heading.focus({ preventScroll: true });
    }
  }
  function valid() {
    if (!prep.checkValidity()) { show("prepare"); prep.reportValidity(); return false; }
    if (!form.checkValidity()) { $(".demo-parameter-details").open = true; show("search"); form.reportValidity(); return false; }
    return true;
  }
  function update() {
    const d = sample(); const a = algorithm(); const m = mode();
    const provider = providerNames[prep.elements.derivatives.value];
    $(".demo-dataset").textContent = d ? `${d.id} · u(x, t)` : "No dataset selected";
    root.querySelectorAll('[data-next="prepare"], [data-next="search"], .demo-nav [data-step="prepare"], .demo-nav [data-step="search"]').forEach((b) => b.disabled = !d);
    root.querySelectorAll("[data-dataset]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.dataset === state.dataset)));
    const kind = prep.elements.smoothing.value;
    const smooth = kind === "savgol";
    $(".demo-window").hidden = !smooth; prep.elements.window.disabled = !smooth;
    $(".demo-lowpass").hidden = kind !== "lowpass";
    prep.elements.modes.disabled = kind !== "lowpass"; prep.elements.time_window.disabled = kind !== "lowpass";
    $(".demo-network").hidden = kind !== "network"; prep.elements.network_seed.disabled = kind !== "network";
    $(".demo-generations").hidden = !usesBudget(); form.elements.generations.disabled = !usesBudget();
    $(".demo-one-shot").hidden = usesBudget();
    $(".demo-generation-help").textContent = `KD default: ${a.params.generations.default}. ${a.id === "sga" ? "The recorded Burgers example uses 20." : a.params.generations.effect === "config_field" ? "This controls the algorithm's internal search iterations." : "Iteration budget for this search."}`;
    $(".demo-mode-summary").textContent = `${a.summary} ${m.description} Data: ${m.topologies.join(", ")}. LHS derivative orders: ${m.lhs_orders.join(", ")}.${m.spatial_dims ? " Spatial dimensions: " + m.spatial_dims.join(", ") + "." : ""} Native score: ${a.score_kind}.`;
    const issues = setupIssues();
    $(".demo-mode-status").textContent = issues.join(" ");
    $(".demo-mode-status").classList.toggle("demo-warning", issues.length > 0);
    $(".demo-requirements").textContent = localRequirements();
    const guide = new URL(`algorithms/${a.id}/`, docsBase).href;
    $(".demo-algorithm-guide").href = guide; $(".demo-dialog-guide").href = guide;
    $(".demo-prep-note").textContent = (kind === "network" ? "Network smoothing trains a neural field with KD's default recipe. Training and network derivatives are performed locally; a worked neural example is linked in the guide." : "The examples below were computed with KD on a fixed noisy copy. Your local recipe uses the selected dataset as loaded, with these processing choices.") + (issues.length ? " " + issues.join(" ") : "");
    $(".demo-empty").hidden = Boolean(d); $(".demo-field").hidden = !d;
    if (d) {
      $(".demo-field-title").textContent = `${d.id}: u(x, t)`;
      $(".demo-time").hidden = state.view !== "profile";
      $(".demo-time input").max = d.t.length - 1;
      $(".demo-time input").value = state.time;
      $(".demo-time output").textContent = `t = ${d.t[state.time].toPrecision(4)}`;
      $(".demo-scale").hidden = state.view !== "field";
      $(".demo-scale span:first-child").textContent = `u = ${d.range[0].toFixed(2)}`;
      $(".demo-scale span:last-child").textContent = d.range[1].toFixed(2);
      $(".demo-field-description").textContent = `Downsampled preview of the bundled ${d.shape[0]} × ${d.shape[1]} field. Click the field or use Time profile to inspect an observed slice.`;
      root.querySelectorAll("[data-view]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === state.view)));
      prep.elements.modes.max = d.shape[0];
      const description = { none: "None", savgol: `Savitzky–Golay, window ${prep.elements.window.value || "…"}`, lowpass: `Low-pass, ${prep.elements.modes.value || "…"} spatial modes, time window ${prep.elements.time_window.value}`, network: `Neural field, seed ${prep.elements.network_seed.value || "…"}` }[kind];
      const rows = [["Dataset", d.id], ["Target", d.lhs], ["Derivatives", provider], ["Smoothing", description]];
      definitionList($(".demo-prep-summary"), rows);
      definitionList($(".demo-run-summary"), [...rows, ["Algorithm", names[a.id]], ["Mode", m.name], ["Seed", form.elements.seed.value || "Enter a seed"], ["Search budget", usesBudget() ? `${form.elements.generations.value || "…"} iterations` : "One-shot fit"], ...Object.entries(parameterChanges()).map(([key, value]) => [key, value === null ? "Not set" : String(value)])]);
    }
    if (state.step === "data" && d) drawField();
    if (state.step === "prepare" && d) drawProcessing();
    if (state.step === "example") drawResult();
  }
  function recipe() {
    const a = algorithm(), m = mode();
    const params = { algorithm: a.id, ...m.selected_by, seed: Number(form.elements.seed.value) };
    if ("derivatives" in a.params) params.derivatives = prep.elements.derivatives.value;
    if (usesBudget()) params.generations = Number(form.elements.generations.value);
    if (smoothing()) params.savgol_window = smoothing();
    Object.assign(params, parameterChanges());
    const py = (value) => value === null ? "None" : value === true ? "True" : value === false ? "False" : Array.isArray(value) ? `[${value.map(py).join(", ")}]` : JSON.stringify(value);
    let code = `import kd\n\n# ${localRequirements() || "Configure and run locally."}\ndata = kd.load("${state.dataset}")\n`;
    if (prep.elements.smoothing.value === "lowpass") code += `data = kd.lowpass_smooth(data, modes={"x": ${Number(prep.elements.modes.value)}}, time_window=${Number(prep.elements.time_window.value)})\n`;
    if (prep.elements.smoothing.value === "network") code += `from kd.models import DLGA_NN_1\ndata = kd.network_smooth(data, recipe=DLGA_NN_1, seed=${Number(prep.elements.network_seed.value)}).dataset\n`;
    return code + `model = kd.Model(${Object.entries(params).map(([key, value]) => `${key}=${py(value)}`).join(", ")}).fit(data)\nprint(model.result_.equation)\nmodel.report("out/demo")`;
  }
  root.querySelectorAll("[data-next], .demo-nav [data-step]").forEach((button) => button.addEventListener("click", () => {
    const next = button.dataset.next || button.dataset.step;
    if (next === "search" && !prep.checkValidity()) { show("prepare"); prep.reportValidity(); return; }
    show(next, true);
  }));
  root.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => { state.view = b.dataset.view; update(); }));
  $(".demo-time input").addEventListener("input", (event) => { state.time = Number(event.target.value); update(); });
  $(".demo-field-chart").addEventListener("click", (event) => {
    if (state.view !== "field") return;
    const bounds = event.currentTarget.getBoundingClientRect(); const x = event.clientX - bounds.left;
    if (x < 46 || x > bounds.width - 20) return;
    state.time = Math.round((x - 46) / (bounds.width - 66) * (sample().t.length - 1));
    state.view = "profile"; update();
  });
  [prep, form].forEach((f) => f.addEventListener("submit", (event) => event.preventDefault()));
  form.addEventListener("input", (event) => {
    if (event.target === form.elements.algorithm) setAlgorithm();
    else if (event.target === form.elements.mode) setMode();
    else rememberKnobs();
    $(".demo-feedback").textContent = ""; update();
  });
  prep.addEventListener("input", (event) => {
    if (event.target === prep.elements.derivatives && !allowedProviders(mode()).includes(event.target.value)) {
      const match = algorithm().modes.find((m) => allowedProviders(m).includes(event.target.value) && sample().applicability[`${algorithm().id}/${m.name}`].accepts);
      if (match) { rememberKnobs(); form.elements.mode.value = match.name; renderKnobs(); }
    }
    $(".demo-feedback").textContent = ""; update();
  });
  $(".demo-processing-view").addEventListener("change", drawProcessing);
  $(".demo-reset").addEventListener("click", () => { state = initial(); defaults(); $(".demo-feedback").textContent = ""; show("data"); });
  $(".demo-check").addEventListener("click", () => {
    if (valid()) $(".demo-feedback").textContent = setupIssues().length ? setupIssues().join(" ") : "Inputs checked. The local KD Studio app validates the complete configuration before running.";
  });
  $(".demo-run").addEventListener("click", () => {
    if (!valid()) return;
    const issues = setupIssues();
    $("#demo-dialog-title").textContent = issues.length ? `${names[algorithm().id]} setup` : "Run locally";
    $(".demo-local-requirements").textContent = [...issues, localRequirements()].filter(Boolean).join(" ");
    $(".demo-recipe-details").hidden = issues.length > 0;
    $(".demo-recipe-hint").textContent = issues.length ? "This mode needs additional local setup. Follow its guide for the matching data and complete configuration." : "";
    $(".demo-recipe").textContent = issues.length ? "" : recipe();
    const extras = ["ui", ...algorithm().requirements.extras].join(",");
    $(".demo-install").textContent = `uv pip install "sail-kd[${extras}]" --torch-backend=auto\nkd-ui --workspace ./kd-ui-workspace`;
    $(".demo-copy-status").textContent = "";
    $(".demo-dialog").showModal();
  });
  [".demo-close", ".demo-back"].forEach((s) => $(s).addEventListener("click", () => $(".demo-dialog").close()));
  $(".demo-show-example").addEventListener("click", () => { $(".demo-dialog").close(); show("example", true); });
  $(".demo-copy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($(".demo-recipe").textContent); $(".demo-copy-status").textContent = "Copied."; }
    catch { $(".demo-copy-status").textContent = "Clipboard access was refused. Select the recipe above to copy it."; }
  });

  // Both views are drawn from observed samples, never from the editable search settings.
  function surface(canvas) {
    const width = canvas.clientWidth; const height = canvas.height / (canvas.dataset.ratio || 1);
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); canvas.dataset.ratio = ratio;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio); ctx.clearRect(0, 0, width, height);
    ctx.font = '12px "IBM Plex Sans", sans-serif'; ctx.fillStyle = "#52514e";
    return { ctx, width, height };
  }
  const colors = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  function color(value) {
    const p = Math.max(0, Math.min(1, value)) * (colors.length - 1); const k = Math.min(colors.length - 2, Math.floor(p));
    return colors[k].map((v, i) => Math.round(v + (colors[k + 1][i] - v) * (p - k)));
  }
  function drawField() {
    const d = sample(); const { ctx, width, height } = surface($(".demo-field-chart"));
    const left = 46, top = 16, w = width - 66, h = height - 56;
    if (state.view === "field") {
      const off = document.createElement("canvas"); off.width = d.t.length; off.height = d.x.length;
      const raster = off.getContext("2d"); const pixels = raster.createImageData(off.width, off.height);
      for (let x = 0; x < d.x.length; x++) for (let t = 0; t < d.t.length; t++) {
        const index = ((d.x.length - 1 - x) * d.t.length + t) * 4;
        pixels.data.set([...color((d.u[x][t] - d.range[0]) / (d.range[1] - d.range[0])), 255], index);
      }
      raster.putImageData(pixels, 0, 0); ctx.drawImage(off, left, top, w, h);
      ctx.fillText(d.x.at(-1).toFixed(2), 1, top + 8); ctx.fillText(d.x[0].toFixed(2), 1, top + h);
      ctx.fillText(d.t[0].toFixed(1), left, height - 20); ctx.fillText(d.t.at(-1).toFixed(1), left + w - 22, height - 20);
      ctx.fillText("t", left + w / 2, height - 4); ctx.fillText("x", 12, top + h / 2);
    } else {
      ctx.strokeStyle = "#e1e0d9"; ctx.strokeRect(left, top, w, h);
      ctx.beginPath(); ctx.strokeStyle = "#0e716c"; ctx.lineWidth = 2;
      d.u.forEach((row, i) => {
        const x = left + w * i / (d.x.length - 1); const y = top + h * (1 - (row[state.time] - d.range[0]) / (d.range[1] - d.range[0]));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }); ctx.stroke();
      ctx.fillText(d.range[1].toFixed(2), 1, top + 8); ctx.fillText(d.range[0].toFixed(2), 1, top + h);
      ctx.fillText(d.x[0].toFixed(2), left, height - 20); ctx.fillText(d.x.at(-1).toFixed(2), left + w - 30, height - 20);
      ctx.fillText("x", left + w / 2, height - 4); ctx.fillText("u", 12, top + h / 2);
    }
  }
  function drawProcessing() {
    const d = sample(), p = d.processing, kind = prep.elements.smoothing.value;
    const view = $(".demo-processing-view").value, provider = prep.elements.derivatives.value;
    const canvas = $(".demo-processing-chart"), legend = $(".demo-processing-legend");
    canvas.hidden = false; legend.replaceChildren(); $(".demo-processing-hint").textContent = "";
    $(".demo-processing-caption").textContent = `Fixed ${d.id} illustration at t = ${p.time.toPrecision(4)}. Gaussian noise has std = ${100 * p.noise_level}% of the full field's std, seed ${p.noise_seed}. These curves were precomputed with KD; the recipe uses your selected dataset as loaded.`;
    const presets = $(".demo-processing-presets");
    const focused = presets.contains(document.activeElement) ? document.activeElement.dataset.value : null;
    presets.replaceChildren();
    const choices = kind === "savgol" ? p.windows : kind === "lowpass" ? p.mode_counts : [];
    choices.forEach((value) => {
      const b = document.createElement("button"); b.type = "button"; b.dataset.value = value;
      b.textContent = kind === "savgol" ? `Window ${value}` : `${value} modes`;
      const input = kind === "savgol" ? prep.elements.window : prep.elements.modes;
      b.setAttribute("aria-pressed", String(Number(input.value) === value));
      b.addEventListener("click", () => { input.value = value; update(); }); presets.append(b);
    });
    // Changing the processing family may remove the previously focused preset.
    if (focused !== null && choices.some((value) => String(value) === focused)) presets.querySelector(`[data-value="${focused}"]`).focus({ preventScroll: true });
    const key = kind === "savgol" ? `savgol:${prep.elements.window.value}` : kind === "lowpass" ? `lowpass:${prep.elements.modes.value}:${prep.elements.time_window.value}` : kind;
    let hint = "";
    if (view !== "spectrum") {
      if (kind === "network") hint = "Neural smoothing needs a trained field model. Configure its seed above and train it in the local app; this preview does not train a network.";
      else if (view === "target" && ["autograd", "none"].includes(provider)) hint = provider === "autograd" ? "Automatic derivatives require a trained network. Finite differences and FD2 have recorded previews here; use the local app for autograd." : "This mode does not use derivatives. Select a derivative-based mode to inspect a target preview.";
      else if (!(key in p.variants)) hint = "There is no prerecorded curve for these settings. Choose an example preset above, or keep your values and run them locally.";
    }
    if (hint) { canvas.hidden = true; $(".demo-processing-hint").textContent = hint; return; }
    const quantity = view === "target" ? provider : "field";
    let curves, xs;
    if (view === "spectrum") {
      xs = p.spectrum.map((_, i) => i + 1);
      curves = [{ label: "Energy retained in the noisy input", values: p.spectrum, color: "#0e716c" }];
      $(".demo-processing-hint").textContent = "The spectrum is measured before smoothing. Automatic cutoff selection needs a candidate equation and runs locally.";
    } else {
      xs = p.x;
      curves = [{ label: "Before added noise", values: p.clean[quantity], color: "#77786f" },
        { label: "Noisy input", values: p.variants.none[quantity], color: "#bb583e" }];
      if (kind !== "none") curves.push({ label: "After processing", values: p.variants[key][quantity], color: "#0e716c" });
    }
    curves.forEach((curve) => { const label = document.createElement("span"); label.textContent = curve.label; label.style.color = curve.color; legend.append(label); });
    const { ctx, width, height } = surface(canvas);
    const all = curves.flatMap((c) => c.values), lo = Math.min(...all), hi = Math.max(...all);
    const left = 54, top = 12, w = width - 76, h = height - 53;
    ctx.strokeStyle = "#e1e0d9"; ctx.strokeRect(left, top, w, h);
    curves.forEach((curve) => {
      ctx.beginPath(); ctx.strokeStyle = curve.color; ctx.lineWidth = curve.label === "Noisy input" ? 1 : 2;
      curve.values.forEach((value, i) => {
        const x = left + w * (xs[i] - xs[0]) / (xs.at(-1) - xs[0]);
        const y = top + h * (1 - (value - lo) / (hi - lo));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }); ctx.stroke();
    });
    const axisLabel = (v) => v !== 0 && (Math.abs(v) < .01 || Math.abs(v) >= 10000) ? v.toExponential(1) : v.toPrecision(3);
    ctx.fillStyle = "#52514e"; ctx.fillText(axisLabel(hi), 1, top + 8); ctx.fillText(axisLabel(lo), 1, top + h);
    ctx.fillText(xs[0].toPrecision(3), left, height - 20); ctx.fillText(xs.at(-1).toPrecision(3), left + w - 35, height - 20);
    ctx.fillText(view === "spectrum" ? "Retained spatial modes" : view === "target" ? `x; target u_t (${provider})` : "x; field u", left + w / 3, height - 3);
  }
  function drawResult() {
    const r = data.recorded;
    $(".demo-example-provenance").textContent = `Recorded with KD ${r.version}: ${r.dataset}, SGA, ${r.generations} generations, seed ${r.seed}. This saved example is independent of the settings you entered above.`;
    katex.render(r.equation_latex, $(".demo-equation"), { displayMode: true });
    $(".demo-metrics").replaceChildren();
    [[r.native_score.kind, r.native_score.value.toFixed(2)], ["NMSE", r.nmse.toExponential(3)], ["Forward relative L2", r.forward_error.toExponential(3)]].forEach(([name, value]) => {
      const box = document.createElement("div"); const label = document.createElement("span"); label.textContent = name;
      const number = document.createElement("strong"); number.textContent = value; box.append(label, number); $(".demo-metrics").append(box);
    });
    const { ctx, width } = surface($(".demo-coefficients")); const center = width * .55; const span = width * .32;
    ctx.strokeStyle = "#d8d7d0"; ctx.beginPath(); ctx.moveTo(center, 15); ctx.lineTo(center, 154); ctx.stroke();
    r.terms.forEach((term, i) => {
      const y = 35 + i * 65; const end = center + span * term.coefficient;
      ctx.fillStyle = "#0e716c"; ctx.fillRect(Math.min(center, end), y, Math.abs(end - center), 22);
      ctx.fillStyle = "#52514e"; ctx.fillText(term.latex.replaceAll("\\", "").replace(/[{}]/g, ""), 8, y + 16);
      ctx.fillText(term.coefficient.toFixed(4), Math.max(center, end) + 8, y + 16);
    }); ctx.fillText("Recorded coefficients", 8, 180);
  }
  defaults(); show("data");
  new ResizeObserver(() => { if (state.step === "data" && sample()) drawField(); if (state.step === "prepare" && sample()) drawProcessing(); if (state.step === "example") drawResult(); }).observe(root);
});
