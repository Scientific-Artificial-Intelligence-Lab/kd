// Equation typesetting with the vendored KaTeX (vendor/katex, loaded by
// index.html as window.katex). The LaTeX always comes from kd through the API;
// this module never builds LaTeX itself.

const katex = window.katex;

/**
 * Typeset `latex` into element `el` (its content is replaced).
 * opts.display: block (display) style; default false (inline style).
 * A string KaTeX cannot parse is shown as plain text with class tex-fallback.
 */
export function renderTex(el, latex, { display = false } = {}) {
  try {
    katex.render(latex, el, { displayMode: display, throwOnError: true, output: "html" });
    el.classList.remove("tex-fallback");
  } catch (err) {
    el.textContent = latex;
    el.classList.add("tex-fallback");
    el.title = err.message;
  }
  return el;
}

/** HTML string of `latex` (for tooltips and table cells built as strings). */
export function texToHtml(latex, { display = false } = {}) {
  return katex.renderToString(latex, { displayMode: display, throwOnError: false, output: "html" });
}

/** Typeset every element under `root` that has a data-tex attribute. */
export function typeset(root = document) {
  for (const el of root.querySelectorAll("[data-tex]")) {
    renderTex(el, el.dataset.tex, { display: el.dataset.display === "true" });
  }
}
