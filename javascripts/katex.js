// Typeset the few equations the site shows. `document$` is Material's
// re-rendering hook, so this also runs after an instant-navigation page swap.
//
// The delimiters are scanned across the whole body rather than only the spans
// the markdown math extension produces: the homepage readout is a raw HTML
// block, which markdown does not look inside.
document$.subscribe(() => {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
  })
})
