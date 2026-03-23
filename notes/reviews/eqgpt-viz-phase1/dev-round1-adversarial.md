# Adversarial Review - Dev Round 1
- REQUEST_CHANGES, High: 2, Medium: 3, Low: 2
- H1: _equation may return paths to non-existent file (render silently fails)
- H2: _plot_ranking leaks figure on savefig exception (no try/finally)
- M3: _reward_ranking takes first-N not top-N (assumes sorted input)
- M4: S/E marker passthrough if embedded in token
- M5: No spaces around + in LaTeX
