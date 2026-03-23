# Logic Review - Round 1
- REQUEST_CHANGES, High: 2, Medium: 3, Low: 2
- H1: n_bars metadata assertions guarded by `if`, vacuously pass
- H2: latex metadata assertion guarded by `if`, vacuously pass
- M1: test_dynamic_yaxis assertion equivalent to smoke test
- M2: S/E marker assertions too loose (allow non-empty returns)
- M3: pad token assertion too loose
