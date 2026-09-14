### DLGA observations

The packaged validation includes structure recovery at 15% relative noise.
A 50000-epoch fit on a 256x101 grid took 1409 s on a laptop GPU. A CPU fit on
501x501 had not returned after 2 hours 45 minutes. On that large grid, 2000
epochs and a four-token library including u_x_t reached 13.4 GB after 23
minutes and were killed by the host memory guard. A 1000-epoch run with three
tokens and no mixed partial returned in 693 s. Those two changes were not
separated experimentally. Probe a large grid with a small library and reduced
training budget before increasing either.

At 3 and 10 training epochs, first-order data probes selected the second time
derivative; at 100 epochs they selected the first. Structural hits in the
packaged suite spanned NMSE 4e-4 to 3.1e-2, while wrong structures on the
hardest case spanned 2.7e-2 to 5.1e-2. The old operational heuristic was to
try epsilon between 1e-2 and 5e-2 for expressions over five tokens at NMSE
above 1e-2; treat it as a trial, not an acceptance criterion. One segment spent
820.9 s between fit_started and search_started, then 28 s searching.

The branch-selection NMSE/MSE A/B held surrogates, seeds, budget and epsilon
fixed and moved recovery from 17/25 to 21/25 seeds. Validated epsilon values
spanned 1e-6 to 1e-3, against the author's tutorial value of 1e-1. The earlier
reference had crossover indexed by generation count instead of population
size, and selection reused the first index for duplicate fitness values; KD
corrects both. A `no_candidate` partial result has empty display, null law and
null NMSE/MSE. Check the library and surrogate as well as population size,
generations and seed; invariance across those probes alone is not proof that
the target is unrepresentable.
