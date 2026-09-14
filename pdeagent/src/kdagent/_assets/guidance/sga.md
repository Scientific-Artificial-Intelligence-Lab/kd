### SGA observations

On a 64x32 grid at noise_level=0.001, twelve direct fits over six seeds and two
budgets gave 4 exact, 4 near-miss and 4 wrong structures; at 0.002 a further
protocol gave 0 recoveries in 18 fits. Thirty generations on 2k-8k point grids
took 2.5-4.3 s; ten generations on a 256x201 grid took 13.5 s. Grid size made
little difference from 2048 to 8192 points in those probes; larger grids did
cost more. Time also depends on generations and population.

The recorded autograd recovery tier used 15000 training epochs against the
then-default 1000. One seed cost 53-104 s on one 1D case and 685-1002 s on
another. Training preceded the search iterations. Warnings about dropped
derivative terminals appeared in logs, not the returned envelope.

The recovery suites used NMSE bounds 0.05, relaxed to 0.10 in three dimensions;
one clean 256x201 run reached 1.1e-5 in ten generations. On one noisy grid,
collapsed fits scored 0.076-0.083 and non-collapsed fits 0.041-0.048, with
near-misses at least as good as exact structures. SGA's native NMSE has also
fallen below a neutral refit's floor on the same terms. Do not rank plausible
structures by that number alone. Both recorded successful runs became flat
within ten generations; another seed can be useful instead of extending such
a run. The recovery gates accepted 2-4 hits in 5 seeds, and 1 in 3 in three
dimensions; these are test criteria, not a predicted success probability.

A field-only law means the default column survived. An invalid result can
report `non_finite` or `structural_reject`. The display rounds to four
significant figures and drops coefficients smaller than 1e-10; retain the
structured coefficients. Equivalent derivatives and identity wrapping require
mathematical normalization, including the chain rule, x_x=1, higher coordinate
derivatives zero and u_x_x=u_xx. KD's SymPy conversion supplies that semantics;
a spelling difference is not a different physical law.
