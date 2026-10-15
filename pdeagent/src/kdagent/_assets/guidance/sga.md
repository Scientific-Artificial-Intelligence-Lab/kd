Use `population`, the facade spelling; `num` is not a JSON parameter.
Each field and its first derivative along each non-target axis can be a leaf,
while only the left-hand-side field is the default regression column.
AIC ranks candidates within the same dataset. A low native NMSE does not
establish recovery, and derivative spellings are not merged; use atomic
terminals where they exist.

Generation cost grows with grid points, spatial axes and population. Each
generation scores at most `2*int(population*p_cro) + population - 1`
candidates (crossover children plus mutated members; duplicates are not
scored), about 1.5 to 1.9 times the population at the defaults. Autograd
trains before the first generation; select it with `derivatives="autograd"`
and inject a trained surrogate to avoid fresh training.
`restart_interval` redraws the population every k generations with seed +
1000*j. A fork's reseed stride can overlap that restart seed schedule;
choose fork seeds with that overlap in mind.
