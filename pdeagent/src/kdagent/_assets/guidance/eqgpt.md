Set `sparsity_alpha` explicitly. Evolution uses 0.02 in the supplied presets;
steady uses 1.0. The penalty multiplies reward by `1 - alpha*log10(k)`.
For steady data use `steady=true`, `steady_activation="sin"`,
`start_words=["S"]`, `sparsity_alpha=1.0`, without boundary deletion or polar
evaluation. Steady preparation retrains a 50000-epoch surrogate on every call,
including resume, on the run's device; there is no reuse channel.

Generation cost follows samples per epoch and per-sample evaluation cost.
A checkpoint includes transformer and optimizer state; its disk use grows
with retained checkpoints. Native spellings can be redundant:
`div(mul(mul(n2(x),u_x),u),n2(x))` expresses `u*u_x` where defined.
A zero reward can come from a sentence gate; inspect status and candidate
validity. Check allowed atoms and gates before increasing the sample budget.
`wave_multicase` needs the exact bundled primary data and is not selectable
from a benchmark input.
