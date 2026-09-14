### PySR tool configuration

An omitted generations budget is short for PySR: it controls internal search
iterations. `resume_from_run_id` restores populations and the hall of fame;
the new generation budget is additional. The tool's seed maps to random_state.
The optional PySR backend must be installed in the worker environment before
running; a missing backend otherwise fails after the term-matrix build.
The supported package range is pysr>=1.5,<2.

### PySR observations

One recorded refit comparison gave NMSE 2.9e-5 for a correct structure and
8.5e-1 for a wrong structure. Near-one NMSE indicates little improvement over
the mean on that domain. `kd.evaluate_terms` over the whole supplied library
tests its linear span, not all nonlinear combinations PySR can form. A large
linear-span NMSE therefore cannot prove that PySR's search space is inadequate.
The envelope gives the selected point; inspect recorder.json for the full
front. PDE display is coefficient-free, for example
add(u_x, mul(u, u_x)); read primary.law for coefficients. In tabular Python
runs, numeric constants stay inside the expression and the recorded outer
scale is separate. Tabular input is not offered by this controller.
