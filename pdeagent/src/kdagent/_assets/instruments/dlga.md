## Search space

DLGA-PDE searches sums of products drawn from a candidate `library`, fitting
constant coefficients. Reachable expressions are polynomials in the supplied
field and derivative tokens. Use the dataset's own field and axis names:
`u_x_t` is a mixed partial, and `diff_t(u_x)` is its open-form spelling.
An atomic name segment is limited to third order; higher compound derivatives
can be written as `u_xx_xx`.

The library determines which axes can enter a result. A library token that
names an unknown field or axis, or a dataset field other than the
left-hand-side one, refuses the whole fit before training. So does, when no
sketch is set, a token that spells a time derivative of the left-hand-side
field that the run fits: the order `target_lhs_order` names (`u_t`,
`diff_t(u)`, `diff1_t(u)` for order 1; `u_tt`, `diff2_t(u)` for order 2), or
both orders while `lhs_auto_select` is on. Such a token would fit its own
branch exactly. The time derivative of an order the run does not fit is an
ordinary library token. An atomic segment
above third order, such as `u_xxxx`, is not refused: the fit runs and every
candidate containing that token is invalid, so use the `u_xx_xx` spelling.
The surrogate models only the left-hand-side field; other fields and their
derivatives cannot enter its library. The constant-coefficient stage is
implemented; the adaptive variable-coefficient modes are rejected at
construction.

Surrogate fitting precedes genetic search and can dominate cost. Both the
number of observations and the derivative library affect resource use; the
epoch budget alone does not bound memory. A shorter fit changes derivative
accuracy and, under `lhs_auto_select`, can change the selected left-hand-side
order. No pretrained weights or optional backend are needed.

## Method

A neural network first fits the observations over their coordinates. The
architecture has five hidden layers of 50 units, `sin` activation and Adam
optimization. Field reads and derivatives both come from this network,
with derivatives obtained by autograd.

A genome contains up to `max_modules` modules of up to `max_module_length` tokens. Each module is a
product and the genome is their sum. Coefficients are found from the augmented
system's null space or by ordinary least squares, according to `solver`.
Fitness is `nmse + epsilon * length`, with length counted in tokens. Each
candidate is fitted against the time derivative of the order
`target_lhs_order` names, which has to equal the dataset's own left-hand-side
order. With `lhs_auto_select` enabled, each candidate is fitted against both
the first and second time derivative and the branch with the lower NMSE is
used, so the returned law can carry the order the dataset does not declare.

Each generation scores the population, retains the better half, refills it
with random genomes, exchanges a module between adjacent pairs, then mutates
by adding or deleting a module and shifting a token to an adjacent library
entry. Library order therefore affects mutation. One generation is one KD
iteration, controlled by `generations`.

## Result interpretation

`model.best_score_` combines the fit error with the per-token length penalty,
so NMSE alone does not rank term sets. NMSE is measured against the network's
field and derivatives, which are the quantities used during search.
`model.result_.equation` carries the fitted terms, coefficients and selected
left-hand side. The returned coefficients come from KD's refit over the
selected structure on that surrogate domain.

The recorder distinguishes surrogate training from genetic search. Read the
training and validation losses alongside the per-generation fitness,
diversity and complexity series. A small NMSE is evidence of a fit on that
domain, and does not by itself establish recovery of the generating equation.
If changing the seed or population budget improves the result, the search
budget was relevant; repeated poor fits also warrant checking the library
and surrogate accuracy.

## References

Xu et al. (2020). "DLGA-PDE: Discovery of PDEs with incomplete candidate library via combination of deep learning and genetic algorithm". *J. Comput. Phys.* 418, 109584. [Paper](https://doi.org/10.1016/j.jcp.2020.109584) · [arXiv:2001.07305](https://arxiv.org/abs/2001.07305)

Code: [woshixuhao/DLGA_tutorial](https://github.com/woshixuhao/DLGA_tutorial) (the author's tutorial notebook, on the KdV case), [woshixuhao/PIC_code](https://github.com/woshixuhao/PIC_code) (the author's own code for a later paper, running the same neural-network-plus-genetic-algorithm search)

<div style="font-size: 0.85em" markdown="1">

Implementation notes. The paper treats the order of the time derivative as
part of what is discovered: the left-hand side is a gene of the genome,
changed by mutation, and genomes with different left-hand sides compete on
fitness. By default KD takes the order from the dataset and fits only that
left-hand side. An earlier implementation fits `u_t` and `u_tt` for every
candidate and keeps the better branch, which is `lhs_auto_select=true` here;
under that option KD compares the branches by NMSE. The paper's
Eq. (6) and that implementation compare raw MSE, which also varies with
the target branch's variance. The length penalty is configurable; the author's
tutorial fixes it for its demonstration. The genetic mechanism is checked
against that implementation, with corrected crossover indexing and selection
of candidates with duplicate fitness values. The adaptive variable-coefficient
extension is not part of this implementation.

</div>
