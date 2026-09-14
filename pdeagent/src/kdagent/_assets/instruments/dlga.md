## Search space

DLGA-PDE searches sums of products drawn from a candidate `library`, fitting
constant coefficients. Reachable expressions are polynomials in the supplied
field and derivative tokens. Use the dataset's own field and axis names:
`u_x_t` is a mixed partial, and `diff_t(u_x)` is its open-form spelling.
An atomic name segment is limited to third order; higher compound derivatives
can be written as `u_xx_xx`.

The library determines which axes can enter a result. Tokens that cannot be
resolved cause the whole candidate containing them to be discarded; other
candidates can still produce a fit. The surrogate models the left-hand-side
field, so derivatives of another field are not available. Check a proposed
library with `kd.validate_terms(dataset, terms)` before training. The
constant-coefficient stage is implemented; the adaptive variable-coefficient
modes are rejected at construction.

Surrogate fitting precedes genetic search and can dominate cost. Both the
number of observations and the derivative library affect resource use; the
epoch budget alone does not bound memory. A shorter fit changes derivative
accuracy and can affect the selected left-hand-side order. No pretrained
weights or optional backend are needed.

## Method

A neural network first fits the observations over their coordinates. The
architecture has five hidden layers of 50 units, sine activation and Adam
optimization. Field reads and derivatives both come from this network,
with derivatives obtained by autograd.

A genome contains up to five modules of up to five tokens. Each module is a
product and the genome is their sum. Coefficients are found from the augmented
system's null space or by ordinary least squares, according to `solver`.
Fitness is `nmse + epsilon * length`, with length counted in tokens. With
`lhs_auto_select` enabled, each candidate is fitted against both the first and
second time derivative, and the branch with the lower NMSE is used.
`target_lhs_order` controls which dataset orders the run accepts; it does not
pin the fitted branch. Second-order input requires auto-selection, while
disabling auto-selection fits only the first derivative.

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

Implementation notes. KD compares the candidate left-hand-side branches by NMSE. The paper's
Eq. (6) and an earlier implementation compare raw MSE, which also varies with
the target branch's variance. The length penalty is configurable; the author's
tutorial fixes it for its demonstration. The genetic mechanism is checked
against that implementation, with corrected crossover indexing and selection
of candidates with duplicate fitness values. The adaptive variable-coefficient
extension is not part of this implementation.

</div>
