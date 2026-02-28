PDE Backends
============

KD provides several PDE-focused symbolic discovery backends. All share a
unified data entry point via :class:`kd.dataset.PDEDataset` and the
``fit_dataset()`` interface.

KD_SGA — Symbolic Genetic Algorithm
------------------------------------

Evolves expression trees and uses STRidge (Sequential Thresholded Ridge
Regression) to fit sparse coefficients. Fast, interpretable, with built-in
LaTeX output.

.. code-block:: python

   from kd.dataset import load_pde
   from kd.model import KD_SGA

   dataset = load_pde("burgers")
   model = KD_SGA(
       sga_run=20,      # generations
       num=20,           # population size
       depth=4,          # max tree depth
       width=5,          # max tree width (terms)
       p_var=0.5,        # variation probability
       p_mute=0.3,       # mutation probability
       p_cro=0.5,        # crossover probability
       seed=0,
   )
   model.fit_dataset(dataset)

   print(model.equation_latex())   # LaTeX string
   print(model.best_pde_)          # raw PDE string
   print(model.best_score_)        # AIC score

KD_DLGA — Deep-Learning Genetic Algorithm
------------------------------------------

Trains a neural surrogate to compute derivatives, then applies genetic
algorithm search over candidate PDE terms. Supports rich optimisation
diagnostics.

.. code-block:: python

   from kd.dataset import load_pde
   from kd.model import KD_DLGA

   dataset = load_pde("kdv")
   model = KD_DLGA(
       operators=["u", "u_x", "u_xx", "u_xxx"],
       epi=1e-3,
       input_dim=2,
   )
   model.fit_dataset(dataset)

KD_DSCV — DISCOVER (RL + STRidge)
----------------------------------

Uses a reinforcement-learning controller to search over expression trees,
combined with STRidge for coefficient fitting. Supports flexible operator
sets.

.. code-block:: python

   from kd.dataset import load_pde
   from kd.model import KD_DSCV

   dataset = load_pde("burgers")
   model = KD_DSCV(
       binary_operators=["add", "mul", "div", "diff", "diff2", "diff3"],
       unary_operators=["n2", "n3"],
       n_samples_per_batch=500,
       n_iterations=100,
       seed=0,
   )
   model.fit_dataset(dataset, n_epochs=100)

KD_DSCV_SPR — DISCOVER + PINN
-------------------------------

Extends KD_DSCV with a Physics-Informed Neural Network (PINN) for handling
very sparse or noisy observations.

.. code-block:: python

   from kd.dataset import load_pde
   from kd.model import KD_DSCV_SPR

   dataset = load_pde("burgers")
   model = KD_DSCV_SPR(
       binary_operators=["add", "mul", "diff", "diff2"],
       unary_operators=["n2"],
       n_iterations=50,
       seed=0,
   )
   model.fit_dataset(dataset, n_epochs=20, sample_ratio=0.1)

KD_PySR — PySR Wrapper (optional)
-----------------------------------

A thin wrapper around the `PySR <https://github.com/MilesCranmer/PySR>`_
project for general symbolic regression of the form ``fit(X, y)``. Requires
the optional ``pysr`` extra.

.. code-block:: python

   from kd.model import KD_PySR

   model = KD_PySR(niterations=1000)
   model.fit(X, y)
   print(model.equation_latex())

.. note::

   ``KD_PySR`` uses the generic ``fit(X, y)`` interface, not ``fit_dataset``.
