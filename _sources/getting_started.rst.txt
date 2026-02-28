Getting Started
===============

This section covers installation and your first equation-discovery experiment.

Installation
------------

Requires **Python >= 3.9** and PyTorch.

.. code-block:: bash

   pip install -e .

PySR-based symbolic regression is optional:

.. code-block:: bash

   pip install "kd[pysr]"

First example — Discover a PDE
-------------------------------

Discover the Burgers equation from data in 3 lines:

.. code-block:: python

   from kd.dataset import load_pde
   from kd.model import KD_SGA

   dataset = load_pde("burgers")
   model = KD_SGA(sga_run=20, num=20, depth=4, width=5)
   model.fit_dataset(dataset)
   print(model.equation_latex())  # u_t = -1.001 u_x u + 0.100 u_xx

Visualise results
-----------------

.. code-block:: python

   from kd.viz import configure, render_equation, plot_field_comparison, plot_parity

   configure(save_dir="artifacts/my_run")
   render_equation(model)
   plot_field_comparison(model, x_coords=None, t_coords=None,
                         true_field=None, predicted_field=None)
   plot_parity(model)

Built-in datasets
-----------------

.. code-block:: python

   from kd.dataset import load_pde, list_available_datasets

   print(list_available_datasets())
   # ['chafee-infante', 'burgers', 'kdv', 'fisher', ...]

   dataset = load_pde("kdv")  # returns a PDEDataset

You can also construct a ``PDEDataset`` from your own NumPy arrays:

.. code-block:: python

   from kd.dataset import PDEDataset

   dataset = PDEDataset(
       equation_name="my_pde",
       fields_data={"u": u_array},
       coords_1d={"x": x, "t": t},
       axis_order=["x", "t"],
       target_field="u",
       lhs_axis="t",
   )

See ``examples/`` for 14+ runnable scripts covering every model and
visualisation mode.
