Getting Started
===============

This section provides a minimal overview of how to install Knowledge Discovery
(KD) and run your first equation-discovery experiment, typically on PDE-focused
benchmarks.

Installation
------------

The recommended way to work with KD from the repository is:

.. code-block:: bash

   conda create -n kd-env python=3.9
   conda activate kd-env
   pip install -r requirements.txt
   pip install .

PySR-based symbolic regression is optional. To enable it, install the extra
dependency:

.. code-block:: bash

   pip install "kd[pysr]"

First example (SGA)
-------------------

A minimal SGA workflow looks like:

.. code-block:: python

   dataset = load_pde("burgers")
   model = KD_SGA()
   model.fit_dataset(dataset)
   print(model.best_equation_latex_)

Imports are omitted for brevity; see ``examples/kd_sga_example.py`` for a
complete script.
