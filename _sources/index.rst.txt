Knowledge Discovery (KD)
========================

**Symbolic PDE Discovery from Data**

KD is a modular Python toolkit for symbolic equation discovery, with a strong
focus on partial differential equations (PDEs). It ships four complementary
discovery engines, a unified dataset interface, and built-in
publication-quality visualisation.

.. code-block:: python

   from kd.dataset import load_pde
   from kd.model import KD_SGA

   dataset = load_pde("burgers")
   model = KD_SGA(sga_run=20, num=20, depth=4, width=5)
   model.fit_dataset(dataset)
   print(model.equation_latex())  # u_t = -1.001 u_x u + 0.100 u_xx

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   pde_backends
   viz
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
