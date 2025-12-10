PDE Backends
============

KD provides several PDE-focused symbolic discovery backends that share a
unified data entry point via :class:`kd.dataset.PDEDataset`.

SGA
---

``KD_SGA`` implements sparse-regression–based PDE discovery on a hand-crafted
library of candidate terms. It is designed for local PDEs of the form
``u_t = N(u, u_x, ...)``.

Typical usage:

.. code-block:: python

   dataset = load_pde("burgers")
   model = KD_SGA()
   model.fit_dataset(dataset)

DLGA
----

``KD_DLGA`` is a deep-learning–aided genetic algorithm for PDE discovery. It
supports the same ``PDEDataset + fit_dataset`` interface and can also be used
with expert-level ``fit(X, y)`` workflows on compatible data.

DSCV (regular and SPR)
----------------------

``KD_DSCV`` and the sparse/PINN variant inside ``kd.model.kd_dscv`` expose
wrappers around the DISCOVER framework. In KD 1.0, these backends are intended
for local PDE benchmarks and simple PINN-style experiments, and are driven via
``import_dataset`` / ``fit_dataset``.

PySR (optional)
---------------

``KD_PySR`` is a thin wrapper around the PySR project. It is intended as a
generic symbolic regression backend for tasks of the form ``fit(X, y)`` and is
enabled only when the optional ``pysr`` extra is installed.

