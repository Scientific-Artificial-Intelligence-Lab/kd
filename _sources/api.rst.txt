API Documentation
=================

Core Modules
------------

Base
~~~~

.. automodule:: kd.base
   :members:
   :undoc-members:
   :show-inheritance:

Datasets
~~~~~~~~

.. automodule:: kd.dataset
   :members:
   :undoc-members:
   :show-inheritance:

Models
------

High-level wrappers
~~~~~~~~~~~~~~~~~~~

.. automodule:: kd.model.kd_sga
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: kd.model.kd_dlga
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: kd.model.kd_dscv
   :members:
   :undoc-members:
   :show-inheritance:

KD_PySR (optional)
^^^^^^^^^^^^^^^^^^

``KD_PySR`` wraps `PySR <https://github.com/MilesCranmer/PySR>`_ for generic
symbolic regression via ``fit(X, y)``. Requires the ``pysr`` extra.

.. note::

   API docs for ``kd.model.kd_pysr`` are excluded from the autodoc build
   because PySR's Julia runtime conflicts with Sphinx on some platforms.

Visualization
-------------

Unified facade
~~~~~~~~~~~~~~

.. automodule:: kd.viz.api
   :members:
   :undoc-members:
   :show-inheritance:

Equation rendering
~~~~~~~~~~~~~~~~~~

.. automodule:: kd.viz.equation_renderer
   :members:
   :undoc-members:
   :show-inheritance:

Utils
-----

.. automodule:: kd.utils
   :members:
   :undoc-members:
   :show-inheritance:
