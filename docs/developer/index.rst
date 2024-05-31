Developer Documentation
==============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

General Considerations
----------------------

Try to adhere to `PEP 8 <https://peps.python.org/pep-0008/>`_.
For docstrings, we use the
`numpy style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Design
------

The overall layout of the libary is shown below.  There are classes of material
models, ``elasticity`` and ``slip``. The elasticity is a simple package that
delivers aniostropic stiffness matrices. The slip modeling is more involved and
provides a basic framework for slip while allowing for multiple models of slip
state variables and their evolution.  The ``microstructure`` package handles the
spatial assignment of phase, grain and orientation.  Tools are available for
handling both measured data and synthetic data. The ``utils`` package provides
support for the modeling packages.

The interfaces have been designed to evaluate properties at a large number
of points efficiently, for compatibility with the companion pacakge,
**polycrystalx**.

.. image:: ../_static/polycrystal-design.pdf
