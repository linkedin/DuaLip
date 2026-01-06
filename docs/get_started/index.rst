Installation Guide
===================

The DuaLip project is distributed as a Python package. We recommend Python 3.9 or newer and using a virtual environment.

Create and activate a virtual environment (recommended):

.. code:: bash

  python3 -m venv .venv
  source .venv/bin/activate

Development install (editable with tooling):

.. code:: bash

  make install  # installs -e .[dev] and sets up pre-commit hooks

Run tests:

.. code:: bash

  make test  # or: pytest

Code style and lint:

.. code:: bash

  make checkstyle  # runs black, isort, flake8 via pre-commit

Basic usage:

.. code:: python

  import dualip

For detailed usage please see the :ref:`solver` and the :ref:`demo`.
