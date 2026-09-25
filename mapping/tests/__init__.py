"""M0 test suite.

Run from the ``mapping/`` directory with no installation step:

    python -m unittest discover -s tests -t .

The ``-t .`` matters: it makes the repository root the top-level directory, so
``tests`` is imported as a package and the suite's relative imports resolve.
The core is pure standard library, so there is nothing to install first.
"""
