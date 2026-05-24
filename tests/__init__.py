"""Importable test package for standard-library unittest discovery.

Importing the package applies source-tree path setup once, so targeted commands
such as ``python -m unittest tests.test_operator_surface`` work without a prior
editable install.
"""

from tests import _pathfix as _pathfix  # noqa: F401
