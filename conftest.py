"""Root conftest: inject the standard import aliases into every doctest namespace.

The docstring examples throughout curie follow the ``import curie as ci`` convention
(with ``np`` for numpy and ``plt`` for pyplot) without repeating the imports in each
example. This fixture makes those names available when the examples run as doctests.
"""
import pytest


@pytest.fixture(autouse=True)
def _curie_doctest_namespace(doctest_namespace):
    import curie as ci
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    doctest_namespace['ci'] = ci
    doctest_namespace['np'] = np
    doctest_namespace['plt'] = plt
