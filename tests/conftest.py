"""Shared fixtures and data-availability handling for the curie public test suite.

Tests declare the databases they need via `requires_data`; a missing database produces an
explicit, reported skip. Until the Stage-3 distribution redesign makes data fetching
available in CI, the minimal CI job ships only ziegler.db (184 kB) and runs the subset that
needs nothing else — everything data-dependent skips visibly rather than failing.
"""
import os
import pathlib

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1] / 'examples'


def _db_present(name):
    from curie.data import _data_path
    path = _data_path(name + '.db')
    return os.path.isfile(path) and os.path.getsize(path) > 0


def requires_data(*names):
    """Skip marker: test needs these curie databases (e.g. requires_data('decay', 'endf'))."""
    missing = [n for n in names if not _db_present(n)]
    return pytest.mark.skipif(
        bool(missing),
        reason='nuclear data not installed: {}'.format(', '.join(missing)))


@pytest.fixture(scope='session')
def eu_spectrum():
    """The shipped 152Eu calibration spectrum, isotopes assigned and peaks fit once."""
    import curie as ci
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU', '40K']
    sp.fit_peaks()
    return sp
