"""Shared fixtures and data-availability handling for the curie public test suite.

Tests declare the databases they need via `requires_data`; a missing database produces an
explicit, reported skip. The CI job provides only a subset of the databases (see
.github/workflows/ci.yml) — everything needing another database skips visibly rather
than failing.
"""
import os
import pathlib

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1] / 'examples'


def _db_present(name):
    from curie.data import _db_available
    return _db_available(name)


def requires_data(*names):
    """Skip marker: test needs these curie databases (e.g. requires_data('decay', 'endf'))."""
    missing = [n for n in names if not _db_present(n)]
    return pytest.mark.skipif(
        bool(missing),
        reason='nuclear data not installed: {}'.format(', '.join(missing)))


def _has_pair_table():
    """True if the installed decay database holds the coincidence-pair table."""
    import sqlite3
    from curie import data
    for path in [data._data_path('decay.db')] + data._site_data_paths('decay.db') + [data._legacy_data_path('decay.db')]:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            con = sqlite3.connect(path)
            try:
                return con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                   "AND name='coincidences'").fetchone() is not None
            finally:
                con.close()
    return False


def requires_pairs():
    """Skip marker: test needs the coincidence-pair table of the decay database."""
    if not _db_present('decay'):
        reason = 'nuclear data not installed: decay'
    elif not _has_pair_table():
        reason = 'the installed decay database has no coincidences table'
    else:
        reason = None
    return pytest.mark.skipif(reason is not None, reason=reason or '')


@pytest.fixture(scope='session')
def eu_spectrum():
    """The shipped 152Eu calibration spectrum, isotopes assigned and peaks fit once."""
    import curie as ci
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU', '40K']
    sp.fit_peaks()
    return sp
