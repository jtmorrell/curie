"""Tests of the coincidence-summing correction as calibrate() and Spectrum apply it.

The data-layer tests isolate through CURIE_DATA_DIR, a monkeypatched registry and
fetch seam (`data._retrieve`) and small fixture databases built on the fly; they
never touch the network or the real data directory.
"""
import hashlib
import sqlite3

import pytest

from curie import data


GENERATION = 'v7'


def _decay_db(path, generation=None, coincidences=True):
    """A decay database with an optional generation stamp and pair table."""
    con = sqlite3.connect(str(path))
    con.execute('CREATE TABLE chart (isotope TEXT, name TEXT)')
    if generation is not None:
        con.execute('CREATE TABLE _version (library TEXT, generation TEXT)')
        con.execute("INSERT INTO _version VALUES ('decay', ?)", (generation,))
    if coincidences:
        con.execute('CREATE TABLE coincidences (isotope TEXT, isomer TEXT, energy REAL, coinc_energy REAL, '
                    'coinc_type TEXT, intensity REAL, unc_intensity REAL)')
        con.execute("INSERT INTO coincidences VALUES ('60CO', 'g', 1173.228, 1332.492, 'g', 99.8, NULL)")
    con.commit()
    con.close()
    with open(str(path), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture
def datadir(tmp_path, monkeypatch):
    """An isolated data directory, registry of generation v7, fetch seam and connection cache."""
    cache = tmp_path / 'cache'
    cache.mkdir()
    (tmp_path / 'remote').mkdir()
    monkeypatch.setenv('CURIE_DATA_DIR', str(cache))
    monkeypatch.setattr(data, '_site_data_paths', lambda db='': [str(tmp_path / 'site' / db)])
    monkeypatch.setattr(data, '_legacy_data_path', lambda db='': str(tmp_path / 'legacy' / db))
    registry = {'base_url': 'https://example.invalid/releases/download/{}/'.format(GENERATION), 'files': {}, 'shards': {}}
    monkeypatch.setattr(data, '_registry', lambda: registry)
    monkeypatch.setattr(data, 'GLOB_CONNECTIONS_DICT', {})

    def fake_retrieve(fnm):
        dest = cache / fnm
        dest.write_bytes((tmp_path / 'remote' / fnm).read_bytes())
        return str(dest)

    monkeypatch.setattr(data, '_retrieve', fake_retrieve)
    return {'tmp': tmp_path, 'cache': cache, 'remote': tmp_path / 'remote', 'registry': registry}


@pytest.mark.parametrize('generation, coincidences, usable, warning', [
    (None, True, False, 'predates the generation stamp'),
    ('v6', True, False, 'is from data generation v6'),
    (GENERATION, True, True, None),
    (GENERATION, False, False, None),
], ids=['no-stamp', 'other-generation', 'expected-generation', 'expected-generation-no-pair-table'])
def test_pair_table_usable(datadir, capsys, generation, coincidences, usable, warning):
    _decay_db(datadir['cache'] / 'decay.db', generation, coincidences)
    assert data._pair_table_usable() is usable
    out = capsys.readouterr().out
    if warning is None:
        assert 'generation' not in out
    else:
        assert warning in out and "ci.download('decay')" in out
    # asked again: the same answer from the same connection, and no second warning
    assert data._pair_table_usable() is usable
    assert 'generation' not in capsys.readouterr().out


def test_pair_table_usable_follows_the_file(datadir, monkeypatch):
    # a stale file and a current one, each in its own data directory
    other = datadir['tmp'] / 'other'
    other.mkdir()
    _decay_db(datadir['cache'] / 'decay.db', 'v6')
    _decay_db(other / 'decay.db', GENERATION)
    assert data._pair_table_usable() is False
    monkeypatch.setenv('CURIE_DATA_DIR', str(other))
    assert data._pair_table_usable() is True
    monkeypatch.setenv('CURIE_DATA_DIR', str(datadir['cache']))
    assert data._pair_table_usable() is False

    # the stale file replaced by a re-download in the same session
    datadir['registry']['files']['decay.db'] = _decay_db(datadir['remote'] / 'decay.db', GENERATION)
    data.download('decay')
    assert data._pair_table_usable() is True
