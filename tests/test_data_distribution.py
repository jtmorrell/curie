"""Distribution-layer tests for curie.data: cache directory, registry aliases,
legacy-data migration, checksummed fetch, and per-target shard assembly.

All tests run offline: network fetches are monkeypatched to copy from local
fixture files built on the fly. The cache directory is redirected through
CURIE_DATA_DIR so the user's real cache is never touched.
"""
import hashlib
import os
import sqlite3

import pytest

from curie import data


def _make_db(path, table='t', rows=((1, 2.0), (2, 3.0))):
	con = sqlite3.connect(path)
	con.execute('CREATE TABLE "{}" (a INTEGER, b REAL)'.format(table))
	con.executemany('INSERT INTO "{}" VALUES (?,?)'.format(table), rows)
	con.commit()
	con.close()
	with open(path, 'rb') as f:
		return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
	"""Isolated cache dir, legacy dir, registry, and connection cache."""
	cache = tmp_path / 'cache'
	legacy = tmp_path / 'legacy'
	cache.mkdir()
	legacy.mkdir()
	monkeypatch.setenv('CURIE_DATA_DIR', str(cache))
	monkeypatch.setattr(data, '_legacy_data_path', lambda db='': str(legacy / db))
	registry = {'base_url': 'test://', 'files': {}, 'shards': {}}
	monkeypatch.setattr(data, '_registry', lambda: registry)
	monkeypatch.setattr(data, 'GLOB_CONNECTIONS_DICT', {})
	fetched = []

	def fake_retrieve(fnm):
		fetched.append(fnm)
		src = tmp_path / 'remote' / fnm
		if not src.exists():
			raise IOError('{} not available from the test remote'.format(fnm))
		dest = cache / fnm
		dest.write_bytes(src.read_bytes())
		return str(dest)

	monkeypatch.setattr(data, '_retrieve', fake_retrieve)
	(tmp_path / 'remote').mkdir()
	return {'cache': cache, 'legacy': legacy, 'remote': tmp_path / 'remote',
			'registry': registry, 'fetched': fetched}


def test_data_path_honors_env_override(sandbox):
	assert data._data_path('x.db') == str(sandbox['cache'] / 'x.db')


def test_canonical_aliases():
	for alias in ['decay', 'DECAY', 'decay.db']:
		assert data._canonical(alias) == 'decay.db'
	for alias in ['tendl_n_rp', 'tendl_nrp', 'tendl_n', 'nrp', 'rpn']:
		assert data._canonical(alias) == 'tendl_n_rp.db'
	for alias in ['iaea', 'cpr', 'iaea_medical', 'iaea-monitor', 'medical']:
		assert data._canonical(alias) == 'iaea_monitors.db'
	assert data._canonical('irdff') == 'IRDFF.db'
	assert data._canonical('/some/user/path.db') is None
	assert data._canonical('my_peaks.db') is None


def test_user_path_branch_untouched(sandbox, tmp_path):
	"""Arbitrary user paths (saveas/load) must never consult registry or network."""
	p = tmp_path / 'user_peaks.db'
	con = data._get_connection(str(p))
	con.execute('CREATE TABLE peaks (x)')
	con.commit()
	assert p.exists()
	assert sandbox['fetched'] == []


def test_managed_name_fetches_when_missing(sandbox):
	sha = _make_db(sandbox['remote'] / 'decay.db', 'chart')
	sandbox['registry']['files']['decay.db'] = sha
	con = data._get_connection('decay')
	assert con.execute('SELECT COUNT(*) FROM chart').fetchone()[0] == 2
	assert sandbox['fetched'] == ['decay.db']


def test_legacy_file_adopted_without_fetch(sandbox):
	sha = _make_db(sandbox['legacy'] / 'decay.db', 'chart')
	sandbox['registry']['files']['decay.db'] = sha
	con = data._get_connection('decay')
	assert con.execute('SELECT COUNT(*) FROM chart').fetchone()[0] == 2
	assert sandbox['fetched'] == []
	assert (sandbox['cache'] / 'decay.db').exists()


def test_corrupt_legacy_file_refetched(sandbox):
	_make_db(sandbox['legacy'] / 'decay.db', 'chart', rows=((9, 9.0),))
	sha = _make_db(sandbox['remote'] / 'decay.db', 'chart')
	sandbox['registry']['files']['decay.db'] = sha
	con = data._get_connection('decay')
	assert con.execute('SELECT COUNT(*) FROM chart').fetchone()[0] == 2
	assert sandbox['fetched'] == ['decay.db']


def test_ensure_table_assembles_shard(sandbox):
	sha = _make_db(sandbox['remote'] / 'endf_XX_1.db', 'XX_1')
	sandbox['registry']['shards']['endf'] = {'endf_XX_1.db': sha}
	con = data._get_connection('endf')  # starts empty: no whole file anywhere
	data._ensure_table('endf', 'XX_1')
	assert con.execute('SELECT COUNT(*) FROM XX_1').fetchone()[0] == 2
	assert sandbox['fetched'] == ['endf_XX_1.db']
	assert not (sandbox['cache'] / 'endf_XX_1.db').exists()  # shard consumed
	data._ensure_table('endf', 'XX_1')  # second call: already present
	assert sandbox['fetched'] == ['endf_XX_1.db']


def test_ensure_table_noop_for_whole_file_dbs(sandbox):
	sha = _make_db(sandbox['remote'] / 'tendl.db', 'RB_85')
	sandbox['registry']['files']['tendl.db'] = sha
	data._get_connection('tendl')
	data._ensure_table('tendl', 'RB_85')
	assert sandbox['fetched'] == ['tendl.db']  # file fetch only, no shard logic


def test_ensure_table_unknown_table_left_to_sql(sandbox):
	"""A table with no shard must not raise here: the caller's SQL reports it."""
	sandbox['registry']['shards']['endf'] = {}
	data._get_connection('endf')
	data._ensure_table('endf', 'NOT_A_TARGET')


def test_download_prefetches_whole_file(sandbox, capsys):
	sha = _make_db(sandbox['remote'] / 'ziegler.db', 'compounds')
	sandbox['registry']['files']['ziegler.db'] = sha
	data.download('ziegler')
	assert (sandbox['cache'] / 'ziegler.db').exists()
	data.download('ziegler')  # second call skips
	assert sandbox['fetched'] == ['ziegler.db']
	assert 'already installed' in capsys.readouterr().out
