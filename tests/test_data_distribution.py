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
	"""Isolated data dir, site dir, legacy dir, registry, and connection cache."""
	cache = tmp_path / 'cache'
	legacy = tmp_path / 'legacy'
	site = tmp_path / 'site'
	cache.mkdir()
	legacy.mkdir()
	site.mkdir()
	monkeypatch.setenv('CURIE_DATA_DIR', str(cache))
	monkeypatch.setattr(data, '_legacy_data_path', lambda db='': str(legacy / db))
	monkeypatch.setattr(data, '_site_data_paths', lambda db='': [str(site / db)])
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
	return {'cache': cache, 'legacy': legacy, 'site': site,
			'remote': tmp_path / 'remote', 'registry': registry, 'fetched': fetched}


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


def test_zero_byte_sharded_db_recovers_across_restart(sandbox):
	"""A crash right after the first connection leaves a zero-byte endf.db;
	the next session must treat it as a fresh assembled database, not an error."""
	(sandbox['cache'] / 'endf.db').write_bytes(b'')
	sha = _make_db(sandbox['remote'] / 'endf_XX_1.db', 'XX_1')
	sandbox['registry']['shards']['endf'] = {'endf_XX_1.db': sha}
	con = data._get_connection('endf')
	data._ensure_table('endf', 'XX_1')
	assert con.execute('SELECT COUNT(*) FROM XX_1').fetchone()[0] == 2


def test_failed_assembly_leaves_no_partial_table(sandbox):
	"""An assembly that dies mid-way must leave no committed empty table
	(which the existence check would forever treat as complete)."""
	sha = _make_db(sandbox['remote'] / 'endf_XX_1.db', 'WRONG_NAME')
	sandbox['registry']['shards']['endf'] = {'endf_XX_1.db': sha}
	con = data._get_connection('endf')
	with pytest.raises(Exception):
		data._ensure_table('endf', 'XX_1')
	assert not con.execute("SELECT name FROM sqlite_master WHERE name='XX_1'").fetchone()
	os.remove(str(sandbox['remote'] / 'endf_XX_1.db'))
	sha = _make_db(sandbox['remote'] / 'endf_XX_1.db', 'XX_1')
	data._ensure_table('endf', 'XX_1')  # a corrected shard then assembles fine
	assert con.execute('SELECT COUNT(*) FROM XX_1').fetchone()[0] == 2


def test_sharded_site_copy_never_assembled_into(sandbox):
	"""A site-provided endf.db is used as-is: a missing table must not
	trigger a shard fetch or any write to the site file."""
	_make_db(sandbox['site'] / 'endf.db', 'YY_1')
	sandbox['registry']['shards']['endf'] = {'endf_XX_1.db': '0' * 64}
	con = data._get_connection('endf')
	before = (sandbox['site'] / 'endf.db').read_bytes()
	data._ensure_table('endf', 'XX_1')
	assert sandbox['fetched'] == []
	assert (sandbox['site'] / 'endf.db').read_bytes() == before
	assert con.execute('SELECT COUNT(*) FROM YY_1').fetchone()[0] == 2


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


def test_site_file_used_readonly_in_place(sandbox):
	"""A site-wide data file (shared/common install) is used where it is,
	without fetching and without copying into the user directory."""
	sha = _make_db(sandbox['site'] / 'decay.db', 'chart')
	sandbox['registry']['files']['decay.db'] = sha
	con = data._get_connection('decay')
	assert con.execute('SELECT COUNT(*) FROM chart').fetchone()[0] == 2
	assert sandbox['fetched'] == []
	assert not (sandbox['cache'] / 'decay.db').exists()


def test_user_file_preferred_over_site(sandbox):
	_make_db(sandbox['site'] / 'decay.db', 'chart', rows=((9, 9.0),))
	_make_db(sandbox['cache'] / 'decay.db', 'chart')
	con = data._get_connection('decay')
	assert con.execute('SELECT COUNT(*) FROM chart').fetchone()[0] == 2
	assert sandbox['fetched'] == []


def test_failed_overwrite_preserves_existing_db(sandbox, monkeypatch):
	"""A failed fetch must never destroy the existing data file."""
	sha = _make_db(sandbox['cache'] / 'ziegler.db', 'compounds')
	sandbox['registry']['files']['ziegler.db'] = sha
	good = (sandbox['cache'] / 'ziegler.db').read_bytes()

	def failing_retrieve(fnm):
		raise IOError('synthetic network failure')

	monkeypatch.setattr(data, '_retrieve', failing_retrieve)
	data.download('ziegler', overwrite=True)
	assert (sandbox['cache'] / 'ziegler.db').read_bytes() == good


def test_download_prefetches_whole_file(sandbox, capsys):
	sha = _make_db(sandbox['remote'] / 'ziegler.db', 'compounds')
	sandbox['registry']['files']['ziegler.db'] = sha
	data.download('ziegler')
	assert (sandbox['cache'] / 'ziegler.db').exists()
	data.download('ziegler')  # second call skips
	assert sandbox['fetched'] == ['ziegler.db']
	assert 'already installed' in capsys.readouterr().out
