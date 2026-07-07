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


def test_shipped_registry_is_self_consistent():
	"""The shipped registry, the shard naming, and the runtime table derivation
	are a load-bearing three-way coupling: every shard filename must round-trip
	through the '<LETTERS>_<DIGITS>[m]' table name that library.py derives and
	the '<library>_<table>.db' name data.py fetches. Each shard group carries
	its own release base_url (GitHub caps assets per release), and every
	sharded library must also have a whole-file entry (the eager path)."""
	import json
	import re
	with open(os.path.join(os.path.dirname(data.__file__), 'data_registry.json')) as f:
		reg = json.load(f)
	# the base_urls are hand-edited when releases move: pin them exactly, so a
	# typo fails here instead of at a user's first fetch
	assert reg['base_url'] == 'https://github.com/jtmorrell/curie-data/releases/download/v1/'
	hexre = re.compile(r'^[0-9a-f]{64}$')
	assert set(reg['files']) == {'decay.db', 'ziegler.db', 'endf.db', 'tendl.db',
								 'tendl_n_rp.db', 'tendl_p_rp.db', 'tendl_d_rp.db',
								 'IRDFF.db', 'iaea_monitors.db'}
	assert all(hexre.match(h) for h in reg['files'].values())
	assert set(reg['shards']) == {'endf', 'tendl', 'tendl_n_rp', 'tendl_p_rp', 'tendl_d_rp'}
	for lib, group in reg['shards'].items():
		assert lib + '.db' in reg['files'], '{} has shards but no whole-file entry'.format(lib)
		assert group['base_url'] == 'https://github.com/jtmorrell/curie-data/releases/download/v1-{}/'.format(lib.replace('_', '-'))
		shards = group['files']
		assert '{}_all_reactions.db'.format(lib) in shards, lib
		namere = re.compile(r'^{}_([A-Z]+_[0-9]+m?|all_reactions)\.db$'.format(lib))
		bad = [s for s in shards if not namere.match(s)]
		assert not bad, '{} shard names the runtime table derivation cannot produce: {}'.format(lib, bad)
		assert all(hexre.match(h) for h in shards.values()), lib
		assert len(shards) > 400, lib
	# GitHub caps a release at 1000 assets: each group is one release
	assert all(len(g['files']) <= 900 for g in reg['shards'].values()), 'shard group too close to the per-release asset cap: split it'


def test_every_endf_table_has_a_registry_shard():
	"""The other direction of the coupling: every table actually present in
	the local ENDF database must be fetchable as a shard by name."""
	import json
	if not data._db_available('endf'):
		pytest.skip('nuclear data not installed: endf')
	with open(os.path.join(os.path.dirname(data.__file__), 'data_registry.json')) as f:
		shards = json.load(f)['shards']['endf']['files']
	con = data._get_connection('endf')
	tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
	missing = [t for t in tables if 'endf_{}.db'.format(t) not in shards]
	assert not missing, 'tables with no shard in the registry: {}'.format(missing)


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
	sandbox['registry']['shards']['endf'] = {'base_url': 'test://', 'files': {'endf_XX_1.db': sha}}
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
	sandbox['registry']['shards']['endf'] = {'base_url': 'test://', 'files': {'endf_XX_1.db': sha}}
	con = data._get_connection('endf')
	data._ensure_table('endf', 'XX_1')
	assert con.execute('SELECT COUNT(*) FROM XX_1').fetchone()[0] == 2


def test_failed_assembly_leaves_no_partial_table(sandbox):
	"""An assembly that dies mid-way must leave no committed empty table
	(which the existence check would forever treat as complete)."""
	sha = _make_db(sandbox['remote'] / 'endf_XX_1.db', 'WRONG_NAME')
	sandbox['registry']['shards']['endf'] = {'base_url': 'test://', 'files': {'endf_XX_1.db': sha}}
	con = data._get_connection('endf')
	with pytest.raises(Exception):
		data._ensure_table('endf', 'XX_1')
	assert not con.execute("SELECT name FROM sqlite_master WHERE name='XX_1'").fetchone()
	os.remove(str(sandbox['remote'] / 'endf_XX_1.db'))
	sha = _make_db(sandbox['remote'] / 'endf_XX_1.db', 'XX_1')
	data._ensure_table('endf', 'XX_1')  # a corrected shard then assembles fine
	assert con.execute('SELECT COUNT(*) FROM XX_1').fetchone()[0] == 2


class _FailAtInsert:
	"""Connection proxy that dies exactly at the shard INSERT — after the
	CREATE has run — simulating a mid-assembly crash (disk full, kill)."""

	def __init__(self, con):
		self._con = con

	def execute(self, sql, *args):
		if sql.lstrip().startswith('INSERT INTO main.'):
			raise sqlite3.OperationalError('synthetic disk I/O error')
		return self._con.execute(sql, *args)

	def __getattr__(self, name):
		return getattr(self._con, name)


def test_insert_failure_after_create_rolls_back_table(sandbox):
	"""The dangerous half of assembly atomicity: the CREATE succeeds, the
	INSERT then fails, and rollback must remove the empty table (which the
	existence check would otherwise treat as complete forever)."""
	sha = _make_db(sandbox['remote'] / 'endf_XX_1.db', 'XX_1')
	sandbox['registry']['shards']['endf'] = {'base_url': 'test://', 'files': {'endf_XX_1.db': sha}}
	con = data._get_connection('endf')
	key = data._data_path('endf.db')
	data.GLOB_CONNECTIONS_DICT[key] = _FailAtInsert(con)
	with pytest.raises(sqlite3.OperationalError):
		data._ensure_table('endf', 'XX_1')
	assert not con.execute("SELECT name FROM sqlite_master WHERE name='XX_1'").fetchone()
	assert not con.in_transaction
	data.GLOB_CONNECTIONS_DICT[key] = con
	data._ensure_table('endf', 'XX_1')  # the next attempt assembles fine
	assert con.execute('SELECT COUNT(*) FROM XX_1').fetchone()[0] == 2


def test_download_skips_site_provided_file(sandbox, capsys):
	sha = _make_db(sandbox['site'] / 'ziegler.db', 'compounds')
	sandbox['registry']['files']['ziegler.db'] = sha
	data.download('ziegler')
	assert sandbox['fetched'] == []
	assert 'site-wide data directory' in capsys.readouterr().out
	assert not (sandbox['cache'] / 'ziegler.db').exists()
	_make_db(sandbox['remote'] / 'ziegler.db', 'compounds')
	data.download('ziegler', overwrite=True)  # explicit overwrite still fetches
	assert sandbox['fetched'] == ['ziegler.db']


def test_retrieve_uses_per_group_base_url(monkeypatch, tmp_path):
	"""Whole files download from the top-level base_url, shards from their
	library's own release; the right hash must accompany each."""
	import sys
	import types
	calls = []

	def fake_pooch_retrieve(url, known_hash, fname, path):
		calls.append((url, known_hash))
		p = os.path.join(path, fname)
		open(p, 'wb').write(b'x')
		return p

	monkeypatch.setitem(sys.modules, 'pooch', types.SimpleNamespace(retrieve=fake_pooch_retrieve))
	monkeypatch.setenv('CURIE_DATA_DIR', str(tmp_path))
	registry = {'base_url': 'whole://', 'files': {'decay.db': 'a' * 64},
				'shards': {'endf': {'base_url': 'shard://', 'files': {'endf_XX_1.db': 'b' * 64}}}}
	monkeypatch.setattr(data, '_registry', lambda: registry)
	data._retrieve('decay.db')
	data._retrieve('endf_XX_1.db')
	assert calls == [('whole://decay.db', 'a' * 64), ('shard://endf_XX_1.db', 'b' * 64)]


def test_sharded_site_copy_never_assembled_into(sandbox):
	"""A site-provided endf.db is used as-is: a missing table must not
	trigger a shard fetch or any write to the site file."""
	_make_db(sandbox['site'] / 'endf.db', 'YY_1')
	sandbox['registry']['shards']['endf'] = {'base_url': 'test://', 'files': {'endf_XX_1.db': '0' * 64}}
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
	sandbox['registry']['shards']['endf'] = {'base_url': 'test://', 'files': {}}
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
