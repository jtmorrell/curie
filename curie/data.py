"""Nuclear-data acquisition and connection layer.

Curie's databases live in a per-user cache directory (override with the
CURIE_DATA_DIR environment variable) and are fetched on first use from the
GitHub data release recorded in data_registry.json, with SHA256 verification.
The ENDF library is additionally published as per-target shards, so looking up
one reaction downloads ~2 MB instead of the full 748 MB file; shards are
assembled into the local endf.db as they arrive.

Data installed by earlier curie versions (in the package's own data directory)
is adopted into the cache on first use, so existing installations re-download
nothing.
"""

import os
import json
import hashlib
import shutil
import sqlite3

GLOB_CONNECTIONS_DICT = {}

_REGISTRY_CACHE = None

_ALIASES = {
	'decay.db': ['decay'],
	'ziegler.db': ['ziegler'],
	'endf.db': ['endf'],
	'tendl.db': ['tendl'],
	'tendl_n_rp.db': ['tendl_n_rp', 'tendl_nrp', 'tendl_n', 'nrp', 'rpn'],
	'tendl_p_rp.db': ['tendl_p_rp', 'tendl_prp', 'tendl_p', 'prp', 'rpp'],
	'tendl_d_rp.db': ['tendl_d_rp', 'tendl_drp', 'tendl_d', 'drp', 'rpd'],
	'IRDFF.db': ['irdff'],
	'iaea_monitors.db': ['iaea', 'iaea-cpr', 'iaea-monitor', 'cpr', 'iaea_cpr',
						 'iaea_monitor', 'medical', 'iaea-medical', 'iaea_medical',
						 'iaea_monitors'],
}


def _registry():
	global _REGISTRY_CACHE
	if _REGISTRY_CACHE is None:
		with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_registry.json')) as f:
			_REGISTRY_CACHE = json.load(f)
	return _REGISTRY_CACHE


def _canonical(db):
	"""Registry filename for a managed database name, or None for user paths."""
	nm = db.lower().replace('.db', '')
	for fnm, aliases in _ALIASES.items():
		if nm == fnm.lower().replace('.db', '') or nm in aliases:
			return fnm
	return None


def _data_path(db=''):
	"""Local nuclear-data cache directory (CURIE_DATA_DIR overrides)."""
	path = os.environ.get('CURIE_DATA_DIR')
	if not path:
		import pooch
		path = str(pooch.os_cache('curie'))
	if not os.path.isdir(path):
		os.makedirs(path, exist_ok=True)
	return os.path.join(path, db) if db else path


def _legacy_data_path(db=''):
	"""Data directory used by curie < 0.1.0 (inside the installed package)."""
	return os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', db))


def _db_available(db):
	"""True if the database can be opened without a network fetch."""
	fnm = _canonical(db)
	if fnm is None:
		return False
	for path in [_data_path(fnm), _legacy_data_path(fnm)]:
		if os.path.isfile(path) and os.path.getsize(path) > 0:
			return True
	return False


def _sha256(path):
	h = hashlib.sha256()
	with open(path, 'rb') as f:
		for chunk in iter(lambda: f.read(1 << 20), b''):
			h.update(chunk)
	return h.hexdigest()


def _retrieve(fnm):
	"""Download one data-release asset into the cache, verifying its SHA256."""
	import pooch
	reg = _registry()
	known = reg['files'].get(fnm)
	if known is None:
		for group in reg.get('shards', {}).values():
			known = group.get(fnm, known)
	print('Downloading {} from the curie data release...'.format(fnm))
	return pooch.retrieve(url=reg['base_url'] + fnm, known_hash=known,
						  fname=fnm, path=_data_path())


def _adopt_legacy(fnm):
	"""Bring a pre-0.1.0 data file into the cache if its checksum still matches."""
	legacy = _legacy_data_path(fnm)
	if not (os.path.isfile(legacy) and os.path.getsize(legacy) > 0):
		return False
	known = _registry()['files'].get(fnm)
	if known is not None and _sha256(legacy) != known:
		print('Existing {} does not match the data release (stale or corrupted); fetching a fresh copy.'.format(fnm))
		return False
	dest = _data_path(fnm)
	try:
		os.link(legacy, dest)
	except OSError:
		shutil.copy2(legacy, dest)
	print('Adopted existing {} from {}'.format(fnm, os.path.dirname(legacy)))
	return True


def _ensure_file(fnm):
	"""Make a managed database file present in the cache; returns its path.

	Sharded libraries (endf) are not fetched whole here: absent a local copy,
	an empty database is created and populated per-target by _ensure_table.
	"""
	path = _data_path(fnm)
	if os.path.isfile(path) and os.path.getsize(path) > 0:
		return path
	if _adopt_legacy(fnm):
		return path
	if fnm[:-3] in _registry().get('shards', {}):
		return path  # assembled lazily, shard by shard
	_retrieve(fnm)
	return path


def _ensure_table(db, table):
	"""Make `table` queryable in a managed database, fetching its shard if needed.

	No-op for whole-file databases and for unknown table names (the caller's
	own SQL then reports the missing table).
	"""
	fnm = _canonical(db)
	if fnm is None:
		return
	group = _registry().get('shards', {}).get(fnm[:-3])
	if group is None:
		return
	con = _get_connection(db)
	if con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
		return
	shard_fnm = '{}_{}.db'.format(fnm[:-3], table)
	if shard_fnm not in group:
		return
	shard_path = _retrieve(shard_fnm)
	con.execute("ATTACH DATABASE ? AS shard", (shard_path,))
	try:
		schema = con.execute("SELECT sql FROM shard.sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
		con.execute(schema)
		con.execute('INSERT INTO "{0}" SELECT * FROM shard."{0}"'.format(table))
		con.commit()
	finally:
		con.execute("DETACH DATABASE shard")
	os.remove(shard_path)


def download(db='all', overwrite=False):
	"""Download nuclear data files as sqlite .db files.

	Data files are fetched automatically the first time they are needed, so
	calling this function is only necessary to prepare an offline machine, to
	pre-populate a fresh cache, or to repair/update existing data with
	`overwrite=True`.  Files are stored in a per-user cache directory
	(printed by `ci.data._data_path()`), which can be overridden with the
	`CURIE_DATA_DIR` environment variable.  Every download is verified against
	the SHA256 recorded for the curie data release.

	Note that the ENDF library is normally assembled on demand from small
	per-target files; downloading it here fetches the complete library in one
	file.

	Parameters
	----------
	db : str, optional
		Name of database to download, default is 'all'.  Options include:
		'all', 'decay', 'ziegler', 'endf', 'tendl', 'tendl_n_rp', 'tendl_p_rp',
		'tendl_d_rp', 'IRDFF', 'iaea_monitors'.

	overwrite : bool, optional
		If overwrite is `True`, will save write over existing data.  Default is `False`.

	Examples
	--------
	The most common use case will be to (re)download all the data files

	>>> ci.download(overwrite=True)

	Some other use cases:
	To update only the 'endf' library

	>>> ci.download('endf', True)

	Or to download the 'decay' library for the first time

	>>> ci.download('decay')

	"""

	if db.lower() in ['all', '*']:
		d = ['decay.db', 'ziegler.db', 'endf.db', 'tendl.db', 'tendl_n_rp.db',
			 'tendl_p_rp.db', 'tendl_d_rp.db', 'IRDFF.db', 'iaea_monitors.db']
	else:
		fnm = _canonical(db)
		if fnm is None:
			print('db={} not recognized.'.format(db))
			return
		d = [fnm]

	for fnm in d:
		path = _data_path(fnm)
		installed = os.path.isfile(path) and os.path.getsize(path) > 0
		if installed and fnm[:-3] in _registry().get('shards', {}):
			# a partially assembled library is present, not installed
			installed = _sha256(path) == _registry()['files'].get(fnm)
		if installed and not overwrite:
			print("{0} already installed. Run ci.download('{0}', overwrite=True) to overwrite these files.".format(fnm.replace('.db', '')))
			continue
		if os.path.isfile(path) and (overwrite or not installed):
			if fnm in GLOB_CONNECTIONS_DICT or path in GLOB_CONNECTIONS_DICT:
				GLOB_CONNECTIONS_DICT.pop(fnm, None)
				GLOB_CONNECTIONS_DICT.pop(path, None)
			os.remove(path)
		try:
			_retrieve(fnm)
		except Exception as e:
			print(e)


def _get_connection(db='decay'):

	def connector(dbnm):
		if os.path.exists(dbnm):
			try:
				if os.path.getsize(dbnm) > 0:
					return sqlite3.connect(dbnm)
				else:
					raise ValueError('{} exists but is of zero size.'.format(dbnm))
			except:
				print('Error connecting to {}.'.format(dbnm))
				if _data_path() in dbnm:
					print('Try using ci.download("all", overwrite=True) to update nuclear data files.')
				raise
		else:
			print('WARNING: database {} does not exist, creating new file.'.format(dbnm))
			return sqlite3.connect(dbnm)

	global GLOB_CONNECTIONS_DICT

	fnm = _canonical(db)
	if fnm is not None:
		path = _ensure_file(fnm)
		if path not in GLOB_CONNECTIONS_DICT:
			# sharded libraries legitimately start as a new empty file
			GLOB_CONNECTIONS_DICT[path] = sqlite3.connect(path) if not os.path.exists(path) else connector(path)
		return GLOB_CONNECTIONS_DICT[path]

	else:
		if db not in GLOB_CONNECTIONS_DICT:
			GLOB_CONNECTIONS_DICT[db] = connector(db)
		return GLOB_CONNECTIONS_DICT[db]


def _get_cursor(db='decay'):
	conn = _get_connection(db)
	return conn.cursor()
